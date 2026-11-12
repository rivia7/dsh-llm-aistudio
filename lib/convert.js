/**
 * Harness request -> Gemini `GenerateContentRequest`.
 *
 * Wire rules this module owns (https://ai.google.dev/gemini-api/docs/thought-signatures):
 * - The leading system message(s) become `systemInstruction`; a later system
 *   message becomes a `<system-reminder>` user turn, kept after any pending
 *   function responses so a call and its responses stay adjacent.
 * - Thought signatures are echoed back on exactly the part they arrived on,
 *   and only to the same route and model; an empty text part that carries a
 *   signature is still sent.
 * - Gemini 3+ validates signatures for function calls in the current turn, so
 *   a foreign or signature-less call there gets Google's documented bypass value.
 * - Gemini 3+ needs explicit function call ids and accepts images inside
 *   function responses.
 * @module dsh-llm-aistudio/convert
 */
import {
  IMAGE_OFFLOAD_REQUIRED_CODE,
  LlmError,
  offloadedImageText,
  projectOffloadedImages,
  requestImageHandleText,
  requiredImageOffload,
  textOnlyImageText,
} from '@deepseek-ai/dsh-llm';
import { REPLAY_KIND, REPLAY_VERSION } from './stream.js';

/** Inline image budget, as base64 length; Gemini caps a request body at 20 MB. */
export const MAX_REQUEST_IMAGE_BASE64_BYTES = 14 * 1024 * 1024;
export const MAX_REQUEST_IMAGES = 100;
const IMAGE_COUNT_QUANTUM = 10;

/** Google's documented value for a function call that has no genuine signature. */
export const BYPASS_SIGNATURE = 'skip_thought_signature_validator';

const BASE64 = /^[A-Za-z0-9+/_-]+={0,2}$/;

/**
 * @param {string} modelId - Gemini model id.
 * @returns {number | undefined} the Gemini major version, when the id names one.
 */
export function geminiMajor(modelId) {
  const match = /^gemini(?:-live)?-(\d+)/i.exec(modelId);
  return match ? Number.parseInt(match[1], 10) : undefined;
}

/** @param {string} modelId @returns {boolean} whether the model is Gemini 3 or later. */
export function isGemini3Plus(modelId) {
  const major = geminiMajor(modelId);
  return major !== undefined && major >= 3;
}

/**
 * A tool-call id in the alphabet Gemini accepts for function call ids.
 * @param {string} id - harness tool-call id.
 * @returns {string} the wire id.
 */
export function wireCallId(id) {
  return String(id).replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 64);
}

/** @param {unknown} text @returns {string} well-formed UTF-16 (lone surrogates replaced). */
function clean(text) {
  return typeof text === 'string' ? text.toWellFormed() : '';
}

/** @param {readonly object[]} content @returns {string} the text blocks joined by newlines. */
function flattenText(content) {
  return content.filter((block) => block.type === 'text').map((block) => clean(block.text)).join('\n');
}

/** @param {unknown} signature @returns {boolean} whether Google could accept it as bytes. */
function validSignature(signature) {
  return typeof signature === 'string' && signature.length > 0 && BASE64.test(signature);
}

/**
 * Per-block replay metadata of one assistant message, when it came from this
 * route and the requested model and the envelope still aligns with the content.
 * @param {object} message - assistant message.
 * @param {string} route - this provider route.
 * @param {string} modelId - requested model.
 * @returns {readonly object[] | undefined} replay blocks, aligned with `message.content`.
 */
export function replayBlocksFor(message, route, modelId) {
  const source = message.source;
  if (source?.provider !== route || source?.model !== modelId) return undefined;
  const state = source.replayState;
  if (state?.response?.kind !== REPLAY_KIND || state.response.version !== REPLAY_VERSION) return undefined;
  if (!Array.isArray(state.blocks) || state.blocks.length !== message.content.length) return undefined;
  return state.blocks;
}

const SCHEMA_NAME_MAPS = new Set(['properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas', 'dependencies']);
const SCHEMA_SUBSCHEMAS = new Set([
  'additionalProperties', 'unevaluatedProperties', 'propertyNames', 'items', 'prefixItems', 'additionalItems',
  'unevaluatedItems', 'contains', 'allOf', 'anyOf', 'oneOf', 'not', 'if', 'then', 'else', 'contentSchema',
]);

/**
 * Rewrite string/number `const` to a one-value `enum`: Gemini ignores `const`
 * in `parametersJsonSchema` but honors `enum`.
 * @param {unknown} schema - JSON Schema (sub)tree.
 * @param {boolean} [isNameMap] - whether keys are property names, not keywords.
 * @returns {unknown} the rewritten copy.
 */
export function rewriteConstAsEnum(schema, isNameMap = false) {
  if (Array.isArray(schema)) return schema.map((item) => rewriteConstAsEnum(item));
  if (schema === null || typeof schema !== 'object') return schema;
  const result = Object.fromEntries(Object.entries(schema).map(([key, value]) => {
    if (isNameMap || SCHEMA_SUBSCHEMAS.has(key)) return [key, rewriteConstAsEnum(value)];
    if (SCHEMA_NAME_MAPS.has(key)) return [key, rewriteConstAsEnum(value, true)];
    return [key, value];
  }));
  if (!isNameMap && 'const' in result && result.enum === undefined
    && (typeof result.const === 'string' || typeof result.const === 'number')) {
    result.enum = [result.const];
    delete result.const;
  }
  return result;
}

/**
 * Gemini function declarations for the request's tools.
 * @param {readonly object[] | undefined} tools - harness ToolSchemas.
 * @returns {object[] | undefined} the `tools` field.
 */
export function toolDeclarations(tools) {
  if (!tools || tools.length === 0) return undefined;
  return [{
    functionDeclarations: tools.map((tool) => {
      const parameters = tool.parameters && Object.keys(tool.parameters).length > 0
        ? tool.parameters
        : { type: 'object', properties: {} };
      return { name: tool.name, description: clean(tool.description), parametersJsonSchema: rewriteConstAsEnum(parameters) };
    }),
  }];
}

/**
 * The `thinkingConfig` for one call.
 * @param {object} model - normalized model.
 * @param {string | undefined} effortId - requested effort.
 * @param {boolean} includeThoughts - whether to request thought summaries.
 * @returns {object | undefined} thinkingConfig, or undefined for a model without reasoning control.
 */
export function thinkingConfig(model, effortId, includeThoughts) {
  if (model.efforts.length === 0) return undefined;
  const chosen = model.efforts.find((effort) => effort.id === effortId)
    ?? model.efforts.find((effort) => effort.id === model.defaultEffort)
    ?? model.efforts.at(-1);
  if (typeof chosen.wire === 'number') {
    return { thinkingBudget: chosen.wire, includeThoughts: includeThoughts && chosen.wire !== 0 };
  }
  return { thinkingLevel: chosen.wire, includeThoughts };
}

/**
 * Give the first function call of every model turn in the current turn a
 * signature: Gemini 3+ rejects one without, and a genuine signature is only
 * ever missing there for history this route did not produce.
 * @param {object[]} contents - Gemini contents, edited in place.
 */
export function fillCurrentTurnSignatures(contents) {
  let start = 0;
  contents.forEach((content, index) => {
    if (content.role === 'user' && content.parts.some((part) => !part.functionResponse)) start = index + 1;
  });
  for (let index = start; index < contents.length; index++) {
    const content = contents[index];
    if (content.role !== 'model') continue;
    const first = content.parts.find((part) => part.functionCall);
    if (first !== undefined && first.thoughtSignature === undefined) first.thoughtSignature = BYPASS_SIGNATURE;
  }
}

/**
 * Build the Gemini request body for one harness call.
 * @param {object} input
 * @param {object} input.options - GenerateOptions after runtime projection.
 * @param {object} input.model - normalized model.
 * @param {string} input.route - this provider route.
 * @param {boolean} input.includeThoughts - request thought summaries.
 * @param {boolean} input.imagesAllowed - whether image blocks are sent as images.
 * @param {(ref: object, signal?: AbortSignal) => Promise<object>} [input.readImage] - request-version reader.
 * @returns {Promise<object>} the request body.
 */
export async function buildRequest({ options, model, route, includeThoughts, imagesAllowed, readImage }) {
  const messages = projectOffloadedImages(options.messages, (ref) => offloadedImageText(ref));

  const systemParts = [];
  if (options.system) systemParts.push(clean(options.system));
  let first = 0;
  for (; first < messages.length && messages[first].role === 'system'; first++) {
    const text = flattenText(messages[first].content);
    if (text.trim().length > 0) systemParts.push(text);
  }
  const history = messages.slice(first);

  // Request-version bytes for every retained image, then the request budget.
  const versions = new Map();
  if (imagesAllowed && readImage !== undefined) {
    const blocks = history.flatMap((message) => (message.role === 'user' || message.role === 'tool')
      ? message.content.filter((block) => block.type === 'image' && !block.offloaded)
      : []);
    for (const block of blocks) {
      const id = block.attachment.attachmentId;
      if (!versions.has(id)) {
        const version = await readImage(block.attachment, options.signal);
        versions.set(id, { version, base64: Buffer.from(version.data).toString('base64') });
      }
    }
    if (blocks.length > 0) {
      const offload = requiredImageOffload(messages, {
        representation: 'base64',
        maxBytes: MAX_REQUEST_IMAGE_BASE64_BYTES,
        maxImages: MAX_REQUEST_IMAGES,
        countQuantum: IMAGE_COUNT_QUANTUM,
      }, (block) => versions.get(block.attachment.attachmentId)?.version.bytes ?? block.attachment.bytes);
      if (offload > 0) {
        throw new LlmError(`aistudio: request images exceed the Gemini request budget; offload the ${offload} oldest`, IMAGE_OFFLOAD_REQUIRED_CODE, { offloadImages: offload });
      }
    }
  }
  const imageHandle = (block) => {
    const read = versions.get(block.attachment.attachmentId);
    return read === undefined
      ? textOnlyImageText(block.attachment)
      : requestImageHandleText(block.attachment, { width: read.version.width, height: read.version.height });
  };
  const inlineImage = (block) => {
    const read = versions.get(block.attachment.attachmentId);
    return read === undefined ? undefined : { inlineData: { mimeType: read.version.mediaType, data: read.base64 } };
  };

  const toolNames = new Map();
  for (const message of history) {
    if (message.role !== 'assistant') continue;
    for (const block of message.content) if (block.type === 'tool-call') toolNames.set(block.id, block.name);
  }

  const callIds = isGemini3Plus(model.id);
  const multimodalResponses = geminiMajor(model.id) === undefined || isGemini3Plus(model.id);
  const contents = [];
  const pendingCalls = new Set();
  let deferredReminders = [];
  const flushReminders = () => {
    if (deferredReminders.length === 0) return;
    contents.push({ role: 'user', parts: deferredReminders.map((text) => ({ text })) });
    deferredReminders = [];
  };

  for (const message of history) {
    switch (message.role) {
      case 'system': {
        const text = flattenText(message.content);
        if (text.trim().length === 0) break;
        deferredReminders.push(`<system-reminder>\n${text.replaceAll('</system-reminder>', '<\\/system-reminder>')}\n</system-reminder>`);
        if (pendingCalls.size === 0) flushReminders();
        break;
      }
      case 'user': {
        pendingCalls.clear();
        flushReminders();
        const parts = [];
        for (const block of message.content) {
          if (block.type === 'text') {
            if (block.text.length > 0) parts.push({ text: clean(block.text) });
          } else if (block.type === 'image') {
            parts.push({ text: imageHandle(block) });
            const inline = inlineImage(block);
            if (inline !== undefined) parts.push(inline);
          }
        }
        if (parts.length > 0) contents.push({ role: 'user', parts });
        break;
      }
      case 'assistant': {
        pendingCalls.clear();
        flushReminders();
        const replay = replayBlocksFor(message, route, model.id);
        const parts = [];
        message.content.forEach((block, index) => {
          const raw = replay?.[index]?.thoughtSignature;
          const signature = validSignature(raw) ? raw : undefined;
          const signed = signature === undefined ? {} : { thoughtSignature: signature };
          if (block.type === 'text') {
            const text = clean(block.text);
            if (text.trim().length > 0 || signature !== undefined) parts.push({ text, ...signed });
          } else if (block.type === 'reasoning') {
            // Another model's thoughts are not Gemini thoughts; only this route's are replayed.
            if (replay === undefined) return;
            const text = clean(block.text);
            if (text.trim().length > 0 || signature !== undefined) parts.push({ thought: true, text, ...signed });
          } else if (block.type === 'tool-call') {
            let args;
            try {
              args = JSON.parse(block.arguments || '{}');
            } catch {
              args = {};
            }
            if (args === null || typeof args !== 'object' || Array.isArray(args)) args = { value: args };
            parts.push({ functionCall: { name: block.name, args, ...(callIds ? { id: wireCallId(block.id) } : {}) }, ...signed });
            pendingCalls.add(block.id);
          }
        });
        if (parts.length > 0) contents.push({ role: 'model', parts });
        break;
      }
      case 'tool': {
        pendingCalls.delete(message.toolCallId);
        const texts = [];
        const inline = [];
        for (const block of message.content) {
          if (block.type === 'text') {
            texts.push(clean(block.text));
          } else if (block.type === 'image') {
            texts.push(imageHandle(block));
            const part = inlineImage(block);
            if (part !== undefined) inline.push(part);
          }
        }
        let output = texts.join('\n');
        if (output.trim().length === 0 && inline.length > 0) output = '(see attached image)';
        const responsePart = {
          functionResponse: {
            name: toolNames.get(message.toolCallId) ?? 'unknown_tool',
            response: message.isError ? { error: output } : { output },
            ...(inline.length > 0 && multimodalResponses ? { parts: inline } : {}),
            ...(callIds ? { id: wireCallId(message.toolCallId) } : {}),
          },
        };
        const last = contents.at(-1);
        if (last?.role === 'user' && last.parts.some((part) => part.functionResponse)) last.parts.push(responsePart);
        else contents.push({ role: 'user', parts: [responsePart] });
        if (inline.length > 0 && !multimodalResponses) contents.push({ role: 'user', parts: [{ text: 'Tool result image:' }, ...inline] });
        if (pendingCalls.size === 0) flushReminders();
        break;
      }
      default:
        // developer messages carry tool updates this route does not declare support for.
        break;
    }
  }
  flushReminders();
  if (callIds) fillCurrentTurnSignatures(contents);

  const generationConfig = {};
  if (options.maxTokens !== undefined) generationConfig.maxOutputTokens = Math.min(options.maxTokens, model.maxTokens);
  if (options.temperature !== undefined) generationConfig.temperature = options.temperature;
  if (options.stop?.length) generationConfig.stopSequences = options.stop.slice(0, 5);
  const thinking = thinkingConfig(model, options.reasoningEffort, includeThoughts);
  if (thinking !== undefined) generationConfig.thinkingConfig = thinking;

  const tools = toolDeclarations(options.tools);
  return {
    contents,
    ...(systemParts.length > 0 ? { systemInstruction: { parts: [{ text: systemParts.join('\n\n') }] } } : {}),
    ...(tools === undefined ? {} : { tools }),
    ...(Object.keys(generationConfig).length > 0 ? { generationConfig } : {}),
  };
}
