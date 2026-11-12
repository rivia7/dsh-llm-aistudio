/**
 * Gemini `GenerateContentResponse` stream -> harness `StreamChunk`s.
 *
 * Every Gemini finish reason maps to a harness outcome, including ones this
 * file has never heard of: an unknown reason becomes a retryable
 * `MALFORMED_RESPONSE` failure instead of an unhandled exception.
 * @module dsh-llm-aistudio/stream
 */
import { randomUUID } from 'node:crypto';
import { EMPTY_RESPONSE_CODE, LlmError } from '@deepseek-ai/dsh-llm';
import { classifyGoogleError } from './errors.js';

/** Replay envelope tag; bump the version when the block metadata shape changes. */
export const REPLAY_KIND = 'aistudio';
export const REPLAY_VERSION = 1;

/** Finish reasons where Google withheld or cut the content: not retried. */
const CONTENT_FILTER_REASONS = new Set([
  'SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'LANGUAGE',
  'IMAGE_SAFETY', 'IMAGE_PROHIBITED_CONTENT', 'IMAGE_RECITATION', 'IMAGE_OTHER', 'NO_IMAGE',
]);

/**
 * Harness usage from Gemini `usageMetadata`. Counts are disjoint: cached
 * prompt tokens are reported apart from uncached input, and thoughts count
 * as output, as Google bills them.
 * @param {object} usage - usageMetadata.
 * @returns {object} TokenUsage.
 */
export function mapUsage(usage) {
  const prompt = (usage.promptTokenCount ?? 0) + (usage.toolUsePromptTokenCount ?? 0);
  const cached = usage.cachedContentTokenCount ?? 0;
  const thoughts = usage.thoughtsTokenCount ?? 0;
  return {
    inputTokens: Math.max(0, prompt - cached),
    outputTokens: (usage.candidatesTokenCount ?? 0) + thoughts,
    ...(Number.isInteger(usage.totalTokenCount) ? { totalTokens: usage.totalTokenCount } : {}),
    ...(cached > 0 ? { cacheReadTokens: cached } : {}),
    ...(thoughts > 0 ? { reasoningTokens: thoughts } : {}),
  };
}

/** @returns {string} a fresh tool-call id in Gemini's accepted alphabet. */
function newCallId() {
  return `call_${randomUUID().replaceAll('-', '').slice(0, 24)}`;
}

/**
 * Harness failure for a terminal non-success finish reason.
 * @param {string} reason - Gemini finishReason.
 * @param {string | undefined} message - Gemini finishMessage.
 * @returns {LlmError} the failure.
 */
export function finishFailure(reason, message) {
  const suffix = message ? `: ${message}` : '';
  if (CONTENT_FILTER_REASONS.has(reason)) {
    return new LlmError(`aistudio: Gemini stopped with finishReason ${reason}${suffix}`, 'CONTENT_FILTERED');
  }
  return new LlmError(`aistudio: Gemini stopped with finishReason ${reason}${suffix}`, 'MALFORMED_RESPONSE');
}

/**
 * Translate parsed SSE events into harness chunks.
 * @param {AsyncIterable<string>} events - raw SSE `data` payloads.
 * @param {{ model: string }} context - requested model id, recorded in the replay envelope.
 * @returns {AsyncGenerator<object>} StreamChunks ending with usage then finish; throws LlmError on failures.
 */
export async function* translateStream(events, context) {
  let nextIndex = 0;
  /** @type {{ index: number, kind: 'text' | 'reasoning', text: string, signature?: string } | undefined} */
  let open;
  const replayBlocks = [];
  let finishReason;
  let finishMessage;
  let usage;
  let responseId;
  let modelVersion;
  let blockReason;
  let sawToolCall = false;
  let sawVisibleText = false;

  function* closeOpen() {
    if (open === undefined) return;
    yield { type: 'block-end', index: open.index, block: { type: open.kind, text: open.text } };
    replayBlocks.push(open.signature === undefined ? {} : { thoughtSignature: open.signature });
    open = undefined;
  }
  function* openBlock(kind) {
    yield* closeOpen();
    open = { index: nextIndex++, kind, text: '' };
    yield { type: 'block-start', index: open.index, blockType: kind };
  }

  for await (const data of events) {
    let payload;
    try {
      payload = JSON.parse(data);
    } catch {
      throw new LlmError(`aistudio: unparseable Gemini stream event: ${data.slice(0, 200)}`, 'TRANSPORT');
    }
    const item = Array.isArray(payload) ? payload[0] : payload;
    if (item === null || typeof item !== 'object') continue;
    if (item.error) throw classifyGoogleError(item.error.code, item.error);
    if (item.usageMetadata) usage = item.usageMetadata;
    responseId ??= item.responseId;
    modelVersion ??= item.modelVersion;
    if (item.promptFeedback?.blockReason) blockReason = item.promptFeedback.blockReason;

    const candidate = item.candidates?.[0];
    if (candidate === undefined) continue;
    for (const part of candidate.content?.parts ?? []) {
      if (part.functionCall) {
        yield* closeOpen();
        const index = nextIndex++;
        const id = typeof part.functionCall.id === 'string' && /^[A-Za-z0-9_-]{1,64}$/.test(part.functionCall.id)
          ? part.functionCall.id
          : newCallId();
        const name = String(part.functionCall.name ?? '');
        const args = JSON.stringify(part.functionCall.args ?? {});
        yield { type: 'block-start', index, blockType: 'tool-call' };
        yield { type: 'tool-call-delta', index, id, name, argumentsDelta: args };
        yield { type: 'block-end', index, block: { type: 'tool-call', id, name, arguments: args } };
        replayBlocks.push(typeof part.thoughtSignature === 'string' ? { thoughtSignature: part.thoughtSignature } : {});
        sawToolCall = true;
        continue;
      }
      if (typeof part.text === 'string') {
        const kind = part.thought === true ? 'reasoning' : 'text';
        const signed = typeof part.thoughtSignature === 'string' && part.thoughtSignature.length > 0;
        // An empty, unsigned part carries nothing worth a block.
        if (part.text.length === 0 && !signed) continue;
        if (open === undefined || open.kind !== kind) yield* openBlock(kind);
        if (part.text.length > 0) {
          open.text += part.text;
          yield { type: kind === 'reasoning' ? 'reasoning-delta' : 'text-delta', index: open.index, text: part.text };
          if (kind === 'text' && part.text.trim().length > 0) sawVisibleText = true;
        }
        if (typeof part.thoughtSignature === 'string' && part.thoughtSignature.length > 0) open.signature = part.thoughtSignature;
        continue;
      }
      if (typeof part.thoughtSignature === 'string' && part.thoughtSignature.length > 0) {
        // A signature-only part: keep it on a block so it is echoed back next turn.
        if (open === undefined) yield* openBlock('text');
        open.signature = part.thoughtSignature;
      }
      // inlineData / executableCode / codeExecutionResult are not requested and are ignored.
    }
    if (candidate.finishReason) {
      finishReason = candidate.finishReason;
      finishMessage = candidate.finishMessage;
    }
  }
  yield* closeOpen();
  if (usage !== undefined) yield { type: 'usage', usage: mapUsage(usage) };

  if (finishReason === undefined) {
    if (blockReason !== undefined) {
      throw new LlmError(`aistudio: Gemini blocked the prompt (blockReason ${blockReason})`, 'CONTENT_FILTERED');
    }
    throw new LlmError('aistudio: Gemini stream ended without a finish reason', 'TRANSPORT');
  }
  let reason;
  if (finishReason === 'STOP') {
    if (sawToolCall) reason = { kind: 'tool-calls' };
    else if (sawVisibleText) reason = { kind: 'stop' };
    else throw new LlmError(`aistudio: model "${context.model}" finished (STOP) without visible output`, EMPTY_RESPONSE_CODE);
  } else if (finishReason === 'MAX_TOKENS') {
    reason = { kind: 'max-tokens' };
  } else {
    throw finishFailure(finishReason, finishMessage);
  }
  yield {
    type: 'finish',
    reason,
    replayState: {
      response: {
        kind: REPLAY_KIND,
        version: REPLAY_VERSION,
        model: context.model,
        ...(responseId === undefined ? {} : { responseId }),
        ...(modelVersion === undefined ? {} : { modelVersion }),
        finishReason,
      },
      blocks: replayBlocks,
    },
  };
}
