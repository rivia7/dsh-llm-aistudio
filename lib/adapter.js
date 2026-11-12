/**
 * The DSH `LlmAdapter` for the AI Studio route: catalog, exact-model metadata
 * and streaming over `streamGenerateContent?alt=sse` with plain `fetch`
 * (which the dsh launcher already routes through HTTPS_PROXY when set).
 * @module dsh-llm-aistudio/adapter
 */
import { LlmAdapter, LlmError, attributionHeaders, resolveRetryPolicy } from '@deepseek-ai/dsh-llm';
import { EFFORT_NAMES, unlistedModel } from './catalog.js';
import { buildRequest } from './convert.js';
import { httpError } from './errors.js';
import { sseEvents } from './sse.js';
import { translateStream } from './stream.js';

/**
 * Transient failures re-asked by dsh-llm-retry with the identical request.
 * MALFORMED_RESPONSE covers Gemini finish reasons such as MALFORMED_FUNCTION_CALL,
 * MALFORMED_RESPONSE and any reason newer than this file.
 */
export const RETRYABLE_CODES = Object.freeze(['EMPTY_RESPONSE', 'RATE_LIMIT', 'SERVER', 'TIMEOUT', 'TRANSPORT', 'MALFORMED_RESPONSE']);
const RETRY_POLICY = resolveRetryPolicy({
  mode: 'normal',
  maxRetries: 4,
  retryableCodes: [...RETRYABLE_CODES],
  backoff: { initialDelayMs: 1_000, maxDelayMs: 20_000, jitterRatio: 0.15 },
}, 'llm-aistudio.retryPolicy');

/** Encoded-byte target for one request image (before base64). */
const REQUEST_IMAGE_MAX_BYTES = 4 * 1024 * 1024;

function describe(error) {
  if (error instanceof Error) {
    const cause = error.cause instanceof Error ? `: ${error.cause.code ?? error.cause.message}` : '';
    return `${error.name}: ${error.message}${cause}`;
  }
  return String(error);
}

/** AI Studio (Gemini API) adapter. */
export class AiStudioAdapter extends LlmAdapter {
  #options;

  /**
   * @param {object} options
   * @param {string} options.displayName - provider group name.
   * @param {() => readonly object[]} options.models - normalized models.
   * @param {() => string} options.baseURL - API base URL without a trailing slash.
   * @param {() => boolean} options.includeThoughts - request thought summaries.
   * @param {() => number} options.idleTimeoutMs - stream idle limit.
   * @param {() => Promise<string>} options.apiKey - resolve the key per request.
   * @param {() => boolean} options.canReadImages - whether the attachment service is mounted.
   * @param {(ref: object, target: object, signal?: AbortSignal) => Promise<object>} options.readImage - request-version reader.
   * @param {typeof fetch} [options.fetch] - transport, for tests.
   */
  constructor(options) {
    super();
    this.#options = options;
  }

  providerInfo(provider) {
    return { id: provider, name: this.#options.displayName };
  }

  providerRetryPolicy(_provider) {
    return RETRY_POLICY;
  }

  imageRequestTarget(_provider, _model, image) {
    return { width: image.width, height: image.height, maxBytes: Math.min(image.bytes, REQUEST_IMAGE_MAX_BYTES) };
  }

  #model(id) {
    const bare = String(id).replace(/^models\//, '');
    return this.#options.models().find((model) => model.id === bare) ?? unlistedModel(bare);
  }

  #modalities(model) {
    return model.input.filter((modality) => modality !== 'image' || this.#options.canReadImages());
  }

  listModels(provider) {
    return Promise.resolve(this.#options.models().map((model) => ({
      provider,
      id: model.id,
      name: model.name,
      description: 'Google AI Studio (Gemini API key)',
      inputModalities: this.#modalities(model),
    })));
  }

  resolveModel(provider, modelId) {
    const model = this.#model(modelId);
    return Promise.resolve({
      provider,
      id: modelId,
      name: model.name,
      inputModalities: this.#modalities(model),
      context: { contextWindow: model.contextWindow },
      ...(model.efforts.length > 0 ? {
        reasoning: {
          efforts: model.efforts.map((effort) => ({ id: effort.id, name: EFFORT_NAMES[effort.id] ?? effort.id })),
          ...(model.defaultEffort === undefined ? {} : { defaultEffort: model.defaultEffort }),
        },
      } : {}),
    });
  }

  async *stream(options) {
    const settings = this.#options;
    const model = this.#model(options.model);
    const apiKey = await settings.apiKey();
    const body = await buildRequest({
      options,
      model,
      route: options.provider,
      includeThoughts: settings.includeThoughts(),
      imagesAllowed: this.#modalities(model).includes('image'),
      readImage: (ref, signal) => settings.readImage(ref, this.imageRequestTarget(options.provider, model.id, ref), signal),
    });
    const url = `${settings.baseURL()}/models/${encodeURIComponent(model.id)}:streamGenerateContent?alt=sse`;

    const controller = new AbortController();
    const caller = options.signal;
    const onAbort = () => controller.abort(caller.reason);
    if (caller?.aborted) controller.abort(caller.reason);
    caller?.addEventListener('abort', onAbort, { once: true });
    const idleMs = settings.idleTimeoutMs();
    let timedOut = false;
    let idleTimer;
    const arm = () => {
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        timedOut = true;
        controller.abort(new Error(`no data for ${idleMs} ms`));
      }, idleMs);
    };
    const transportError = (error) => {
      if (caller?.aborted || error instanceof LlmError) return error;
      if (timedOut) return new LlmError(`aistudio: Gemini sent no data for ${Math.round(idleMs / 1000)} s`, 'TIMEOUT', { cause: error });
      return new LlmError(`aistudio: network error talking to Gemini (${describe(error)})`, 'TRANSPORT', { cause: error });
    };

    arm();
    try {
      let response;
      try {
        response = await (settings.fetch ?? fetch)(url, {
          method: 'POST',
          headers: {
            ...attributionHeaders(),
            'content-type': 'application/json',
            accept: 'text/event-stream',
            'x-goog-api-key': apiKey,
          },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
      } catch (error) {
        throw transportError(error);
      }
      if (!response.ok) throw await httpError(response);
      if (response.body === null) throw new LlmError('aistudio: Gemini response has no body', 'TRANSPORT');
      arm();
      try {
        yield* translateStream(sseEvents(response.body, arm, controller.signal), { model: model.id });
      } catch (error) {
        throw transportError(error);
      }
    } finally {
      clearTimeout(idleTimer);
      caller?.removeEventListener('abort', onAbort);
    }
  }
}
