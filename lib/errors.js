/**
 * Gemini API failures mapped to stable harness codes, so the retry layer and
 * compaction route on the code and never parse provider text.
 *
 * | Gemini answer                                   | code                      | retried |
 * |-------------------------------------------------|---------------------------|---------|
 * | 400 + ErrorInfo API_KEY_INVALID                 | INVALID_CREDENTIAL        | no      |
 * | 400 input token count exceeds the maximum       | CONTEXT_WINDOW_EXCEEDED   | no (compaction) |
 * | 400 other / 404                                 | INVALID_REQUEST / UNKNOWN_MODEL | no |
 * | 401 / 403                                       | AUTH                      | no      |
 * | 429 per-day quota (QuotaFailure ...PerDay...)   | QUOTA                     | no      |
 * | 429 other                                       | RATE_LIMIT (+retryDelay)  | yes     |
 * | 5xx                                             | SERVER                    | yes     |
 * @module dsh-llm-aistudio/errors
 */
import {
  CONTEXT_WINDOW_EXCEEDED_CODE,
  INVALID_CREDENTIAL_CODE,
  LlmError,
  QUOTA_EXCEEDED_CODE,
  isContextWindowExceededError,
} from '@deepseek-ai/dsh-llm';

const MAX_ERROR_BODY_CHARS = 64 * 1024;
const GEMINI_CONTEXT_OVERFLOW = /input token count[^]*?exceeds the maximum number of tokens/i;

/**
 * Parse a protobuf Duration as Google renders it ("31s", "0.5s").
 * @param {unknown} value - RetryInfo.retryDelay.
 * @returns {number | undefined} milliseconds, when positive.
 */
export function parseDurationMs(value) {
  if (typeof value !== 'string') return undefined;
  const match = /^(\d+(?:\.\d+)?)s$/.exec(value.trim());
  if (!match) return undefined;
  const ms = Math.ceil(Number(match[1]) * 1000);
  return ms > 0 ? ms : undefined;
}

/**
 * Parse a Retry-After header (delta seconds or an HTTP date).
 * @param {Headers | undefined} headers - response headers.
 * @returns {number | undefined} milliseconds, when positive.
 */
export function retryAfterHeaderMs(headers) {
  const raw = headers?.get?.('retry-after');
  if (!raw) return undefined;
  const seconds = Number(raw);
  if (Number.isFinite(seconds)) return seconds > 0 ? Math.ceil(seconds * 1000) : undefined;
  const at = Date.parse(raw);
  if (Number.isNaN(at)) return undefined;
  const ms = at - Date.now();
  return ms > 0 ? ms : undefined;
}

function detail(details, typeSuffix) {
  return Array.isArray(details) ? details.find((d) => typeof d?.['@type'] === 'string' && d['@type'].endsWith(typeSuffix)) : undefined;
}

/**
 * Classify one Gemini API error object.
 * @param {number} httpStatus - HTTP status, or the error's own `code` for in-stream errors.
 * @param {{ code?: number, message?: string, status?: string, details?: unknown[] } | undefined} error - Google error body.
 * @param {Headers} [headers] - response headers, for Retry-After.
 * @returns {LlmError} the harness failure.
 */
export function classifyGoogleError(httpStatus, error, headers) {
  const status = Number.isInteger(httpStatus) && httpStatus >= 100 && httpStatus <= 599 ? httpStatus : undefined;
  const grpc = typeof error?.status === 'string' ? error.status : '';
  const text = typeof error?.message === 'string' && error.message.length > 0 ? error.message : `HTTP ${httpStatus}`;
  const reason = detail(error?.details, 'ErrorInfo')?.reason;
  const retryDelayMs = parseDurationMs(detail(error?.details, 'RetryInfo')?.retryDelay) ?? retryAfterHeaderMs(headers);
  const quotaIds = (detail(error?.details, 'QuotaFailure')?.violations ?? [])
    .map((v) => `${v?.quotaId ?? ''} ${v?.quotaMetric ?? ''}`).join(' ');
  const message = `aistudio: Gemini API ${status ?? ''}${grpc ? ` ${grpc}` : ''}${reason ? ` (${reason})` : ''}: ${text}`.replace(/\s+:/, ':');
  const options = { ...(status === undefined ? {} : { status }) };

  if (reason === 'API_KEY_INVALID' || /api key not valid/i.test(text)) {
    return new LlmError(`${message} — check the GEMINI_API_KEY stored for dsh`, INVALID_CREDENTIAL_CODE, options);
  }
  if (GEMINI_CONTEXT_OVERFLOW.test(text) || isContextWindowExceededError(text)) {
    return new LlmError(message, CONTEXT_WINDOW_EXCEEDED_CODE, options);
  }
  if (status === 429 || grpc === 'RESOURCE_EXHAUSTED') {
    const daily = /per\s*day|PerDay/i.test(quotaIds);
    const withDelay = retryDelayMs === undefined ? options : { ...options, providerRetryAfterMs: retryDelayMs };
    return new LlmError(message, daily ? QUOTA_EXCEEDED_CODE : 'RATE_LIMIT', withDelay);
  }
  if (status === 401 || status === 403 || grpc === 'UNAUTHENTICATED' || grpc === 'PERMISSION_DENIED') {
    return new LlmError(message, 'AUTH', options);
  }
  if (status === 404 || grpc === 'NOT_FOUND') return new LlmError(message, 'UNKNOWN_MODEL', options);
  if (status === 408 || grpc === 'DEADLINE_EXCEEDED') return new LlmError(message, 'TIMEOUT', options);
  if (status === 499 || grpc === 'CANCELLED') return new LlmError(message, 'TRANSPORT', options);
  if ((status !== undefined && status >= 500) || grpc === 'UNAVAILABLE' || grpc === 'INTERNAL') {
    const withDelay = retryDelayMs === undefined ? options : { ...options, providerRetryAfterMs: retryDelayMs };
    return new LlmError(message, 'SERVER', withDelay);
  }
  return new LlmError(message, 'INVALID_REQUEST', options);
}

/**
 * Build the failure for a non-2xx HTTP response.
 * @param {Response} response - the failed response; its body is consumed.
 * @returns {Promise<LlmError>} the harness failure.
 */
export async function httpError(response) {
  let raw = '';
  try {
    raw = (await response.text()).slice(0, MAX_ERROR_BODY_CHARS);
  } catch {
    // An unreadable body still classifies by status.
  }
  let error;
  try {
    const parsed = JSON.parse(raw);
    error = Array.isArray(parsed) ? parsed[0]?.error : parsed?.error;
  } catch {
    error = undefined;
  }
  if (error === undefined && raw.trim().length > 0) error = { message: raw.trim().slice(0, 500) };
  return classifyGoogleError(response.status, error, response.headers);
}
