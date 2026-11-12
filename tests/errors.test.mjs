import assert from 'node:assert/strict';
import { test } from 'node:test';
import { classifyGoogleError, httpError, parseDurationMs } from '../lib/errors.js';

const code = (status, error, headers) => classifyGoogleError(status, error, headers).failure;

test('invalid API key -> INVALID_CREDENTIAL', () => {
  const f = code(400, { status: 'INVALID_ARGUMENT', message: 'API key not valid. Please pass a valid API key.', details: [{ '@type': 'type.googleapis.com/google.rpc.ErrorInfo', reason: 'API_KEY_INVALID' }] });
  assert.equal(f.code, 'INVALID_CREDENTIAL');
  assert.equal(f.status, 400);
});

test('context overflow -> CONTEXT_WINDOW_EXCEEDED', () => {
  const f = code(400, { status: 'INVALID_ARGUMENT', message: 'The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).' });
  assert.equal(f.code, 'CONTEXT_WINDOW_EXCEEDED');
});

test('429 rate limit carries RetryInfo; a per-day quota is QUOTA', () => {
  const rate = code(429, { status: 'RESOURCE_EXHAUSTED', message: 'slow down', details: [{ '@type': 'type.googleapis.com/google.rpc.RetryInfo', retryDelay: '31s' }] });
  assert.equal(rate.code, 'RATE_LIMIT');
  assert.equal(rate.providerRetryAfterMs, 31_000);
  const daily = code(429, { status: 'RESOURCE_EXHAUSTED', message: 'quota', details: [{ '@type': 'type.googleapis.com/google.rpc.QuotaFailure', violations: [{ quotaId: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' }] }] });
  assert.equal(daily.code, 'QUOTA');
});

test('status families', () => {
  assert.equal(code(403, { status: 'PERMISSION_DENIED', message: 'no' }).code, 'AUTH');
  assert.equal(code(404, { status: 'NOT_FOUND', message: 'models/x is not found' }).code, 'UNKNOWN_MODEL');
  assert.equal(code(503, { status: 'UNAVAILABLE', message: 'The model is overloaded.' }).code, 'SERVER');
  assert.equal(code(400, { status: 'FAILED_PRECONDITION', message: 'User location is not supported for the API use.' }).code, 'INVALID_REQUEST');
});

test('httpError reads a JSON or plain body', async () => {
  const json = await httpError(new Response(JSON.stringify({ error: { code: 500, status: 'INTERNAL', message: 'oops' } }), { status: 500 }));
  assert.equal(json.failure.code, 'SERVER');
  const plain = await httpError(new Response('<html>bad gateway</html>', { status: 502, headers: { 'retry-after': '3' } }));
  assert.equal(plain.failure.code, 'SERVER');
  assert.equal(plain.failure.providerRetryAfterMs, 3000);
});

test('parseDurationMs', () => {
  assert.equal(parseDurationMs('0.5s'), 500);
  assert.equal(parseDurationMs('nope'), undefined);
});
