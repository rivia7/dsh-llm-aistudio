import assert from 'node:assert/strict';
import { test } from 'node:test';
import { AiStudioAdapter } from '../lib/adapter.js';
import { DEFAULT_MODELS, normalizeModels } from '../lib/catalog.js';
import { sseEvents } from '../lib/sse.js';

const models = normalizeModels(DEFAULT_MODELS, 'high');

function sseBody(events, { split = 7, hang = false } = {}) {
  const text = events.map((e) => `data: ${JSON.stringify(e)}\r\n\r\n`).join('');
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += split) controller.enqueue(bytes.slice(i, i + split));
      if (!hang) controller.close();
    },
  });
}

function adapter(fetchImpl, extra = {}) {
  return new AiStudioAdapter({
    displayName: 'AI Studio',
    models: () => models,
    baseURL: () => 'https://example.test/v1beta',
    includeThoughts: () => true,
    idleTimeoutMs: () => 200,
    apiKey: async () => 'test-key',
    canReadImages: () => false,
    readImage: async () => { throw new Error('unused'); },
    fetch: fetchImpl,
    ...extra,
  });
}

async function collect(iterable) {
  const chunks = [];
  try {
    for await (const chunk of iterable) chunks.push(chunk);
    return { chunks };
  } catch (error) {
    return { chunks, error };
  }
}

const request = { provider: 'aistudio', model: 'gemini-4-argon', reasoningEffort: 'low', messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] };

test('streams a reply over SSE split at arbitrary byte boundaries', async () => {
  let seen;
  const a = adapter(async (url, init) => {
    seen = { url, headers: init.headers, body: JSON.parse(init.body) };
    return new Response(sseBody([
      { candidates: [{ content: { parts: [{ text: 'Hel' }] } }] },
      { candidates: [{ content: { parts: [{ text: 'lo', thoughtSignature: 'QUJD' }] }, finishReason: 'STOP' }], usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 2, totalTokenCount: 5 } },
    ]), { status: 200 });
  });
  const { chunks, error } = await collect(a.stream(request));
  assert.equal(error, undefined);
  assert.equal(seen.url, 'https://example.test/v1beta/models/gemini-4-argon:streamGenerateContent?alt=sse');
  assert.equal(seen.headers['x-goog-api-key'], 'test-key');
  assert.match(seen.headers['user-agent'], /\S/);
  assert.deepEqual(seen.body.generationConfig.thinkingConfig, { thinkingLevel: 'LOW', includeThoughts: true });
  assert.equal(chunks.filter((c) => c.type === 'text-delta').map((c) => c.text).join(''), 'Hello');
  assert.deepEqual(chunks.at(-1).reason, { kind: 'stop' });
});

test('HTTP errors become classified LlmErrors', async () => {
  const a = adapter(async () => new Response(JSON.stringify({ error: { code: 429, status: 'RESOURCE_EXHAUSTED', message: 'busy' } }), { status: 429 }));
  const { error } = await collect(a.stream(request));
  assert.equal(error.failure.code, 'RATE_LIMIT');
});

test('a silent stream times out as TIMEOUT', async () => {
  const a = adapter(async (_url, init) => {
    const body = sseBody([{ candidates: [{ content: { parts: [{ text: 'x' }] } }] }], { hang: true });
    init.signal.addEventListener('abort', () => {});
    return new Response(body, { status: 200 });
  });
  const { error } = await collect(a.stream(request));
  assert.equal(error.failure.code, 'TIMEOUT');
});

test('network failures are TRANSPORT; caller aborts are rethrown untouched', async () => {
  const offline = adapter(async () => { throw new TypeError('fetch failed', { cause: Object.assign(new Error('x'), { code: 'ECONNRESET' }) }); });
  assert.equal((await collect(offline.stream(request))).error.failure.code, 'TRANSPORT');

  const controller = new AbortController();
  const aborting = adapter(async (_url, init) => {
    controller.abort(new Error('user stop'));
    if (init.signal.aborted) throw init.signal.reason;
    return new Response('', { status: 200 });
  });
  const { error } = await collect(aborting.stream({ ...request, signal: controller.signal }));
  assert.equal(error.message, 'user stop');
  assert.equal(error.failure, undefined);
});

test('metadata: efforts least to greatest, default effort, modalities follow the attachment service', async () => {
  const a = adapter(async () => new Response(''));
  const info = await a.resolveModel('aistudio', 'gemini-4-carbon');
  assert.deepEqual(info.reasoning.efforts.map((e) => e.id), ['low', 'medium', 'high']);
  assert.equal(info.reasoning.defaultEffort, 'high');
  assert.deepEqual(info.inputModalities, ['text']);
  assert.equal(info.context.contextWindow, 1_048_576);
  const listed = await a.listModels('aistudio');
  assert.deepEqual(listed.map((m) => m.id), ['gemini-4-carbon', 'gemini-4-argon']);
  const unlisted = await a.resolveModel('aistudio', 'gemini-9-future');
  assert.equal(unlisted.reasoning, undefined);
  assert.equal(a.providerRetryPolicy('aistudio').retryableCodes.includes('MALFORMED_RESPONSE'), true);
});

test('sseEvents handles CRLF, multi-line data and a trailing event without a blank line', async () => {
  const body = new Response('data: {"a":1}\r\n\r\n: comment\n\ndata: line1\ndata: line2\n\ndata: tail').body;
  const events = [];
  for await (const event of sseEvents(body)) events.push(event);
  assert.deepEqual(events, ['{"a":1}', 'line1\nline2', 'tail']);
});
