import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mapUsage, translateStream } from '../lib/stream.js';

async function run(events) {
  const chunks = [];
  try {
    for await (const chunk of translateStream((async function* () { for (const e of events) yield JSON.stringify(e); })(), { model: 'gemini-4-argon' })) {
      chunks.push(chunk);
    }
    return { chunks };
  } catch (error) {
    return { chunks, error };
  }
}
const candidate = (parts, finishReason, extra = {}) => ({ candidates: [{ content: { role: 'model', parts }, ...(finishReason ? { finishReason } : {}), ...extra }] });

test('reasoning, text and a call become ordered blocks with aligned replay metadata', async () => {
  const { chunks, error } = await run([
    candidate([{ text: 'think ', thought: true }]),
    candidate([{ text: 'more', thought: true, thoughtSignature: 'T1' }]),
    candidate([{ text: 'Let me check.' }]),
    candidate([{ functionCall: { id: 'call_1', name: 'read', args: { path: 'a' } }, thoughtSignature: 'S1' }, { text: '' }], 'STOP'),
    { usageMetadata: { promptTokenCount: 100, cachedContentTokenCount: 40, candidatesTokenCount: 10, thoughtsTokenCount: 5, totalTokenCount: 115 } },
  ]);
  assert.equal(error, undefined);
  assert.deepEqual(chunks.map((c) => c.type), [
    'block-start', 'reasoning-delta', 'reasoning-delta', 'block-end',
    'block-start', 'text-delta', 'block-end',
    'block-start', 'tool-call-delta', 'block-end',
    'usage', 'finish',
  ]);
  const ends = chunks.filter((c) => c.type === 'block-end');
  assert.deepEqual(ends.map((c) => c.index), [0, 1, 2]);
  assert.deepEqual(ends[0].block, { type: 'reasoning', text: 'think more' });
  assert.deepEqual(ends[2].block, { type: 'tool-call', id: 'call_1', name: 'read', arguments: '{"path":"a"}' });
  const finish = chunks.at(-1);
  assert.deepEqual(finish.reason, { kind: 'tool-calls' });
  assert.deepEqual(finish.replayState.blocks, [{ thoughtSignature: 'T1' }, {}, { thoughtSignature: 'S1' }]);
  assert.equal(finish.replayState.response.kind, 'aistudio');
  assert.deepEqual(chunks.at(-2).usage, { inputTokens: 60, outputTokens: 15, totalTokens: 115, cacheReadTokens: 40, reasoningTokens: 5 });
});

test('a signature-only part rides on a block so it is echoed back', async () => {
  const { chunks } = await run([candidate([{ text: 'Answer.' }]), candidate([{ thoughtSignature: 'SIG' }], 'STOP')]);
  assert.deepEqual(chunks.at(-1).replayState.blocks, [{ thoughtSignature: 'SIG' }]);
  assert.deepEqual(chunks.at(-1).reason, { kind: 'stop' });
});

test('a missing call id is generated in the Gemini alphabet', async () => {
  const { chunks } = await run([candidate([{ functionCall: { name: 'ls', args: {} } }], 'STOP')]);
  const call = chunks.find((c) => c.type === 'tool-call-delta');
  assert.match(call.id, /^call_[a-f0-9]{24}$/);
});

const failures = [
  ['MALFORMED_FUNCTION_CALL', 'MALFORMED_RESPONSE'],
  ['MALFORMED_RESPONSE', 'MALFORMED_RESPONSE'],
  ['UNEXPECTED_TOOL_CALL', 'MALFORMED_RESPONSE'],
  ['SOME_FUTURE_REASON', 'MALFORMED_RESPONSE'],
  ['OTHER', 'MALFORMED_RESPONSE'],
  ['SAFETY', 'CONTENT_FILTERED'],
  ['RECITATION', 'CONTENT_FILTERED'],
];
for (const [reason, code] of failures) {
  test(`finishReason ${reason} -> ${code}`, async () => {
    const { chunks, error } = await run([candidate([{ text: 'partial' }], reason, { finishMessage: 'detail' })]);
    assert.equal(error?.failure?.code, code);
    assert.match(error.message, new RegExp(`${reason}: detail`));
    assert.ok(!chunks.some((c) => c.type === 'finish'), 'failures are thrown, never a success finish');
  });
}

test('STOP without visible text is EMPTY_RESPONSE (retryable), even with thoughts', async () => {
  const { error } = await run([candidate([{ text: 'hmm', thought: true }], 'STOP')]);
  assert.equal(error?.failure?.code, 'EMPTY_RESPONSE');
});

test('MAX_TOKENS is a max-tokens finish', async () => {
  const { chunks } = await run([candidate([{ text: 'cut' }], 'MAX_TOKENS')]);
  assert.deepEqual(chunks.at(-1).reason, { kind: 'max-tokens' });
});

test('no finish reason is a TRANSPORT failure; a blocked prompt is CONTENT_FILTERED', async () => {
  assert.equal((await run([candidate([{ text: 'a' }])])).error?.failure?.code, 'TRANSPORT');
  assert.equal((await run([{ promptFeedback: { blockReason: 'SAFETY' } }])).error?.failure?.code, 'CONTENT_FILTERED');
});

test('an in-stream error object is classified', async () => {
  const { error } = await run([{ error: { code: 503, status: 'UNAVAILABLE', message: 'overloaded' } }]);
  assert.equal(error?.failure?.code, 'SERVER');
});

test('mapUsage keeps counts disjoint and omits zero optional fields', () => {
  assert.deepEqual(mapUsage({ promptTokenCount: 7, candidatesTokenCount: 3 }), { inputTokens: 7, outputTokens: 3 });
});
