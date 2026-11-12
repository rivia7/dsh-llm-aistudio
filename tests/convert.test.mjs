import assert from 'node:assert/strict';
import { test } from 'node:test';
import { DEFAULT_MODELS, normalizeModels } from '../lib/catalog.js';
import { BYPASS_SIGNATURE, buildRequest, rewriteConstAsEnum, thinkingConfig, toolDeclarations, wireCallId } from '../lib/convert.js';

const [CARBON] = normalizeModels(DEFAULT_MODELS, 'high');
const [FLASH25] = normalizeModels([{ id: 'gemini-2.5-flash', efforts: { low: 1024, high: 24576 } }], 'high');
const SIG = 'EpECCo4CAWkUfRO/bhvQk3rH';
const route = 'aistudio';

const user = (text) => ({ role: 'user', content: [{ type: 'text', text }] });
const system = (text) => ({ role: 'system', content: [{ type: 'text', text }], source: { kind: 'system-prompt' } });
const toolResult = (id, text, isError) => ({ role: 'tool', toolCallId: id, isError, content: [{ type: 'text', text }], source: { kind: 'tool', callId: id } });
const ours = (content, blocks, model = CARBON.id) => ({
  role: 'assistant',
  content,
  source: { kind: 'model', provider: route, model, replayState: { response: { kind: 'aistudio', version: 1, model }, blocks } },
});
const foreign = (content) => ({ role: 'assistant', content, source: { kind: 'model', provider: 'antigravity', model: 'gemini-4-carbon' } });
const build = (messages, extra = {}, model = CARBON) =>
  buildRequest({ options: { messages, ...extra }, model, route, includeThoughts: true, imagesAllowed: false });

test('leading system messages become systemInstruction; a later one becomes a reminder turn', async () => {
  const body = await build([system('rules'), user('hi'), system('updated rules')]);
  assert.deepEqual(body.systemInstruction, { parts: [{ text: 'rules' }] });
  assert.deepEqual(body.contents, [
    { role: 'user', parts: [{ text: 'hi' }] },
    { role: 'user', parts: [{ text: '<system-reminder>\nupdated rules\n</system-reminder>' }] },
  ]);
});

test('a reminder arriving between a call and its result waits for the result', async () => {
  const call = { type: 'tool-call', id: 'call_1', name: 'read', arguments: '{"path":"a"}' };
  const body = await build([user('go'), ours([call], [{ thoughtSignature: SIG }]), system('late rules'), toolResult('call_1', 'data')]);
  assert.equal(body.contents[1].role, 'model');
  assert.ok(body.contents[2].parts[0].functionResponse, 'function response follows the call directly');
  assert.match(body.contents[3].parts[0].text, /late rules/);
});

test('same-route same-model replay echoes each signature on its own part, including empty text', async () => {
  const content = [
    { type: 'reasoning', text: 'thinking' },
    { type: 'text', text: '' },
    { type: 'tool-call', id: 'call_9', name: 'read', arguments: '{"path":"x"}' },
  ];
  const body = await build([user('go'), ours(content, [{ thoughtSignature: 'AAAA' }, { thoughtSignature: SIG }, {}])]);
  assert.deepEqual(body.contents[1], {
    role: 'model',
    parts: [
      { thought: true, text: 'thinking', thoughtSignature: 'AAAA' },
      { text: '', thoughtSignature: SIG },
      // The current-turn call has no genuine signature, so it gets the bypass value.
      { functionCall: { name: 'read', args: { path: 'x' }, id: 'call_9' }, thoughtSignature: BYPASS_SIGNATURE },
    ],
  });
});

test('another model drops reasoning and signatures; only current-turn calls get the bypass', async () => {
  const old = foreign([{ type: 'reasoning', text: 'x' }, { type: 'tool-call', id: 'a', name: 't', arguments: '{}' }]);
  const now = foreign([{ type: 'text', text: 'checking' }, { type: 'tool-call', id: 'b', name: 't', arguments: '{}' }, { type: 'tool-call', id: 'c', name: 't', arguments: '{}' }]);
  const body = await build([user('one'), old, toolResult('a', 'r1'), user('two'), now, toolResult('b', 'r2'), toolResult('c', 'r3')]);
  const [, oldModel, , , nowModel, responses] = body.contents;
  assert.deepEqual(oldModel.parts, [{ functionCall: { name: 't', args: {}, id: 'a' } }], 'earlier turn: no bypass, no reasoning');
  assert.equal(nowModel.parts[0].thoughtSignature, undefined);
  assert.equal(nowModel.parts[1].thoughtSignature, BYPASS_SIGNATURE, 'first call of the current turn');
  assert.equal(nowModel.parts[2].thoughtSignature, undefined, 'parallel calls carry no signature');
  assert.equal(responses.parts.length, 2, 'parallel results share one user turn');
});

test('a replay envelope from another model or misaligned is ignored', async () => {
  const content = [{ type: 'text', text: 'hello' }];
  const otherModel = await build([user('a'), ours(content, [{ thoughtSignature: SIG }], 'gemini-4-argon')]);
  assert.deepEqual(otherModel.contents[1].parts, [{ text: 'hello' }]);
  const misaligned = await build([user('a'), ours(content, [{ thoughtSignature: SIG }, {}])]);
  assert.deepEqual(misaligned.contents[1].parts, [{ text: 'hello' }]);
});

test('tool results: names looked up, ids normalized, errors under `error`', async () => {
  const call = { type: 'tool-call', id: 'toolu:01/x', name: 'bash', arguments: '{"cmd":"ls"}' };
  const body = await build([user('go'), ours([call], [{ thoughtSignature: SIG }]), toolResult('toolu:01/x', 'boom', true)]);
  assert.equal(body.contents[1].parts[0].functionCall.id, 'toolu_01_x');
  assert.deepEqual(body.contents[2].parts[0], { functionResponse: { name: 'bash', response: { error: 'boom' }, id: 'toolu_01_x' } });
});

test('Gemini 2.5: no call ids, no bypass, budget-based thinking', async () => {
  const call = { type: 'tool-call', id: 'c1', name: 't', arguments: '{}' };
  const body = await build([user('go'), foreign([call]), toolResult('c1', 'ok')], { reasoningEffort: 'low' }, FLASH25);
  assert.deepEqual(body.contents[1].parts, [{ functionCall: { name: 't', args: {} } }]);
  assert.deepEqual(body.contents[2].parts[0].functionResponse, { name: 't', response: { output: 'ok' } });
  assert.deepEqual(body.generationConfig.thinkingConfig, { thinkingBudget: 1024, includeThoughts: true });
});

test('thinkingConfig picks the requested effort, then the default, then the strongest', () => {
  assert.deepEqual(thinkingConfig(CARBON, 'medium', true), { thinkingLevel: 'MEDIUM', includeThoughts: true });
  assert.deepEqual(thinkingConfig(CARBON, 'bogus', false), { thinkingLevel: 'HIGH', includeThoughts: false });
  assert.equal(thinkingConfig({ ...CARBON, efforts: [] }, 'high', true), undefined);
});

test('generationConfig clamps maxTokens and limits stop sequences', async () => {
  const body = await build([user('x')], { maxTokens: 999_999, stop: ['1', '2', '3', '4', '5', '6'], temperature: 0.5 });
  assert.equal(body.generationConfig.maxOutputTokens, CARBON.maxTokens);
  assert.equal(body.generationConfig.stopSequences.length, 5);
  assert.equal(body.generationConfig.temperature, 0.5);
});

test('tool declarations use parametersJsonSchema with const rewritten to enum', () => {
  const [{ functionDeclarations }] = toolDeclarations([
    { name: 'a', description: 'd', parameters: { type: 'object', properties: { mode: { const: 'fast' }, const: { type: 'string' } } } },
    { name: 'b', description: 'no params', parameters: {} },
  ]);
  assert.deepEqual(functionDeclarations[0].parametersJsonSchema.properties, { mode: { enum: ['fast'] }, const: { type: 'string' } });
  assert.deepEqual(functionDeclarations[1].parametersJsonSchema, { type: 'object', properties: {} });
  assert.deepEqual(rewriteConstAsEnum({ anyOf: [{ const: 1 }] }), { anyOf: [{ enum: [1] }] });
});

test('images are inlined after a handle line; an over-budget request asks for offload', async () => {
  const ref = (id, bytes) => ({ attachmentId: id, mediaType: 'image/png', bytes, width: 10, height: 10 });
  const readImage = async (r) => ({ data: new Uint8Array(r.bytes), mediaType: 'image/png', bytes: r.bytes, width: 10, height: 10 });
  const small = { role: 'user', content: [{ type: 'text', text: 'look' }, { type: 'image', attachment: ref('att_a', 3) }] };
  const body = await buildRequest({ options: { messages: [small] }, model: CARBON, route, includeThoughts: true, imagesAllowed: true, readImage });
  assert.equal(body.contents[0].parts.length, 3);
  assert.deepEqual(body.contents[0].parts[2], { inlineData: { mimeType: 'image/png', data: 'AAAA' } });

  const huge = { role: 'user', content: [1, 2].map((n) => ({ type: 'image', attachment: ref(`att_${n}`, 8 * 1024 * 1024) })) };
  await assert.rejects(
    buildRequest({ options: { messages: [huge] }, model: CARBON, route, includeThoughts: true, imagesAllowed: true, readImage }),
    (error) => error.failure?.code === 'IMAGE_OFFLOAD_REQUIRED' && error.failure.offloadImages > 0,
  );
});

test('wireCallId keeps Gemini-safe characters only', () => {
  assert.equal(wireCallId('toolu_01:ab/c'), 'toolu_01_ab_c');
  assert.equal(wireCallId('x'.repeat(80)).length, 64);
});
