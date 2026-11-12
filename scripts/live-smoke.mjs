#!/usr/bin/env node
/**
 * Live smoke test against the real Gemini API, driving the adapter directly
 * (no dsh process). Reads GEMINI_API_KEY from the environment or the dsh
 * credential store; never prints it. Run `npm run link-dev` first.
 *
 *   node scripts/live-smoke.mjs [text|tools|bypass|image|all] [model ...]
 */
import { readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import YAML from 'yaml';
import { AiStudioAdapter } from '../lib/adapter.js';
import { DEFAULT_MODELS, normalizeModels } from '../lib/catalog.js';
import { DEFAULT_BASE_URL } from '../lib/config.js';

function loadKey() {
  if (process.env.GEMINI_API_KEY) return process.env.GEMINI_API_KEY;
  const home = process.env.DSH_HOME || join(homedir(), '.dsh');
  const store = YAML.parse(readFileSync(join(home, '.credentials.yaml'), 'utf8'));
  const key = store?.refs?.GEMINI_API_KEY;
  if (!key) throw new Error('GEMINI_API_KEY not in env or the dsh credential store');
  return key;
}

const key = loadKey();
const models = normalizeModels(DEFAULT_MODELS, 'high');
// 1x1 red PNG.
const PNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==', 'base64');
const adapter = new AiStudioAdapter({
  displayName: 'AI Studio',
  models: () => models,
  baseURL: () => DEFAULT_BASE_URL,
  includeThoughts: () => true,
  idleTimeoutMs: () => 120_000,
  apiKey: async () => key,
  canReadImages: () => true,
  readImage: async (ref) => ({ data: PNG, mediaType: 'image/png', bytes: PNG.length, width: ref.width, height: ref.height }),
});

const ROUTE = 'aistudio';
const user = (text) => ({ role: 'user', content: [{ type: 'text', text }] });
const system = (text) => ({ role: 'system', content: [{ type: 'text', text }], source: { kind: 'system-prompt' } });

/** Run one call; return the assembled assistant message, finish, usage, and timing. */
async function call(options) {
  const started = Date.now();
  const blocks = [];
  let finish;
  let usage;
  try {
    for await (const chunk of adapter.stream({ provider: ROUTE, ...options })) {
      if (chunk.type === 'block-end') blocks.push(chunk.block);
      else if (chunk.type === 'usage') usage = chunk.usage;
      else if (chunk.type === 'finish') finish = chunk;
    }
  } catch (error) {
    return { error: `${error.failure?.code ?? error.name}: ${error.message}`, ms: Date.now() - started };
  }
  const message = {
    role: 'assistant',
    content: blocks,
    source: { kind: 'model', provider: ROUTE, model: options.model, replayState: finish?.replayState },
  };
  return { message, finish: finish?.reason, usage, ms: Date.now() - started };
}

function summarize(label, result) {
  if (result.error) return console.log(`${label}: ERROR ${result.error} (${result.ms} ms)`);
  const text = result.message.content.filter((b) => b.type === 'text').map((b) => b.text).join('').trim();
  const kinds = result.message.content.map((b) => b.type).join(',');
  const sigs = (result.message.source.replayState?.blocks ?? []).map((b) => (b.thoughtSignature ? 'S' : '-')).join('');
  console.log(`${label}: finish=${result.finish?.kind} blocks=[${kinds}] sigs=[${sigs}] out=${result.usage?.outputTokens} reasoning=${result.usage?.reasoningTokens ?? 0} ${result.ms} ms text=${JSON.stringify(text.slice(0, 80))}`);
}

const WEATHER = [{
  name: 'get_weather',
  description: 'Get the current weather for a city.',
  parameters: { type: 'object', properties: { city: { type: 'string', description: 'City name' } }, required: ['city'] },
}];

async function textScenario(modelIds) {
  for (const model of modelIds) {
    for (const effort of ['low', 'medium', 'high']) {
      summarize(`text ${model}/${effort}`, await call({
        model,
        reasoningEffort: effort,
        messages: [system('You are terse.'), user('Reply with exactly: OK')],
      }));
    }
  }
}

async function toolsScenario(modelIds) {
  for (const model of modelIds) {
    const history = [system('Use tools when asked.'), user('What is the weather in Paris? Call get_weather, then answer in one short sentence.')];
    const first = await call({ model, reasoningEffort: 'low', messages: history, tools: WEATHER });
    summarize(`tools#1 ${model}`, first);
    if (first.error || first.finish?.kind !== 'tool-calls') continue;
    const calls = first.message.content.filter((b) => b.type === 'tool-call');
    console.log(`  tool calls: ${calls.map((c) => `${c.name}(${c.arguments}) id=${c.id}`).join(' | ')}`);
    const results = calls.map((c) => ({ role: 'tool', toolCallId: c.id, content: [{ type: 'text', text: 'Sunny, 21 C, light wind.' }], source: { kind: 'tool', callId: c.id } }));
    const second = await call({ model, reasoningEffort: 'low', messages: [...history, first.message, ...results], tools: WEATHER });
    summarize(`tools#2 ${model}`, second);
  }
}

/** A function call this route did not produce (no replay state) in the current turn. */
async function bypassScenario(modelIds) {
  for (const model of modelIds) {
    const foreign = {
      role: 'assistant',
      content: [{ type: 'tool-call', id: 'toolu_01ABCdef', name: 'get_weather', arguments: '{"city":"Paris"}' }],
      source: { kind: 'model', provider: 'antigravity', model: 'gemini-4-carbon' },
    };
    const result = { role: 'tool', toolCallId: 'toolu_01ABCdef', content: [{ type: 'text', text: 'Rain, 12 C.' }], source: { kind: 'tool', callId: 'toolu_01ABCdef' } };
    summarize(`bypass ${model}`, await call({
      model,
      reasoningEffort: 'low',
      messages: [user('What is the weather in Paris? Answer in one short sentence after using the tool.'), foreign, result],
      tools: WEATHER,
    }));
  }
}

async function imageScenario(modelIds) {
  const ref = { attachmentId: 'att_smoke_png', mediaType: 'image/png', bytes: PNG.length, width: 1, height: 1, name: 'pixel.png' };
  for (const model of modelIds) {
    summarize(`image ${model}`, await call({
      model,
      reasoningEffort: 'low',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'What single color is this image? Answer with one word.' }, { type: 'image', attachment: ref }] }],
    }));
  }
}

const [scenario = 'all', ...rest] = process.argv.slice(2);
const modelIds = rest.length > 0 ? rest : models.map((m) => m.id);
if (scenario === 'text' || scenario === 'all') await textScenario(modelIds);
if (scenario === 'tools' || scenario === 'all') await toolsScenario(modelIds);
if (scenario === 'bypass' || scenario === 'all') await bypassScenario(modelIds);
if (scenario === 'image' || scenario === 'all') await imageScenario(modelIds);
