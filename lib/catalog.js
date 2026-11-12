/**
 * Model catalog for the AI Studio route: shipped defaults, normalization of
 * configured entries, and reasoning-effort helpers.
 *
 * The catalog is configuration, not a bundled SDK table, so a model Google
 * ships tomorrow is one `models` entry away instead of a dependency upgrade.
 * @module dsh-llm-aistudio/catalog
 */

/** Gemini `thinkingLevel` values, least to greatest. */
export const THINKING_LEVELS = ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH'];

/** Selector names for the effort ids this plugin offers by default. */
export const EFFORT_NAMES = { minimal: 'Minimal', low: 'Low', medium: 'Medium', high: 'High' };

/** Capacity fallbacks for a configured model that states none. */
export const DEFAULT_CONTEXT_WINDOW = 1_048_576;
export const DEFAULT_MAX_TOKENS = 65_536;

const GEMINI_4_EFFORTS = { low: 'LOW', medium: 'MEDIUM', high: 'HIGH' };

/**
 * Shipped defaults: the Gemini 4 models AI Studio serves. Capacities come from
 * `GET /v1beta/models/{id}` (inputTokenLimit / outputTokenLimit), 2026-11-12.
 */
export const DEFAULT_MODELS = Object.freeze([
  Object.freeze({
    id: 'gemini-4-carbon',
    name: 'Gemini 4 Carbon',
    contextWindow: 1_048_576,
    maxTokens: 32_768,
    efforts: GEMINI_4_EFFORTS,
    input: ['text', 'image'],
  }),
  Object.freeze({
    id: 'gemini-4-argon',
    name: 'Gemini 4 Argon',
    contextWindow: 1_048_576,
    maxTokens: 32_768,
    efforts: GEMINI_4_EFFORTS,
    input: ['text', 'image'],
  }),
]);

/**
 * Order key of one effort's wire value: thinking levels by strength, token
 * budgets by size, so selectors list efforts least to greatest.
 * @param {string | number} wire - thinkingLevel name or thinkingBudget.
 * @returns {number} sort key.
 */
function effortRank(wire) {
  if (typeof wire === 'number') return 1_000 + wire;
  const index = THINKING_LEVELS.indexOf(String(wire).toUpperCase());
  return index === -1 ? 500 : index;
}

/**
 * Validate one effort's wire value.
 * @param {string} modelId - model the effort belongs to, for diagnostics.
 * @param {string} effort - effort id.
 * @param {unknown} wire - configured value.
 * @returns {string | number} the thinkingLevel (upper case) or thinkingBudget.
 */
function effortWire(modelId, effort, wire) {
  if (typeof wire === 'number') {
    if (!Number.isInteger(wire) || wire < -1) {
      throw new Error(`llm-aistudio: model "${modelId}" effort "${effort}" budget must be an integer >= -1 (-1 = dynamic)`);
    }
    return wire;
  }
  const level = String(wire).toUpperCase();
  if (!THINKING_LEVELS.includes(level)) {
    throw new Error(`llm-aistudio: model "${modelId}" effort "${effort}" must be one of ${THINKING_LEVELS.join('/')} or a token budget, not "${wire}"`);
  }
  return level;
}

/**
 * Normalize configured models into the shape the adapter serves.
 * @param {readonly object[]} entries - `models` from the plugin configuration.
 * @param {string} fallbackEffort - plugin-level default effort.
 * @returns {readonly object[]} frozen models: id, name, capacities, ordered efforts, default effort, input.
 * @throws Error naming the model when an entry cannot be served.
 */
export function normalizeModels(entries, fallbackEffort) {
  const seen = new Set();
  return Object.freeze(entries.map((entry) => {
    const id = String(entry.id ?? '').replace(/^models\//, '').trim();
    if (id.length === 0) throw new Error('llm-aistudio: a configured model has an empty id');
    if (seen.has(id)) throw new Error(`llm-aistudio: model "${id}" is listed more than once`);
    seen.add(id);
    const efforts = Object.entries(entry.efforts ?? {})
      .map(([effort, wire]) => ({ id: effort, wire: effortWire(id, effort, wire) }))
      .sort((a, b) => effortRank(a.wire) - effortRank(b.wire));
    const ids = efforts.map((effort) => effort.id);
    const preferred = [entry.defaultEffort, fallbackEffort].find((effort) => effort && ids.includes(effort));
    const input = entry.input?.length ? [...new Set(entry.input)] : ['text', 'image'];
    return Object.freeze({
      id,
      name: entry.name || id,
      contextWindow: entry.contextWindow || DEFAULT_CONTEXT_WINDOW,
      maxTokens: entry.maxTokens || DEFAULT_MAX_TOKENS,
      efforts: Object.freeze(efforts.map((effort) => Object.freeze(effort))),
      defaultEffort: preferred ?? ids.at(-1),
      input: Object.freeze(input),
    });
  }));
}

/**
 * Metadata for a model id the configuration does not list. Core routing
 * accepts unlisted ids, so serve them conservatively: text in, no reasoning
 * control, the generic capacity fallbacks.
 * @param {string} id - requested model id.
 * @returns {object} a normalized model.
 */
export function unlistedModel(id) {
  return Object.freeze({
    id: id.replace(/^models\//, ''),
    name: id,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: DEFAULT_MAX_TOKENS,
    efforts: Object.freeze([]),
    defaultEffort: undefined,
    input: Object.freeze(['text']),
  });
}
