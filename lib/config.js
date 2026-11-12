/**
 * Plugin configuration schema. `apiKeyEnv` is volatile so a credential
 * reference edited in settings applies to the next request; every other field
 * restarts the plugin, which re-registers the route with the new catalog.
 * @module dsh-llm-aistudio/config
 */
import z from '@deepseek-ai/schemastery';
import { DEFAULT_MODELS } from './catalog.js';

/** Public Gemini API endpoint used by AI Studio keys. */
export const DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta';

function maybeVolatile(schema) {
  return typeof schema.volatile === 'function' ? schema.volatile() : schema;
}

const ModelSchema = z.object({
  id: z.string().required().description('Model id sent to the Gemini API, e.g. gemini-4-argon.'),
  name: z.string().description('Name shown in the model picker; defaults to the id.'),
  contextWindow: z.natural().description('Context window in tokens (default 1,048,576).'),
  maxTokens: z.natural().description('Maximum output tokens (default 65,536).'),
  efforts: z.dict(z.union([z.string(), z.number()]))
    .description('Reasoning efforts for this model: effort id -> Gemini thinkingLevel (MINIMAL/LOW/MEDIUM/HIGH) or a thinkingBudget token count. Empty or absent: no reasoning control.'),
  defaultEffort: z.string().description('Effort used when a request names none.'),
  input: z.array(z.union(['text', 'image'])).description('Accepted request modalities (default text and image).'),
});

/** Configuration schema. */
export const Config = z.object({
  provider: z.string().default('aistudio')
    .description('Provider route id, as used by the model picker and agent-default-model.'),
  displayName: z.string().default('AI Studio').description('Provider name shown in the model picker.'),
  apiKeyEnv: maybeVolatile(z.string().role('credential-ref').default('GEMINI_API_KEY'))
    .description('Credential reference for the Gemini API key: the dsh credential store first, then the launching environment.'),
  baseURL: z.string().default(DEFAULT_BASE_URL).description('Gemini API base URL.'),
  models: z.array(ModelSchema).default(DEFAULT_MODELS.map((model) => ({ ...model, efforts: { ...model.efforts }, input: [...model.input] })))
    .description('Models offered on this route.'),
  defaultEffort: z.string().default('high').description('Reasoning effort used when neither the request nor the model names one.'),
  includeThoughts: z.boolean().default(true).description('Request thought summaries and show them as reasoning.'),
  streamIdleTimeoutMs: z.natural().default(300_000).description('Fail a request whose stream delivers no data for this long (ms).'),
});

/**
 * Read the current value of a possibly-volatile field.
 * @param {unknown} value - plain value or volatile reference.
 * @returns {any} the current value.
 */
export function current(value) {
  if (value !== null && typeof value === 'object' && typeof value.get === 'function') return value.get();
  return value;
}
