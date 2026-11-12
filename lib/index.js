/**
 * DeepSeek Harness model provider for Google AI Studio.
 *
 * Registers one route (default `aistudio`) whose models stream straight from
 * the Gemini API with the user's own API key. No pi-ai, no @google/genai: the
 * request/stream mapping and every finish reason live in this package, so a
 * new Gemini model or finish reason needs no dependency upgrade.
 * @module dsh-llm-aistudio
 */
import { LlmError, assertUsableApiKey } from '@deepseek-ai/dsh-llm';
import { credentialRef } from '@deepseek-ai/dsh-credentials';
import { launchEnvironmentOf } from '@deepseek-ai/dsh-launch-environment';
import { AiStudioAdapter } from './adapter.js';
import { normalizeModels } from './catalog.js';
import { current } from './config.js';

export { Config } from './config.js';
export const name = 'llm-aistudio';
export const inject = ['llm'];

/**
 * Plugin entry.
 * @param {import('@deepseek-ai/cordis').Context} ctx - plugin context.
 * @param {object} config - validated configuration.
 */
export function apply(ctx, config) {
  const route = config.provider;
  const models = normalizeModels(config.models, config.defaultEffort);
  const baseURL = String(config.baseURL).replace(/\/+$/, '');

  const apiKey = async () => {
    const ref = credentialRef(current(config.apiKeyEnv) || 'GEMINI_API_KEY');
    const credentials = ctx.get('credentials');
    if (credentials !== undefined) {
      const hit = await credentials.resolve(ref);
      if (hit !== undefined) return assertUsableApiKey(hit.value, 'llm-aistudio', ref);
    } else {
      const ambient = launchEnvironmentOf(ctx).get(ref);
      if (ambient !== undefined && ambient.value.length > 0) return assertUsableApiKey(ambient.value, 'llm-aistudio', ref);
    }
    throw new LlmError(`llm-aistudio: no Gemini API key for provider route "${route}"; store ${ref} through the credentials service (the web Models page writes it) or export it in the launching environment`, 'MISSING_CREDENTIAL');
  };

  const adapter = new AiStudioAdapter({
    displayName: config.displayName,
    models: () => models,
    baseURL: () => baseURL,
    includeThoughts: () => config.includeThoughts,
    idleTimeoutMs: () => config.streamIdleTimeoutMs,
    apiKey,
    canReadImages: () => ctx.get('attachments') !== undefined,
    readImage: (ref, target, signal) => {
      const store = ctx.get('attachments');
      if (store === undefined) throw new LlmError('llm-aistudio: the attachment service is not mounted', 'INVALID_REQUEST');
      return store.readImageRequest(ref, target, signal);
    },
  });

  const settingsNs = ctx.fiber?.entry?.options?.id ?? name;
  ctx.llm.registerConfigurableProviders?.([{
    provider: route,
    displayName: config.displayName,
    settingsNs,
    settingsPath: [],
    defaultBaseURL: baseURL,
  }]);
  ctx.llm.registerAdapter([route], adapter);
}
