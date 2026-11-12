/**
 * Minimal Server-Sent Events reader for `streamGenerateContent?alt=sse`.
 * @module dsh-llm-aistudio/sse
 */

/**
 * Yield the `data` payload of each SSE event in a response body.
 * @param {ReadableStream<Uint8Array>} body - HTTP response body.
 * @param {() => void} [onActivity] - called whenever bytes arrive (idle-timer reset).
 * @param {AbortSignal} [signal] - cancels a pending read and throws its reason,
 *   whether or not the body itself honours the request signal.
 * @returns {AsyncGenerator<string>} one joined `data` field per event.
 */
export async function* sseEvents(body, onActivity, signal) {
  const reader = body.getReader();
  const onAbort = () => {
    reader.cancel(signal.reason).catch(() => {});
  };
  if (signal?.aborted) onAbort();
  else signal?.addEventListener('abort', onAbort, { once: true });
  const decoder = new TextDecoder();
  let buffer = '';
  let data = [];
  let finished = false;
  const take = (line) => {
    if (line.endsWith('\r')) line = line.slice(0, -1);
    if (line === '') {
      if (data.length === 0) return undefined;
      const event = data.join('\n');
      data = [];
      return event;
    }
    if (line.startsWith(':')) return undefined;
    if (line.startsWith('data:')) data.push(line.slice(line.charAt(5) === ' ' ? 6 : 5));
    return undefined;
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (signal?.aborted) throw signal.reason ?? new Error('aborted');
      if (done) break;
      onActivity?.();
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf('\n')) !== -1) {
        const event = take(buffer.slice(0, newline));
        buffer = buffer.slice(newline + 1);
        if (event !== undefined) yield event;
      }
    }
    finished = true;
    buffer += decoder.decode();
    for (const line of [...buffer.split('\n'), '']) {
      const event = take(line);
      if (event !== undefined) yield event;
    }
  } finally {
    signal?.removeEventListener('abort', onAbort);
    if (!finished) await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
