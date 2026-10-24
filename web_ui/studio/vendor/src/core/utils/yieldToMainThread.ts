function nowMs(): number {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }

  return Date.now();
}

export async function yieldToMainThread(): Promise<void> {
  if (typeof globalThis.requestAnimationFrame === 'function') {
    await new Promise<void>((resolve) => {
      globalThis.requestAnimationFrame(() => resolve());
    });
    return;
  }

  await Promise.resolve();
}

export function createMainThreadYieldController(budgetMs: number = 8): () => Promise<void> {
  let lastYieldAt = nowMs();

  return async () => {
    const elapsed = nowMs() - lastYieldAt;
    if (elapsed < budgetMs) {
      return;
    }

    await yieldToMainThread();
    lastYieldAt = nowMs();
  };
}
