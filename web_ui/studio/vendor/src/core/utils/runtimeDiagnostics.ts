export function isDevelopmentBuild(): boolean {
  const runtimeEnv = (import.meta as ImportMeta & { env?: { DEV?: boolean } }).env;
  if (typeof runtimeEnv?.DEV === 'boolean') {
    return runtimeEnv.DEV;
  }

  if (typeof process !== 'undefined' && process?.env) {
    return process.env.NODE_ENV !== 'production';
  }

  return false;
}

export function isRegressionDebugEnabled(): boolean {
  try {
    const search = (globalThis as { location?: { search?: string } }).location?.search ?? '';
    return new URLSearchParams(search).get('regressionDebug') === '1';
  } catch {
    return false;
  }
}

export function normalizeRuntimeError(error: unknown, fallbackMessage: string): Error {
  if (error instanceof Error) {
    return error;
  }

  if (typeof error === 'string' && error.trim()) {
    return new Error(error);
  }

  return new Error(fallbackMessage);
}

export function logRuntimeFailure(
  scope: string,
  error: unknown,
  level: 'error' | 'warn' = 'error',
): Error {
  const normalized = normalizeRuntimeError(error, `${scope} failed.`);
  console[level](`[${scope}]`, normalized);
  return normalized;
}

export function failFastInDev(
  scope: string,
  error: unknown,
  level: 'error' | 'warn' = 'error',
): Error {
  const normalized = logRuntimeFailure(scope, error, level);
  if (isDevelopmentBuild()) {
    throw normalized;
  }
  return normalized;
}

export function scheduleFailFastInDev(
  scope: string,
  error: unknown,
  level: 'error' | 'warn' = 'error',
): Error {
  const normalized = logRuntimeFailure(scope, error, level);
  if (isDevelopmentBuild()) {
    queueMicrotask(() => {
      throw normalized;
    });
  }
  return normalized;
}
