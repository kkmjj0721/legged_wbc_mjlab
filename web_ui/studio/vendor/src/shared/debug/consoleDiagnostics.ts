import { isRegressionDebugEnabled } from './regressionDebugEnabled';

export function shouldEmitRegressionConsoleDiagnostics(
  targetWindow: Pick<Window, 'location'> | null =
    typeof window !== 'undefined' ? window : null,
): boolean {
  return Boolean(targetWindow && isRegressionDebugEnabled(targetWindow));
}

export function logRegressionInfo(...args: unknown[]): void {
  if (shouldEmitRegressionConsoleDiagnostics()) {
    console.info(...args);
  }
}

export function logRegressionWarn(...args: unknown[]): void {
  if (shouldEmitRegressionConsoleDiagnostics()) {
    console.warn(...args);
  }
}

export function logRegressionError(...args: unknown[]): void {
  if (shouldEmitRegressionConsoleDiagnostics()) {
    console.error(...args);
  }
}
