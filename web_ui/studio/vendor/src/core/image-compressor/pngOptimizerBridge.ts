import { createWorkerPoolClient, type WorkerLike } from '@/core/workers/workerPoolClient';
import type {
  PngOptimizeLevel,
  PngOptimizeRequest,
  PngOptimizeResult,
  PngOptimizeWorkerResponse,
} from './pngOptimizeWorkerProtocol.ts';

export interface PngOptimizeBufferOptions {
  /** oxipng optimization level (0–6). Higher = smaller output, slower. */
  level: PngOptimizeLevel;
  interlace?: boolean;
  optimiseAlpha?: boolean;
  signal?: AbortSignal;
}

export interface CreatePngOptimizerClientOptions {
  canUseWorker?: () => boolean;
  createWorker?: () => WorkerLike;
}

export interface PngOptimizerClient {
  optimize: (source: ArrayBuffer, options: PngOptimizeBufferOptions) => Promise<ArrayBuffer>;
  dispose: (rejectPendingWith?: unknown) => void;
}

const DEFAULT_PNG_OPTIMIZE_REQUEST_TIMEOUT_MS = 2 * 60 * 1000;

function createPngOptimizeAbortError(): Error | DOMException {
  if (typeof DOMException !== 'undefined') {
    return new DOMException('PNG optimization aborted', 'AbortError');
  }

  const error = new Error('PNG optimization aborted');
  error.name = 'AbortError';
  return error;
}

/**
 * Build a PNG optimizer client backed by a single oxipng worker.
 *
 * Exported mainly for tests / advanced callers that want to manage the worker
 * lifecycle themselves. Most call sites should use {@link optimizePngBuffer},
 * which spins up a worker per export and tears it down afterwards.
 */
export function createPngOptimizerClient({
  canUseWorker = () => typeof Worker !== 'undefined',
  createWorker = () =>
    new Worker(new URL('./workers/pngOptimize.worker.ts', import.meta.url), {
      type: 'module',
    }),
}: CreatePngOptimizerClientOptions = {}): PngOptimizerClient {
  const client = createWorkerPoolClient<PngOptimizeWorkerResponse, PngOptimizeResult>({
    label: 'PNG optimization',
    createWorker,
    canUseWorker,
    requestTimeoutMs: DEFAULT_PNG_OPTIMIZE_REQUEST_TIMEOUT_MS,
    getRequestId: (response) => response.requestId,
    isError: (response) => response.type === 'optimize-png-error',
    getError: (response) =>
      (response as { error?: string }).error || 'PNG optimization worker failed',
    getResult: (response) => (response as { result: PngOptimizeResult }).result,
  });

  const optimize = async (
    source: ArrayBuffer,
    options: PngOptimizeBufferOptions,
  ): Promise<ArrayBuffer> => {
    if (!client.canUseWorker) {
      throw new Error('PNG optimization worker is not available in this environment');
    }

    const request: PngOptimizeRequest = {
      type: 'optimize-png',
      sourceBuffer: source,
      level: options.level,
      interlace: options.interlace ?? false,
      optimiseAlpha: options.optimiseAlpha ?? false,
    };

    // The source buffer is transferred (detached on this side) to avoid copying
    // a multi-megabyte payload. Callers must hold their own fallback copy.
    const result = await client.dispatch(request, [source]);
    return result.outputBuffer;
  };

  return {
    optimize,
    dispose: (rejectPendingWith) => client.dispose(rejectPendingWith),
  };
}

/**
 * Losslessly optimize an encoded PNG buffer with oxipng in a worker.
 *
 * A dedicated worker is created per call and terminated once the result
 * settles, so the codec's wasm heap (which can grow large for high-resolution
 * exports) is never held idle. Rejects if the worker is unavailable or the
 * codec fails; callers should fall back to the unoptimized PNG.
 */
export async function optimizePngBuffer(
  source: ArrayBuffer,
  options: PngOptimizeBufferOptions,
): Promise<ArrayBuffer> {
  const client = createPngOptimizerClient();
  const abortError = createPngOptimizeAbortError();
  const handleAbort = () => {
    client.dispose(abortError);
  };

  if (options.signal?.aborted) {
    client.dispose();
    throw abortError;
  }

  options.signal?.addEventListener('abort', handleAbort, { once: true });
  try {
    const result = await client.optimize(source, options);
    if (options.signal?.aborted) {
      throw abortError;
    }
    return result;
  } finally {
    options.signal?.removeEventListener('abort', handleAbort);
    client.dispose();
  }
}
