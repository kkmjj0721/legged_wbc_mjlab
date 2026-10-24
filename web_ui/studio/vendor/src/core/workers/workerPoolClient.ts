/**
 * Generic worker pool client factory.
 *
 * Provides the shared infrastructure used by all worker bridges:
 * - Worker pool creation and lifecycle (create, terminate, pick least-loaded)
 * - Pending request tracking with cleanup
 * - Optional per-request watchdog timeouts for silent worker failures
 * - Error normalization
 * - `workerUnavailable` flag management
 * - Optional LRU cache with configurable limit
 *
 * Two usage patterns are supported via the same factory:
 * - Pool-based (poolSize > 1): STL/OBJ/Collada parse bridges with LRU caching
 * - Single-worker (poolSize = 1, default): USD/ProjectArchive bridges without caching
 */

// ---- Types ----

export interface WorkerLike {
  addEventListener(
    type: 'message' | 'error' | 'messageerror',
    listener: EventListenerOrEventListenerObject,
  ): void;
  removeEventListener(
    type: 'message' | 'error' | 'messageerror',
    listener: EventListenerOrEventListenerObject,
  ): void;
  postMessage(message: unknown, transfer?: Transferable[]): void;
  terminate(): void;
}

export interface WorkerPoolEntry {
  worker: WorkerLike;
  pendingCount: number;
}

export interface PendingRequest<Result, Progress = unknown> {
  resolve: (value: Result) => void;
  reject: (error: unknown) => void;
  workerEntry?: WorkerPoolEntry;
  onProgress?: (progress: Progress) => void;
  timeoutId?: ReturnType<typeof setTimeout>;
}

export interface WorkerPoolClientConfig<Response, Result, Progress = unknown> {
  /** Label for error/log messages (e.g. 'STL parse') */
  label: string;
  /** Factory to create a new Worker instance */
  createWorker: () => WorkerLike;
  /** Check if Worker API is available (default: () => typeof Worker !== 'undefined') */
  canUseWorker?: () => boolean;
  /** Pool size or function returning pool size (default: 1 for single-worker) */
  poolSize?: number | (() => number);
  /** LRU cache limit; 0 or undefined means no caching */
  cacheLimit?: number;
  /** Per-request timeout in ms; 0 or undefined disables the watchdog */
  requestTimeoutMs?: number;

  // Response routing
  getRequestId: (response: Response) => number;
  isError: (response: Response) => boolean;
  getError: (response: Response) => string;
  getResult: (response: Response) => Result;
  /** Optional: check if response is a progress update */
  isProgress?: (response: Response) => boolean;
  /** Optional: extract progress payload and deliver to pending request callback */
  handleProgress?: (response: Response, request: PendingRequest<Result, Progress>) => void;
}

export interface WorkerPoolClient<Result> {
  dispose(rejectPendingWith?: unknown): void;
  clearCache(): void;
  /** Current pending request count (for diagnostics) */
  readonly pendingCount: number;
  /** Current number of initialized workers */
  readonly workerCount: number;
  /** Whether the worker pool has crashed and is unavailable */
  readonly unavailable: boolean;
  /** Whether workers can be used in this environment */
  readonly canUseWorker: boolean;
  /** Ensure a worker is initialized and pick the least-loaded one */
  ensureWorker(): WorkerPoolEntry;
  /** Dispatch a request to the worker pool and return a promise for the result */
  dispatch(
    request: unknown,
    transfer?: Transferable[],
    onProgress?: (progress: unknown) => void,
  ): Promise<Result>;
  /** Look up cached result by key */
  getCached(key: string): Result | undefined;
  /** Store result in cache */
  setCached(key: string, result: Result): void;
}

// ---- Shared Utilities ----

export function createWorkerError(
  event: ErrorEvent | { error?: unknown; message?: string },
  label: string,
): Error {
  if ((event as ErrorEvent).error instanceof Error) {
    return (event as ErrorEvent).error;
  }

  return new Error((event as { message?: string }).message || `${label} worker failed`);
}

function createWorkerDisposedError(label: string): Error {
  return new Error(`${label} worker disposed`);
}

function createWorkerTimeoutError(label: string, requestId: number, timeoutMs: number): Error {
  return new Error(
    `${label} worker did not respond within ${timeoutMs} ms. Request id: ${requestId}.`,
  );
}

export function resolveDefaultWorkerCount(): number {
  if (typeof navigator === 'undefined') {
    return 1;
  }

  const hardwareConcurrency = Number(navigator.hardwareConcurrency || 2);
  return Math.max(1, Math.min(10, hardwareConcurrency - 1));
}

function touchCacheEntry<Result>(
  resolvedCache: Map<string, Result> | null,
  cacheLimit: number,
  key: string,
  result: Result,
): void {
  if (!resolvedCache) return;

  if (resolvedCache.has(key)) {
    resolvedCache.delete(key);
  }
  resolvedCache.set(key, result);

  while (resolvedCache.size > cacheLimit) {
    const oldestEntry = resolvedCache.keys().next();
    if (oldestEntry.done) return;
    resolvedCache.delete(oldestEntry.value);
  }
}

function pickLeastLoadedWorker(workerPool: WorkerPoolEntry[]): WorkerPoolEntry {
  let best = workerPool[0];
  for (let i = 1; i < workerPool.length; i += 1) {
    if (workerPool[i].pendingCount < best.pendingCount) {
      best = workerPool[i];
    }
  }
  return best;
}

function rejectPendingRequestsForWorker<Result, Progress>(
  pendingRequests: Map<number, PendingRequest<Result, Progress>>,
  entry: WorkerPoolEntry,
  clearPendingRequest: (requestId: number) => PendingRequest<Result, Progress> | null,
  rejectPendingWith: unknown,
): void {
  Array.from(pendingRequests.entries()).forEach(([requestId, request]) => {
    if (request.workerEntry !== entry) return;

    clearPendingRequest(requestId);
    request.reject(rejectPendingWith);
  });
}

interface RegisterRequestTimeoutArgs<Result, Progress> {
  disposeWorkerEntry: (entry: WorkerPoolEntry, rejectPendingWith: unknown) => void;
  entry: WorkerPoolEntry;
  label: string;
  pending: PendingRequest<Result, Progress>;
  requestId: number;
  requestTimeoutMs: number;
}

function registerRequestTimeoutForWorker<Result, Progress>({
  disposeWorkerEntry,
  entry,
  label,
  pending,
  requestId,
  requestTimeoutMs,
}: RegisterRequestTimeoutArgs<Result, Progress>): void {
  if (!requestTimeoutMs || !Number.isFinite(requestTimeoutMs) || requestTimeoutMs <= 0) return;

  pending.timeoutId = setTimeout(() => {
    const timeoutError = createWorkerTimeoutError(label, requestId, requestTimeoutMs);
    console.error(`[${label}WorkerBridge] ${label} worker request timed out.`, timeoutError);
    disposeWorkerEntry(entry, timeoutError);
  }, requestTimeoutMs);
}

// ---- Factory ----

export function createWorkerPoolClient<Response, Result, Progress = unknown>(
  config: WorkerPoolClientConfig<Response, Result, Progress>,
): WorkerPoolClient<Result> {
  const { label, createWorker, getRequestId, isError, getError, getResult } = config;
  const { isProgress, handleProgress } = config;
  const canUseWorker = config.canUseWorker ?? (() => typeof Worker !== 'undefined');
  const cacheLimit = config.cacheLimit ?? 0;
  const requestTimeoutMs = config.requestTimeoutMs ?? 0;

  const poolSize = config.poolSize ?? 1;
  const resolvePoolSize = typeof poolSize === 'function' ? poolSize : () => poolSize;

  const workerPool: WorkerPoolEntry[] = [];
  const pendingRequests = new Map<number, PendingRequest<Result, Progress>>();
  const resolvedCache = cacheLimit > 0 ? new Map<string, Result>() : null;
  let workerUnavailable = false;
  let nextRequestId = 1;

  // ---- Pending Request Management ----

  function clearPendingRequest(requestId: number): PendingRequest<Result, Progress> | null {
    const pendingRequest = pendingRequests.get(requestId) ?? null;
    if (!pendingRequest) return null;

    pendingRequests.delete(requestId);
    if (pendingRequest.timeoutId !== undefined) {
      clearTimeout(pendingRequest.timeoutId);
      pendingRequest.timeoutId = undefined;
    }
    if (pendingRequest.workerEntry) {
      pendingRequest.workerEntry.pendingCount = Math.max(
        0,
        pendingRequest.workerEntry.pendingCount - 1,
      );
    }
    return pendingRequest;
  }

  // ---- Worker Lifecycle ----

  function handleWorkerMessage(event: MessageEvent<Response>): void {
    let requestId: number | null = null;

    try {
      const message = event.data;
      if (!message) return;

      requestId = getRequestId(message);

      // Progress messages don't clear the pending request
      if (isProgress?.(message)) {
        const pendingRequest = pendingRequests.get(requestId);
        if (pendingRequest) {
          handleProgress!(message, pendingRequest);
        }
        return;
      }

      const pendingRequest = clearPendingRequest(requestId);
      if (!pendingRequest) return;

      try {
        if (isError(message)) {
          const workerError = new Error(getError(message) || `${label} worker failed`);
          console.error(`[${label}WorkerBridge] Worker returned an error.`, workerError);
          pendingRequest.reject(workerError);
          return;
        }

        pendingRequest.resolve(getResult(message));
      } catch (error) {
        pendingRequest.reject(error);
      }
    } catch (error) {
      if (requestId != null) {
        clearPendingRequest(requestId)?.reject(error);
        return;
      }

      console.error(`[${label}WorkerBridge] Failed to process worker response.`, error);
      workerUnavailable = true;
      disposePool(error);
    }
  }

  function handleWorkerError(event: ErrorEvent): void {
    const workerError = createWorkerError(event, label);
    console.error(`[${label}WorkerBridge] ${label} worker crashed.`, workerError);
    workerUnavailable = true;
    disposePool(workerError);
  }

  function handleWorkerMessageError(): void {
    const workerError = new Error(`${label} worker message transfer failed`);
    console.error(`[${label}WorkerBridge] ${label} worker message transfer failed.`, workerError);
    workerUnavailable = true;
    disposePool(workerError);
  }

  function createWorkerEntry(): WorkerPoolEntry {
    const worker = createWorker();
    worker.addEventListener('message', handleWorkerMessage as EventListener);
    worker.addEventListener('error', handleWorkerError as EventListener);
    worker.addEventListener('messageerror', handleWorkerMessageError as EventListener);
    const entry = { worker, pendingCount: 0 };
    workerPool.push(entry);
    return entry;
  }

  function ensureWorker(): WorkerPoolEntry {
    if (workerUnavailable && workerPool.length === 0) {
      workerUnavailable = false;
    }

    if (workerPool.length === 0) {
      return createWorkerEntry();
    }

    const best = pickLeastLoadedWorker(workerPool);
    const count = Math.max(1, resolvePoolSize());
    if (best.pendingCount > 0 && workerPool.length < count) {
      return createWorkerEntry();
    }

    return best;
  }

  function disposeWorkerEntry(entry: WorkerPoolEntry, rejectPendingWith?: unknown): void {
    const entryIndex = workerPool.indexOf(entry);
    if (entryIndex >= 0) {
      workerPool.splice(entryIndex, 1);
    }

    entry.worker.removeEventListener('message', handleWorkerMessage as EventListener);
    entry.worker.removeEventListener('error', handleWorkerError as EventListener);
    entry.worker.removeEventListener('messageerror', handleWorkerMessageError as EventListener);
    entry.worker.terminate();
    entry.pendingCount = 0;

    if (rejectPendingWith !== undefined) {
      rejectPendingRequestsForWorker(
        pendingRequests,
        entry,
        clearPendingRequest,
        rejectPendingWith,
      );
    }
  }

  function disposePool(rejectPendingWith?: unknown): void {
    const rejectionReason = rejectPendingWith ?? createWorkerDisposedError(label);

    workerPool.forEach((entry) => {
      entry.worker.removeEventListener('message', handleWorkerMessage as EventListener);
      entry.worker.removeEventListener('error', handleWorkerError as EventListener);
      entry.worker.removeEventListener('messageerror', handleWorkerMessageError as EventListener);
      entry.worker.terminate();
      entry.pendingCount = 0;
    });
    workerPool.length = 0;

    Array.from(pendingRequests.entries()).forEach(([requestId, request]) => {
      clearPendingRequest(requestId);
      request.reject(rejectionReason);
    });
  }

  // ---- Dispatch ----

  function dispatch(
    request: unknown,
    transfer?: Transferable[],
    onProgress?: (progress: unknown) => void,
  ): Promise<Result> {
    if (workerUnavailable && workerPool.length > 0) {
      throw new Error(`${label} worker is unavailable`);
    }

    if (!canUseWorker()) {
      throw new Error(`${label} worker is not available in this environment`);
    }

    const entry = ensureWorker();
    const requestId = nextRequestId;
    nextRequestId += 1;

    return new Promise<Result>((resolveRequest, rejectRequest) => {
      const pending: PendingRequest<Result, Progress> = {
        resolve: resolveRequest,
        reject: rejectRequest,
        workerEntry: entry,
        onProgress: onProgress as ((progress: Progress) => void) | undefined,
      };
      pendingRequests.set(requestId, pending);
      entry.pendingCount += 1;
      registerRequestTimeoutForWorker({
        disposeWorkerEntry,
        entry,
        label,
        pending,
        requestId,
        requestTimeoutMs,
      });

      try {
        entry.worker.postMessage({ ...(request as Record<string, unknown>), requestId }, transfer);
      } catch (error) {
        console.error(`[${label}WorkerBridge] Failed to dispatch request to worker.`, error);
        clearPendingRequest(requestId);
        workerUnavailable = true;
        disposePool(error);
        rejectRequest(error);
      }
    });
  }

  return {
    dispose: disposePool,
    clearCache: () => resolvedCache?.clear(),
    get pendingCount() {
      return pendingRequests.size;
    },
    get workerCount() {
      return workerPool.length;
    },
    get unavailable() {
      return workerUnavailable;
    },
    get canUseWorker() {
      return canUseWorker();
    },
    ensureWorker,
    dispatch,
    getCached: (key) => resolvedCache?.get(key),
    setCached: (key, result) => touchCacheEntry(resolvedCache, cacheLimit, key, result),
  };
}
