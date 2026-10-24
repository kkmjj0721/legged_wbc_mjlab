import {
  cloneSerializedMshGeometryData,
  parseMshGeometryData,
  type SerializedMshGeometryData,
} from './mshGeometryData';
import type { MshParseWorkerResponse, ParseMshWorkerRequest } from './mshParseWorkerProtocol';
import {
  createWorkerPoolClient,
  resolveDefaultWorkerCount,
  type WorkerLike,
} from '@/core/workers/workerPoolClient';

interface CreateMshParseWorkerPoolClientOptions {
  cacheLimit?: number;
  canUseWorker?: () => boolean;
  createWorker?: () => WorkerLike;
  getWorkerCount?: () => number;
}

interface MshParseWorkerPoolClient {
  clearCache: () => void;
  dispose: (rejectPendingWith?: unknown) => void;
  load: (assetUrl: string) => Promise<SerializedMshGeometryData>;
}

const DEFAULT_CACHE_LIMIT = 300;
const FAILURE_CACHE_LIMIT = 200;
const DEFAULT_MSH_PARSE_REQUEST_TIMEOUT_MS = 5 * 60 * 1000;

async function loadSerializedMshGeometryDataInline(
  assetUrl: string,
): Promise<SerializedMshGeometryData> {
  const response = await fetch(assetUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch legacy MSH asset: ${response.status} ${response.statusText}`);
  }

  return parseMshGeometryData(await response.arrayBuffer());
}

export function createMshParseWorkerPoolClient({
  cacheLimit = DEFAULT_CACHE_LIMIT,
  canUseWorker = () => typeof Worker !== 'undefined',
  createWorker = () =>
    new Worker(new URL('./workers/mshParse.worker.ts', import.meta.url), { type: 'module' }),
  getWorkerCount = resolveDefaultWorkerCount,
}: CreateMshParseWorkerPoolClientOptions = {}): MshParseWorkerPoolClient {
  const client = createWorkerPoolClient<MshParseWorkerResponse, SerializedMshGeometryData>({
    label: 'MSH parse',
    createWorker,
    canUseWorker,
    poolSize: getWorkerCount,
    cacheLimit,
    requestTimeoutMs: DEFAULT_MSH_PARSE_REQUEST_TIMEOUT_MS,
    getRequestId: (response) => response.requestId,
    isError: (response) => response.type === 'parse-msh-error',
    getError: (response) => (response as { error?: string }).error || 'MSH parse worker failed',
    getResult: (response) =>
      cloneSerializedMshGeometryData(
        (response as { result: SerializedMshGeometryData }).result,
      ),
  });

  const pendingLoads = new Map<string, Promise<SerializedMshGeometryData>>();
  const failureCache = new Map<string, unknown>();

  const rememberFailure = (assetUrl: string, error: unknown): void => {
    if (failureCache.size >= FAILURE_CACHE_LIMIT) {
      const oldestKey = failureCache.keys().next().value;
      if (oldestKey !== undefined) failureCache.delete(oldestKey);
    }
    failureCache.set(assetUrl, error);
  };

  const load = async (assetUrl: string): Promise<SerializedMshGeometryData> => {
    const cachedResult = client.getCached(assetUrl);
    if (cachedResult) {
      return cloneSerializedMshGeometryData(cachedResult);
    }

    if (failureCache.has(assetUrl)) {
      throw failureCache.get(assetUrl);
    }

    const pendingLoad = pendingLoads.get(assetUrl);
    if (pendingLoad) {
      return cloneSerializedMshGeometryData(await pendingLoad);
    }

    const nextLoad = (async () => {
      if (!client.canUseWorker || client.unavailable) {
        return await loadSerializedMshGeometryDataInline(assetUrl);
      }

      try {
        return await client.dispatch({
          type: 'parse-msh',
          assetUrl,
        } satisfies Omit<ParseMshWorkerRequest, 'requestId'>);
      } catch (error) {
        if (/worker/i.test(error instanceof Error ? error.message : String(error))) {
          return await loadSerializedMshGeometryDataInline(assetUrl);
        }
        throw error;
      }
    })()
      .then((result) => {
        client.setCached(assetUrl, cloneSerializedMshGeometryData(result));
        return result;
      })
      .catch((error) => {
        rememberFailure(assetUrl, error);
        throw error;
      })
      .finally(() => {
        pendingLoads.delete(assetUrl);
      });

    pendingLoads.set(assetUrl, nextLoad);
    return cloneSerializedMshGeometryData(await nextLoad);
  };

  return {
    clearCache: () => {
      client.clearCache();
      failureCache.clear();
    },
    dispose: (rejectPendingWith) => client.dispose(rejectPendingWith),
    load,
  };
}

const sharedMshParseWorkerPoolClient = createMshParseWorkerPoolClient();

export async function loadSerializedMshGeometryData(
  assetUrl: string,
): Promise<SerializedMshGeometryData> {
  return await sharedMshParseWorkerPoolClient.load(assetUrl);
}

export function clearMshParseWorkerPoolClientCache(): void {
  sharedMshParseWorkerPoolClient.clearCache();
}

export function disposeMshParseWorkerPoolClient(rejectPendingWith?: unknown): void {
  sharedMshParseWorkerPoolClient.dispose(rejectPendingWith);
}
