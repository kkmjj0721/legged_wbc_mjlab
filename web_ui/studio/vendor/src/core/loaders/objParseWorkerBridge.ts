import {
  createObjectFromSerializedObjData,
  createObjectFromSerializedObjDataAsync,
  type SerializedObjModelData,
} from './objModelData';
import { parseObjModelDataFromBytes } from './objWasmParser';
import {
  createDefaultMeshParseWorker,
  createMeshParseWorkerPoolClient,
  resolveDefaultMeshParseWorkerCount,
  sharedMeshParseWorkerPoolClient,
  type MeshParseWorkerPoolClient,
  type MeshParseWorkerPoolDiagnostics,
} from './meshParseWorkerBridge';
import { type WorkerLike } from '@/core/workers/workerPoolClient';

interface CreateObjParseWorkerPoolClientOptions {
  cacheLimit?: number;
  canUseWorker?: () => boolean;
  createWorker?: () => WorkerLike;
  getWorkerCount?: () => number;
  meshClient?: MeshParseWorkerPoolClient;
}

interface ObjParseWorkerPoolClient {
  clearCache: () => void;
  dispose: (rejectPendingWith?: unknown) => void;
  getDiagnostics: () => MeshParseWorkerPoolDiagnostics;
  load: (assetUrl: string) => Promise<SerializedObjModelData>;
}

const DEFAULT_CACHE_LIMIT = 300;
const FAILURE_CACHE_LIMIT = 200;

async function loadSerializedObjModelDataInline(assetUrl: string): Promise<SerializedObjModelData> {
  const response = await fetch(assetUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch OBJ asset: ${response.status} ${response.statusText}`);
  }

  return await parseObjModelDataFromBytes(await response.arrayBuffer());
}

export function createObjParseWorkerPoolClient({
  cacheLimit = DEFAULT_CACHE_LIMIT,
  canUseWorker = () => typeof Worker !== 'undefined',
  createWorker = createDefaultMeshParseWorker,
  getWorkerCount = resolveDefaultMeshParseWorkerCount,
  meshClient = createMeshParseWorkerPoolClient({
    label: 'OBJ parse',
    canUseWorker,
    createWorker,
    getWorkerCount,
  }),
}: CreateObjParseWorkerPoolClientOptions = {}): ObjParseWorkerPoolClient {
  const pendingLoads = new Map<string, Promise<SerializedObjModelData>>();
  const failureCache = new Map<string, unknown>();
  const resolvedCache = cacheLimit > 0 ? new Map<string, SerializedObjModelData>() : null;

  const getCached = (assetUrl: string): SerializedObjModelData | undefined => {
    return resolvedCache?.get(assetUrl);
  };

  const setCached = (assetUrl: string, result: SerializedObjModelData): void => {
    if (!resolvedCache) return;

    if (resolvedCache.has(assetUrl)) {
      resolvedCache.delete(assetUrl);
    }
    resolvedCache.set(assetUrl, result);

    while (resolvedCache.size > cacheLimit) {
      const oldestKey = resolvedCache.keys().next().value;
      if (oldestKey === undefined) return;
      resolvedCache.delete(oldestKey);
    }
  };

  const rememberFailure = (assetUrl: string, error: unknown): void => {
    if (failureCache.size >= FAILURE_CACHE_LIMIT) {
      const oldestKey = failureCache.keys().next().value;
      if (oldestKey !== undefined) failureCache.delete(oldestKey);
    }
    failureCache.set(assetUrl, error);
  };

  const load = async (assetUrl: string): Promise<SerializedObjModelData> => {
    const cachedResult = getCached(assetUrl);
    if (cachedResult) {
      return cachedResult;
    }

    if (failureCache.has(assetUrl)) {
      throw failureCache.get(assetUrl);
    }

    const pendingLoad = pendingLoads.get(assetUrl);
    if (pendingLoad) {
      return await pendingLoad;
    }

    const nextLoad = (
      meshClient.canUseWorker
        ? meshClient.dispatchObj(assetUrl)
        : loadSerializedObjModelDataInline(assetUrl)
    )
      .then((result) => {
        setCached(assetUrl, result);
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
    return await nextLoad;
  };

  return {
    clearCache: () => {
      resolvedCache?.clear();
      failureCache.clear();
    },
    dispose: (rejectPendingWith) => meshClient.dispose(rejectPendingWith),
    getDiagnostics: () => meshClient.getDiagnostics(),
    load,
  };
}

const sharedObjParseWorkerPoolClient = createObjParseWorkerPoolClient({
  meshClient: sharedMeshParseWorkerPoolClient,
});

export async function loadSerializedObjModelData(
  assetUrl: string,
): Promise<SerializedObjModelData> {
  return await sharedObjParseWorkerPoolClient.load(assetUrl);
}

export function clearObjParseWorkerPoolClientCache(): void {
  sharedObjParseWorkerPoolClient.clearCache();
}

export function disposeObjParseWorkerPoolClient(rejectPendingWith?: unknown): void {
  sharedObjParseWorkerPoolClient.dispose(rejectPendingWith);
}

export { createObjectFromSerializedObjData, createObjectFromSerializedObjDataAsync };
