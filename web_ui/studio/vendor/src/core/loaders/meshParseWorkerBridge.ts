import type { SerializedColladaSceneData } from './colladaWorkerSceneData';
import type {
  ColladaParseWorkerResponse,
  ParseColladaWorkerRequest,
} from './colladaParseWorkerProtocol';
import type { SerializedObjModelData } from './objModelData';
import type { ObjParseWorkerResponse, ParseObjWorkerRequest } from './objParseWorkerProtocol';
import {
  createWorkerPoolClient,
  type WorkerLike,
} from '@/core/workers/workerPoolClient';
import {
  markPostMessageReceived,
  readHighResolutionEpochMs,
} from './meshLoadPerformance';

type MeshParseWorkerResponse = ColladaParseWorkerResponse | ObjParseWorkerResponse;
type MeshParseWorkerResult = SerializedColladaSceneData | SerializedObjModelData;

interface CreateMeshParseWorkerPoolClientOptions {
  canUseWorker?: () => boolean;
  createWorker?: () => WorkerLike;
  getWorkerCount?: () => number;
  label?: string;
}

export interface MeshParseWorkerPoolDiagnostics {
  pendingCount: number;
  unavailable: boolean;
  workerCount: number;
  workerLimit: number;
}

export interface MeshParseWorkerPoolClient {
  dispatchCollada: (assetUrl: string) => Promise<SerializedColladaSceneData>;
  dispatchObj: (assetUrl: string) => Promise<SerializedObjModelData>;
  dispose: (rejectPendingWith?: unknown) => void;
  getDiagnostics: () => MeshParseWorkerPoolDiagnostics;
  readonly canUseWorker: boolean;
  readonly pendingCount: number;
  readonly unavailable: boolean;
  readonly workerCount: number;
}

const DEFAULT_MESH_PARSE_WORKER_LIMIT = 12;
const DEFAULT_MESH_PARSE_REQUEST_TIMEOUT_MS = 5 * 60 * 1000;

export function resolveDefaultMeshParseWorkerCount(): number {
  if (typeof navigator === 'undefined') {
    return 1;
  }

  const hardwareConcurrency = Number(navigator.hardwareConcurrency || 2);
  return Math.max(2, Math.min(DEFAULT_MESH_PARSE_WORKER_LIMIT, hardwareConcurrency - 1));
}

export function createDefaultMeshParseWorker(): WorkerLike {
  return new Worker(new URL('./workers/meshParse.worker.ts', import.meta.url), { type: 'module' });
}

function getMeshParseRequestId(response: MeshParseWorkerResponse): number {
  return response.requestId;
}

function isMeshParseError(response: MeshParseWorkerResponse): boolean {
  return response.type === 'parse-collada-error' || response.type === 'parse-obj-error';
}

function getMeshParseError(response: MeshParseWorkerResponse): string {
  if (response.type === 'parse-collada-error') {
    return response.error || 'Collada parse worker failed';
  }

  if (response.type === 'parse-obj-error') {
    return response.error || 'OBJ parse worker failed';
  }

  return 'Mesh parse worker failed';
}

function getMeshParseResult(response: MeshParseWorkerResponse): MeshParseWorkerResult {
  if (response.type === 'parse-collada-result' || response.type === 'parse-obj-result') {
    markPostMessageReceived(response.result.loadPerformance);
    return response.result;
  }

  throw new Error(getMeshParseError(response));
}

function assertWorkerAvailable(client: {
  canUseWorker: boolean;
}, label: string): void {
  if (!client.canUseWorker) {
    throw new Error(`${label} worker is not available in this environment`);
  }
}

export function createMeshParseWorkerPoolClient({
  canUseWorker = () => typeof Worker !== 'undefined',
  createWorker = createDefaultMeshParseWorker,
  getWorkerCount = resolveDefaultMeshParseWorkerCount,
  label = 'Mesh parse',
}: CreateMeshParseWorkerPoolClientOptions = {}): MeshParseWorkerPoolClient {
  const client = createWorkerPoolClient<MeshParseWorkerResponse, MeshParseWorkerResult>({
    label,
    createWorker,
    canUseWorker,
    poolSize: getWorkerCount,
    requestTimeoutMs: DEFAULT_MESH_PARSE_REQUEST_TIMEOUT_MS,
    getRequestId: getMeshParseRequestId,
    isError: isMeshParseError,
    getError: getMeshParseError,
    getResult: getMeshParseResult,
  });

  const dispatchObj = async (assetUrl: string): Promise<SerializedObjModelData> => {
    assertWorkerAvailable(client, 'OBJ parse');
    return (await client.dispatch({
      type: 'parse-obj',
      assetUrl,
      dispatchedAtEpochMs: readHighResolutionEpochMs(),
    } satisfies Omit<ParseObjWorkerRequest, 'requestId'>)) as SerializedObjModelData;
  };

  const dispatchCollada = async (
    assetUrl: string,
  ): Promise<SerializedColladaSceneData> => {
    assertWorkerAvailable(client, 'Collada parse');
    return (await client.dispatch({
      type: 'parse-collada',
      assetUrl,
      dispatchedAtEpochMs: readHighResolutionEpochMs(),
    } satisfies Omit<ParseColladaWorkerRequest, 'requestId'>)) as SerializedColladaSceneData;
  };

  return {
    dispatchCollada,
    dispatchObj,
    dispose: (rejectPendingWith) => client.dispose(rejectPendingWith),
    get canUseWorker() {
      return client.canUseWorker;
    },
    get pendingCount() {
      return client.pendingCount;
    },
    get unavailable() {
      return client.unavailable;
    },
    get workerCount() {
      return client.workerCount;
    },
    getDiagnostics: () => ({
      pendingCount: client.pendingCount,
      unavailable: client.unavailable,
      workerCount: client.workerCount,
      workerLimit: Math.max(1, getWorkerCount()),
    }),
  };
}

export const sharedMeshParseWorkerPoolClient = createMeshParseWorkerPoolClient();
