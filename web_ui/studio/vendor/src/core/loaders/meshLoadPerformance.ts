export type MeshLoadPerformanceFormat = 'collada' | 'obj';

export interface MeshLoadPerformanceEntry {
  assetUrl: string;
  byteLength?: number;
  colladaWasmFeatureCheckMs?: number;
  colladaWasmGeometriesMs?: number;
  colladaWasmImagesMs?: number;
  colladaWasmLibraryNodesMs?: number;
  colladaWasmMaterialsMs?: number;
  colladaWasmUnitScaleMs?: number;
  colladaWasmUpAxisMs?: number;
  colladaWasmVisualSceneMs?: number;
  colladaWasmWriteResultMs?: number;
  format: MeshLoadPerformanceFormat;
  mainThreadBuildMs?: number;
  mainThreadMaterialMs?: number;
  postMessageTransferMs?: number;
  requestDispatchedAtEpochMs?: number;
  requestId?: number;
  roundTripToMainMs?: number;
  wasmDecodeMs?: number;
  wasmInputCopyMs?: number;
  wasmModuleMs?: number;
  wasmParseMs?: number;
  wasmResultCopyMs?: number;
  wasmTotalMs?: number;
  workerFetchMs?: number;
  workerModuleImportMs?: number;
  workerPostedAtEpochMs?: number;
  workerQueueMs?: number;
  workerReceivedAtEpochMs?: number;
  workerTotalMs?: number;
}

type MeshLoadPerformanceTarget = {
  __URDF_STUDIO_MESH_LOAD_PERFORMANCE__?: MeshLoadPerformanceEntry;
  __URDF_STUDIO_MESH_LOAD_PERFORMANCE_HISTORY__?: MeshLoadPerformanceEntry[];
  location?: { search?: string };
};

const MESH_LOAD_PERFORMANCE_HISTORY_LIMIT = 64;

export function readHighResolutionEpochMs(): number {
  if (
    typeof performance !== 'undefined' &&
    typeof performance.now === 'function' &&
    typeof performance.timeOrigin === 'number'
  ) {
    return performance.timeOrigin + performance.now();
  }

  return Date.now();
}

export function durationMs(startEpochMs: number, endEpochMs = readHighResolutionEpochMs()): number {
  return Math.max(0, endEpochMs - startEpochMs);
}

export function snapshotMeshLoadPerformance(
  entry: MeshLoadPerformanceEntry,
): MeshLoadPerformanceEntry {
  return { ...entry };
}

function isMeshLoadPerformanceDebugEnabled(target: MeshLoadPerformanceTarget): boolean {
  try {
    const search = target.location?.search ?? '';
    return new URLSearchParams(search).get('regressionDebug') === '1';
  } catch {
    return false;
  }
}

export function recordMeshLoadPerformance(entry: MeshLoadPerformanceEntry | undefined): void {
  if (!entry) {
    return;
  }

  const target = globalThis as MeshLoadPerformanceTarget;
  if (!isMeshLoadPerformanceDebugEnabled(target)) {
    return;
  }

  const snapshot = snapshotMeshLoadPerformance(entry);
  target.__URDF_STUDIO_MESH_LOAD_PERFORMANCE__ = snapshot;
  const history = Array.isArray(target.__URDF_STUDIO_MESH_LOAD_PERFORMANCE_HISTORY__)
    ? target.__URDF_STUDIO_MESH_LOAD_PERFORMANCE_HISTORY__.slice(
        -(MESH_LOAD_PERFORMANCE_HISTORY_LIMIT - 1),
      )
    : [];
  history.push(snapshot);
  target.__URDF_STUDIO_MESH_LOAD_PERFORMANCE_HISTORY__ = history;
}

export function getMeshLoadPerformanceHistory(
  target: MeshLoadPerformanceTarget = globalThis as MeshLoadPerformanceTarget,
): MeshLoadPerformanceEntry[] {
  return Array.isArray(target.__URDF_STUDIO_MESH_LOAD_PERFORMANCE_HISTORY__)
    ? target.__URDF_STUDIO_MESH_LOAD_PERFORMANCE_HISTORY__.map(snapshotMeshLoadPerformance)
    : [];
}

export function markPostMessageReceived(
  entry: MeshLoadPerformanceEntry | undefined,
): void {
  if (!entry) {
    return;
  }

  const receivedAt = readHighResolutionEpochMs();
  if (typeof entry.workerPostedAtEpochMs === 'number') {
    entry.postMessageTransferMs = durationMs(entry.workerPostedAtEpochMs, receivedAt);
  }
  if (typeof entry.requestDispatchedAtEpochMs === 'number') {
    entry.roundTripToMainMs = durationMs(entry.requestDispatchedAtEpochMs, receivedAt);
  }
  recordMeshLoadPerformance(entry);
}

export function markMainThreadBuildPerformance(
  entry: MeshLoadPerformanceEntry | undefined,
  buildMs: number,
): void {
  if (!entry) {
    return;
  }

  entry.mainThreadBuildMs = buildMs;
  recordMeshLoadPerformance(entry);
}

export function addMainThreadMaterialPerformance(
  entry: MeshLoadPerformanceEntry | undefined,
  materialMs: number,
): void {
  if (!entry) {
    return;
  }

  entry.mainThreadMaterialMs = (entry.mainThreadMaterialMs ?? 0) + materialMs;
  recordMeshLoadPerformance(entry);
}
