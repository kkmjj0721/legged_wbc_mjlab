/// <reference lib="webworker" />

import type {
  ColladaParseWorkerResponse,
  ParseColladaWorkerRequest,
} from '../colladaParseWorkerProtocol';
import type { ObjParseWorkerResponse, ParseObjWorkerRequest } from '../objParseWorkerProtocol';
import {
  durationMs,
  type MeshLoadPerformanceEntry,
  readHighResolutionEpochMs,
} from '../meshLoadPerformance';

declare const self: DedicatedWorkerGlobalScope;

type MeshParseWorkerRequest = ParseColladaWorkerRequest | ParseObjWorkerRequest;

async function loadAssetBytes(assetUrl: string, formatLabel: string): Promise<{
  bytes: ArrayBuffer;
  fetchMs: number;
}> {
  const startedAt = readHighResolutionEpochMs();
  const response = await fetch(assetUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${formatLabel} asset: ${response.status} ${response.statusText}`);
  }

  const bytes = await response.arrayBuffer();
  return {
    bytes,
    fetchMs: durationMs(startedAt),
  };
}

function extractUrlBase(assetUrl: string): string {
  const queryIndex = assetUrl.search(/[?#]/);
  const cleanUrl = queryIndex >= 0 ? assetUrl.slice(0, queryIndex) : assetUrl;
  const slashIndex = cleanUrl.lastIndexOf('/');
  return slashIndex >= 0 ? assetUrl.slice(0, slashIndex + 1) : '';
}

async function handleObjParse(message: ParseObjWorkerRequest): Promise<void> {
  const workerReceivedAt = readHighResolutionEpochMs();
  try {
    const moduleImportStartedAt = readHighResolutionEpochMs();
    const [{ collectSerializedObjTransferables }, { parseObjModelDataFromBytes }] =
      await Promise.all([
        import('../objModelData'),
        import('../objWasmParser'),
      ]);
    const workerModuleImportMs = durationMs(moduleImportStartedAt);
    const { bytes: objBytes, fetchMs } = await loadAssetBytes(message.assetUrl, 'OBJ');
    const loadPerformance: MeshLoadPerformanceEntry = {
      assetUrl: message.assetUrl,
      byteLength: objBytes.byteLength,
      format: 'obj',
      requestDispatchedAtEpochMs: message.dispatchedAtEpochMs,
      requestId: message.requestId,
      workerFetchMs: fetchMs,
      workerModuleImportMs,
      workerQueueMs:
        typeof message.dispatchedAtEpochMs === 'number'
          ? durationMs(message.dispatchedAtEpochMs, workerReceivedAt)
          : undefined,
      workerReceivedAtEpochMs: workerReceivedAt,
    };
    const result = await parseObjModelDataFromBytes(objBytes, loadPerformance);
    const workerPostedAt = readHighResolutionEpochMs();
    loadPerformance.workerTotalMs = durationMs(workerReceivedAt, workerPostedAt);
    loadPerformance.workerPostedAtEpochMs = workerPostedAt;
    const response: ObjParseWorkerResponse = {
      type: 'parse-obj-result',
      requestId: message.requestId,
      result,
    };
    self.postMessage(response, collectSerializedObjTransferables(result));
  } catch (error) {
    const normalized = error instanceof Error ? error : new Error(String(error));
    console.error(`[MeshParseWorker] Failed to parse OBJ "${message.assetUrl}":`, normalized);
    const response: ObjParseWorkerResponse = {
      type: 'parse-obj-error',
      requestId: message.requestId,
      error: normalized.message || 'OBJ parse worker failed',
    };
    self.postMessage(response);
  }
}

async function handleColladaParse(message: ParseColladaWorkerRequest): Promise<void> {
  const workerReceivedAt = readHighResolutionEpochMs();
  try {
    const moduleImportStartedAt = readHighResolutionEpochMs();
    const [
      { collectSerializedColladaTransferables },
      { parseColladaMeshDataWithWasm },
      { ensureWorkerXmlDomApis },
    ] = await Promise.all([
      import('../colladaWorkerSceneData'),
      import('../colladaWasmParser'),
      import('../../utils/ensureWorkerXmlDomApis'),
    ]);
    const workerModuleImportMs = durationMs(moduleImportStartedAt);
    ensureWorkerXmlDomApis();
    const { bytes: colladaBytes, fetchMs } = await loadAssetBytes(message.assetUrl, 'Collada');
    const loadPerformance: MeshLoadPerformanceEntry = {
      assetUrl: message.assetUrl,
      byteLength: colladaBytes.byteLength,
      format: 'collada',
      requestDispatchedAtEpochMs: message.dispatchedAtEpochMs,
      requestId: message.requestId,
      workerFetchMs: fetchMs,
      workerModuleImportMs,
      workerQueueMs:
        typeof message.dispatchedAtEpochMs === 'number'
          ? durationMs(message.dispatchedAtEpochMs, workerReceivedAt)
          : undefined,
      workerReceivedAtEpochMs: workerReceivedAt,
    };
    const result = await parseColladaMeshDataWithWasm(
      colladaBytes,
      extractUrlBase(message.assetUrl),
      loadPerformance,
    );
    const workerPostedAt = readHighResolutionEpochMs();
    loadPerformance.workerTotalMs = durationMs(workerReceivedAt, workerPostedAt);
    loadPerformance.workerPostedAtEpochMs = workerPostedAt;
    const response: ColladaParseWorkerResponse = {
      type: 'parse-collada-result',
      requestId: message.requestId,
      result,
    };
    self.postMessage(response, collectSerializedColladaTransferables(result));
  } catch (error) {
    const normalized = error instanceof Error ? error : new Error(String(error));
    console.error(`[MeshParseWorker] Failed to parse Collada "${message.assetUrl}":`, normalized);
    const response: ColladaParseWorkerResponse = {
      type: 'parse-collada-error',
      requestId: message.requestId,
      error: normalized.message || 'Collada parse worker failed',
    };
    self.postMessage(response);
  }
}

self.addEventListener('message', (event: MessageEvent<MeshParseWorkerRequest>) => {
  const message = event.data;
  if (!message) {
    return;
  }

  if (message.type === 'parse-obj') {
    void handleObjParse(message);
    return;
  }

  if (message.type === 'parse-collada') {
    void handleColladaParse(message);
  }
});

export {};
