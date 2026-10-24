import type { SerializedColladaSceneData } from './colladaWorkerSceneData';

export interface ParseColladaWorkerRequest {
  assetUrl: string;
  dispatchedAtEpochMs?: number;
  requestId: number;
  type: 'parse-collada';
}

export interface ParseColladaWorkerResultResponse {
  requestId: number;
  result: SerializedColladaSceneData;
  type: 'parse-collada-result';
}

export interface ParseColladaWorkerErrorResponse {
  error: string;
  requestId: number;
  type: 'parse-collada-error';
}

export type ColladaParseWorkerResponse =
  | ParseColladaWorkerErrorResponse
  | ParseColladaWorkerResultResponse;
