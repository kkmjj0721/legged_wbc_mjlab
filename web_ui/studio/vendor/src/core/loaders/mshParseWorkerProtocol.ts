import type { SerializedMshGeometryData } from './mshGeometryData';

export interface ParseMshWorkerRequest {
  assetUrl: string;
  requestId: number;
  type: 'parse-msh';
}

export interface ParseMshWorkerResultResponse {
  requestId: number;
  result: SerializedMshGeometryData;
  type: 'parse-msh-result';
}

export interface ParseMshWorkerErrorResponse {
  error: string;
  requestId: number;
  type: 'parse-msh-error';
}

export type MshParseWorkerResponse =
  | ParseMshWorkerResultResponse
  | ParseMshWorkerErrorResponse;
