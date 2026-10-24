import type { SerializedObjModelData } from './objModelData';

export interface ParseObjWorkerRequest {
    assetUrl: string;
    dispatchedAtEpochMs?: number;
    requestId: number;
    type: 'parse-obj';
}

export interface ParseObjWorkerResultResponse {
    requestId: number;
    result: SerializedObjModelData;
    type: 'parse-obj-result';
}

export interface ParseObjWorkerErrorResponse {
    error: string;
    requestId: number;
    type: 'parse-obj-error';
}

export type ObjParseWorkerResponse = ParseObjWorkerResultResponse | ParseObjWorkerErrorResponse;
