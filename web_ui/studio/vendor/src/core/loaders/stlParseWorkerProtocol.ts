import type { SerializedStlGeometryData } from './stlGeometryData';

export interface ParseStlWorkerRequest {
    type: 'parse-stl';
    requestId: number;
    assetUrl: string;
}

export interface ParseStlWorkerSuccessResponse {
    type: 'parse-stl-result';
    requestId: number;
    result: SerializedStlGeometryData;
}

export interface ParseStlWorkerErrorResponse {
    type: 'parse-stl-error';
    requestId: number;
    error: string;
}

export type StlParseWorkerRequest = ParseStlWorkerRequest;
export type StlParseWorkerResponse = ParseStlWorkerSuccessResponse | ParseStlWorkerErrorResponse;
