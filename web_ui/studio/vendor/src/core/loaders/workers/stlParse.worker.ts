/// <reference lib="webworker" />

import { parseStlGeometryData } from '../stlGeometryData';
import type { StlParseWorkerRequest, StlParseWorkerResponse } from '../stlParseWorkerProtocol';

const workerScope = globalThis as unknown as DedicatedWorkerGlobalScope;

async function fetchAssetBuffer(assetUrl: string): Promise<ArrayBuffer> {
    const response = await fetch(assetUrl);
    if (!response.ok) {
        throw new Error(`Failed to fetch STL asset: ${response.status} ${response.statusText}`);
    }

    return await response.arrayBuffer();
}

workerScope.addEventListener('message', async (event: MessageEvent<StlParseWorkerRequest>) => {
    const message = event.data;
    if (!message || message.type !== 'parse-stl') {
        return;
    }

    try {
        const assetBuffer = await fetchAssetBuffer(message.assetUrl);
        const result = parseStlGeometryData(assetBuffer);
        const response: StlParseWorkerResponse = {
            type: 'parse-stl-result',
            requestId: message.requestId,
            result,
        };

        workerScope.postMessage(response, [result.positions, result.normals]);
    } catch (error) {
        const response: StlParseWorkerResponse = {
            type: 'parse-stl-error',
            requestId: message.requestId,
            error: error instanceof Error ? error.message : 'Failed to parse STL in worker',
        };

        workerScope.postMessage(response);
    }
});

export {};
