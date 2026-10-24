/// <reference lib="webworker" />

import {
  collectSerializedMshTransferables,
  parseMshGeometryData,
} from '../mshGeometryData';
import type { MshParseWorkerResponse, ParseMshWorkerRequest } from '../mshParseWorkerProtocol';

const workerScope = globalThis as unknown as DedicatedWorkerGlobalScope;

async function loadMshBytes(assetUrl: string): Promise<ArrayBuffer> {
  const response = await fetch(assetUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch legacy MSH asset: ${response.status} ${response.statusText}`);
  }

  return await response.arrayBuffer();
}

workerScope.addEventListener('message', async (event: MessageEvent<ParseMshWorkerRequest>) => {
  const message = event.data;
  if (!message || message.type !== 'parse-msh') {
    return;
  }

  try {
    const mshBytes = await loadMshBytes(message.assetUrl);
    const result = parseMshGeometryData(mshBytes);
    const response: MshParseWorkerResponse = {
      type: 'parse-msh-result',
      requestId: message.requestId,
      result,
    };
    workerScope.postMessage(response, collectSerializedMshTransferables(result));
  } catch (error) {
    const response: MshParseWorkerResponse = {
      type: 'parse-msh-error',
      requestId: message.requestId,
      error: error instanceof Error ? error.message : 'MSH parse worker failed',
    };
    workerScope.postMessage(response);
  }
});

export {};
