/// <reference lib="webworker" />

// Use the single-threaded oxipng codec directly instead of the package's
// default `init()` entry. The dev/preview server is cross-origin isolated, so
// the default entry would otherwise select the multi-threaded (wasm-bindgen-rayon)
// build and spawn its own helper sub-workers. The single-threaded build is
// deterministic, needs no SharedArrayBuffer, and is plenty fast for an
// occasional export. The wasm binary is provided explicitly via a Vite `?url`
// import so bundling does not depend on the codec's relative-fetch fallback.
import initOxipngWasm, {
  optimise as runOxipng,
} from '@jsquash/oxipng/codec/pkg/squoosh_oxipng.js';
import OXIPNG_WASM_URL from '@jsquash/oxipng/codec/pkg/squoosh_oxipng_bg.wasm?url';
import type {
  PngOptimizeWorkerRequest,
  PngOptimizeWorkerResponse,
} from '../pngOptimizeWorkerProtocol.ts';

const workerScope = globalThis as unknown as DedicatedWorkerGlobalScope;

let wasmReady: Promise<unknown> | null = null;

function ensureWasmReady(): Promise<unknown> {
  if (!wasmReady) {
    wasmReady = initOxipngWasm(OXIPNG_WASM_URL);
  }
  return wasmReady;
}

workerScope.addEventListener('message', (event: MessageEvent<PngOptimizeWorkerRequest>) => {
  const message = event.data;
  if (!message || message.type !== 'optimize-png') {
    return;
  }

  const { requestId, sourceBuffer, level, interlace, optimiseAlpha } = message;

  void (async () => {
    try {
      await ensureWasmReady();
      const optimised = runOxipng(new Uint8Array(sourceBuffer), level, interlace, optimiseAlpha);
      // `optimised` may be a view onto wasm memory; copy into a standalone
      // ArrayBuffer so it survives detaching and can be transferred back.
      const outputBuffer = optimised.slice().buffer;

      const response: PngOptimizeWorkerResponse = {
        type: 'optimize-png-result',
        requestId,
        result: { outputBuffer },
      };
      workerScope.postMessage(response, [outputBuffer]);
    } catch (error) {
      const response: PngOptimizeWorkerResponse = {
        type: 'optimize-png-error',
        requestId,
        error: error instanceof Error ? error.message : 'Failed to optimize PNG in worker',
      };
      workerScope.postMessage(response);
    }
  })();
});

export {};
