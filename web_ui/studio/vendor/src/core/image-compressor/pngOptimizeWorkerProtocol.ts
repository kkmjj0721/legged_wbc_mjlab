/**
 * Message protocol for the PNG optimization worker.
 *
 * The worker losslessly re-compresses an already-encoded PNG buffer with
 * oxipng (Squoosh codec). Buffers are transferred in both directions so the
 * potentially large pixel payloads avoid a structured-clone copy.
 */

/** oxipng optimization level (0 = fastest/least, 6 = slowest/most). */
export type PngOptimizeLevel = 0 | 1 | 2 | 3 | 4 | 5 | 6;

/** Request payload handed to {@link createWorkerPoolClient}'s `dispatch`. */
export interface PngOptimizeRequest {
  type: 'optimize-png';
  /** Source PNG bytes (transferred — detached on the caller after dispatch). */
  sourceBuffer: ArrayBuffer;
  level: PngOptimizeLevel;
  /** Emit an interlaced PNG. Kept off for snapshot exports. */
  interlace: boolean;
  /**
   * Allow oxipng to rewrite the RGB of fully-transparent pixels. Visually
   * lossless, but kept off so the output is byte-faithful to the source pixels.
   */
  optimiseAlpha: boolean;
}

/** Request as received inside the worker (the pool client appends `requestId`). */
export interface PngOptimizeWorkerRequest extends PngOptimizeRequest {
  requestId: number;
}

export interface PngOptimizeResult {
  /** Optimized PNG bytes (transferred back to the caller). */
  outputBuffer: ArrayBuffer;
}

export interface PngOptimizeSuccessResponse {
  type: 'optimize-png-result';
  requestId: number;
  result: PngOptimizeResult;
}

export interface PngOptimizeErrorResponse {
  type: 'optimize-png-error';
  requestId: number;
  error: string;
}

export type PngOptimizeWorkerResponse = PngOptimizeSuccessResponse | PngOptimizeErrorResponse;
