import { normalizeSnapshotCaptureOptions, type SnapshotCaptureOptions } from './snapshotConfig';

const SNAPSHOT_PREVIEW_LONG_EDGE = 800;

export function resolveSnapshotPreviewCaptureOptions(
  options?: Partial<SnapshotCaptureOptions> | null,
): SnapshotCaptureOptions {
  const normalized = normalizeSnapshotCaptureOptions(options);

  return {
    ...normalized,
    // The on-screen snapshot dialog preview is capped at 800px long-edge for
    // speed. Automation callers (batch thumbnail exporter) set
    // bypassPreviewResolutionCap to honour --long-edge and emit full-resolution
    // output instead of being silently downscaled to 800.
    longEdgePx: normalized.bypassPreviewResolutionCap
      ? normalized.longEdgePx
      : SNAPSHOT_PREVIEW_LONG_EDGE,
    detailLevel: normalized.detailLevel === 'ultra' ? 'high' : normalized.detailLevel,
  };
}
