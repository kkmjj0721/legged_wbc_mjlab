export const SNAPSHOT_MIN_LONG_EDGE = 3840;
export const SNAPSHOT_MAX_LONG_EDGE_FALLBACK = 8192;
export const SNAPSHOT_TILE_INTERNAL_PIXEL_BUDGET = 10_000_000;

interface ResolveSnapshotRenderPlanOptions {
  baseWidth: number;
  baseHeight: number;
  basePixelRatio: number;
  targetLongEdge?: number | null;
  minLongEdge?: number;
  maxRenderbufferSize?: number | null;
  maxTextureSize?: number | null;
}

export interface SnapshotRenderPlan {
  baseWidth: number;
  baseHeight: number;
  basePixelRatio: number;
  scale: number;
  targetWidth: number;
  targetHeight: number;
  targetPixelRatio: number;
}

export interface SnapshotRenderTile {
  outputX: number;
  outputY: number;
  outputWidth: number;
  outputHeight: number;
  renderX: number;
  renderY: number;
  renderWidth: number;
  renderHeight: number;
}

export interface SnapshotTiledRenderPlan {
  outputWidth: number;
  outputHeight: number;
  fullRenderWidth: number;
  fullRenderHeight: number;
  supersampleScale: number;
  tileOutputLongEdge: number;
  tiles: SnapshotRenderTile[];
}

interface ResolveSnapshotTiledRenderPlanOptions {
  outputWidth: number;
  outputHeight: number;
  supersampleScale: number;
  maxRenderbufferSize?: number | null;
  maxTextureSize?: number | null;
  tileInternalPixelBudget?: number | null;
}

function sanitizePositiveInteger(value: number, fallback: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return fallback;
  }

  return Math.max(1, Math.round(value));
}

function sanitizePositiveNumber(value: number, fallback: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return fallback;
  }

  return value;
}

function sanitizeNonNegativeInteger(value: number, fallback: number) {
  if (!Number.isFinite(value) || value < 0) {
    return fallback;
  }

  return Math.max(0, Math.round(value));
}

function resolveMaxTargetLongEdge(
  baseLongEdge: number,
  maxRenderbufferSize?: number | null,
  maxTextureSize?: number | null,
) {
  const gpuCaps = [maxRenderbufferSize, maxTextureSize]
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value) && value > 0)
    .map((value) => Math.max(baseLongEdge, Math.floor(value)));
  return gpuCaps.length > 0
    ? Math.min(...gpuCaps)
    : Math.max(baseLongEdge, SNAPSHOT_MAX_LONG_EDGE_FALLBACK);
}

export function resolveSnapshotRenderPlan({
  baseWidth,
  baseHeight,
  basePixelRatio,
  targetLongEdge = null,
  minLongEdge = SNAPSHOT_MIN_LONG_EDGE,
  maxRenderbufferSize,
  maxTextureSize,
}: ResolveSnapshotRenderPlanOptions): SnapshotRenderPlan {
  const safeBaseWidth = sanitizePositiveInteger(baseWidth, 1);
  const safeBaseHeight = sanitizePositiveInteger(baseHeight, 1);
  const safeBasePixelRatio = sanitizePositiveNumber(basePixelRatio, 1);
  const baseLongEdge = Math.max(safeBaseWidth, safeBaseHeight);
  const desiredLongEdge =
    targetLongEdge == null
      ? Math.max(baseLongEdge, sanitizePositiveInteger(minLongEdge, SNAPSHOT_MIN_LONG_EDGE))
      : sanitizePositiveInteger(targetLongEdge, baseLongEdge);

  const maxLongEdge = resolveMaxTargetLongEdge(baseLongEdge, maxRenderbufferSize, maxTextureSize);
  const resolvedTargetLongEdge = Math.min(desiredLongEdge, maxLongEdge);
  const scale = resolvedTargetLongEdge / baseLongEdge;

  return {
    baseWidth: safeBaseWidth,
    baseHeight: safeBaseHeight,
    basePixelRatio: safeBasePixelRatio,
    scale,
    targetWidth: Math.max(1, Math.round(safeBaseWidth * scale)),
    targetHeight: Math.max(1, Math.round(safeBaseHeight * scale)),
    targetPixelRatio: safeBasePixelRatio * scale,
  };
}

export function clampSnapshotRenderPlanToPixelBudget(
  plan: SnapshotRenderPlan,
  maxPixelCount: number | null | undefined,
): SnapshotRenderPlan {
  const safePixelBudget = maxPixelCount == null ? null : sanitizePositiveInteger(maxPixelCount, 1);
  if (safePixelBudget == null) {
    return plan;
  }

  const currentPixelCount = plan.targetWidth * plan.targetHeight;
  if (!Number.isFinite(currentPixelCount) || currentPixelCount <= safePixelBudget) {
    return plan;
  }

  const budgetScale = Math.sqrt(safePixelBudget / currentPixelCount);
  const targetWidth = Math.max(1, Math.floor(plan.targetWidth * budgetScale));
  const targetHeight = Math.max(1, Math.floor(plan.targetHeight * budgetScale));
  const effectiveScale = Math.min(targetWidth / plan.baseWidth, targetHeight / plan.baseHeight);

  return {
    ...plan,
    scale: effectiveScale,
    targetWidth,
    targetHeight,
    targetPixelRatio: plan.basePixelRatio * effectiveScale,
  };
}

export function resolveSnapshotRenderTargetSamples({
  width,
  height,
  requestedSamples,
  maxSupportedSamples,
}: {
  width: number;
  height: number;
  requestedSamples: number;
  maxSupportedSamples: number;
}): number {
  const supportedSamples = Math.min(
    sanitizeNonNegativeInteger(requestedSamples, 0),
    sanitizeNonNegativeInteger(maxSupportedSamples, 0),
  );
  if (supportedSamples <= 0) {
    return 0;
  }

  const pixelCount = sanitizePositiveInteger(width, 1) * sanitizePositiveInteger(height, 1);
  if (pixelCount >= 24_000_000) {
    return 0;
  }

  if (pixelCount >= 12_000_000) {
    return Math.min(2, supportedSamples);
  }

  if (pixelCount >= 6_000_000) {
    return Math.min(4, supportedSamples);
  }

  return supportedSamples;
}

export function resolveSnapshotTiledRenderPlan({
  outputWidth,
  outputHeight,
  supersampleScale,
  maxRenderbufferSize,
  maxTextureSize,
  tileInternalPixelBudget = SNAPSHOT_TILE_INTERNAL_PIXEL_BUDGET,
}: ResolveSnapshotTiledRenderPlanOptions): SnapshotTiledRenderPlan {
  const safeOutputWidth = sanitizePositiveInteger(outputWidth, 1);
  const safeOutputHeight = sanitizePositiveInteger(outputHeight, 1);
  const safeSupersampleScale = sanitizePositiveNumber(supersampleScale, 1);
  const fullRenderWidth = Math.max(1, Math.round(safeOutputWidth * safeSupersampleScale));
  const fullRenderHeight = Math.max(1, Math.round(safeOutputHeight * safeSupersampleScale));
  const maxTargetLongEdge = resolveMaxTargetLongEdge(
    1,
    maxRenderbufferSize,
    maxTextureSize,
  );
  const safeTilePixelBudget = sanitizePositiveInteger(
    tileInternalPixelBudget ?? SNAPSHOT_TILE_INTERNAL_PIXEL_BUDGET,
    SNAPSHOT_TILE_INTERNAL_PIXEL_BUDGET,
  );
  const maxTileRenderLongEdge = Math.max(
    1,
    Math.min(maxTargetLongEdge, Math.floor(Math.sqrt(safeTilePixelBudget))),
  );
  const tileOutputLongEdge = Math.max(1, Math.floor(maxTileRenderLongEdge / safeSupersampleScale));
  const tiles: SnapshotRenderTile[] = [];

  for (let outputY = 0; outputY < safeOutputHeight; outputY += tileOutputLongEdge) {
    const nextOutputY = Math.min(safeOutputHeight, outputY + tileOutputLongEdge);
    const renderY = Math.round(outputY * safeSupersampleScale);
    const nextRenderY =
      nextOutputY === safeOutputHeight
        ? fullRenderHeight
        : Math.round(nextOutputY * safeSupersampleScale);

    for (let outputX = 0; outputX < safeOutputWidth; outputX += tileOutputLongEdge) {
      const nextOutputX = Math.min(safeOutputWidth, outputX + tileOutputLongEdge);
      const renderX = Math.round(outputX * safeSupersampleScale);
      const nextRenderX =
        nextOutputX === safeOutputWidth
          ? fullRenderWidth
          : Math.round(nextOutputX * safeSupersampleScale);

      tiles.push({
        outputX,
        outputY,
        outputWidth: nextOutputX - outputX,
        outputHeight: nextOutputY - outputY,
        renderX,
        renderY,
        renderWidth: Math.max(1, nextRenderX - renderX),
        renderHeight: Math.max(1, nextRenderY - renderY),
      });
    }
  }

  return {
    outputWidth: safeOutputWidth,
    outputHeight: safeOutputHeight,
    fullRenderWidth,
    fullRenderHeight,
    supersampleScale: safeSupersampleScale,
    tileOutputLongEdge,
    tiles,
  };
}
