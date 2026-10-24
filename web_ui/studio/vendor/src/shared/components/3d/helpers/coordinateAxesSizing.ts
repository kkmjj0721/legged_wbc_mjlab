export const ORIGIN_AXES_SIZE_MIN = 0.01;
export const ORIGIN_AXES_SIZE_STEP = 0.01;
export const DEFAULT_ORIGIN_AXES_SIZE = 0.07;
export const ORIGIN_AXES_SIZE_FALLBACK_MAX = 0.5;
export const ORIGIN_AXES_SIZE_ABSOLUTE_MAX = 2;
export const ORIGIN_AXES_SIZE_MODEL_EXTENT_FACTOR = 0.5;
export const ORIGIN_AXES_SIZE_MODEL_MIN_MAX = 0.5;

export function normalizeOriginAxesSize(
  value: number | string | null | undefined,
  fallback: number = DEFAULT_ORIGIN_AXES_SIZE,
  max: number = ORIGIN_AXES_SIZE_ABSOLUTE_MAX,
): number {
  const numericValue = typeof value === 'number' ? value : Number(value);
  const finiteMax = Number.isFinite(max) ? Math.max(max, ORIGIN_AXES_SIZE_MIN) : fallback;
  const safeFallback = Math.min(Math.max(fallback, ORIGIN_AXES_SIZE_MIN), finiteMax);

  if (!Number.isFinite(numericValue)) {
    return safeFallback;
  }

  return Math.min(Math.max(numericValue, ORIGIN_AXES_SIZE_MIN), finiteMax);
}

export function resolveOriginAxesSizeMax(modelExtent: number | null | undefined): number {
  if (!Number.isFinite(modelExtent) || Number(modelExtent) <= 0) {
    return ORIGIN_AXES_SIZE_FALLBACK_MAX;
  }

  const modelScaledMax = Number(modelExtent) * ORIGIN_AXES_SIZE_MODEL_EXTENT_FACTOR;
  return Math.min(
    Math.max(modelScaledMax, ORIGIN_AXES_SIZE_MODEL_MIN_MAX),
    ORIGIN_AXES_SIZE_ABSOLUTE_MAX,
  );
}
