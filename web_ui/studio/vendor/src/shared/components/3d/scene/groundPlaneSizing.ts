import * as THREE from 'three';

export interface GroundPlaneLayout {
  centerX: number;
  centerY: number;
  size: number;
  fadeDistance: number;
  fadeFrom: number;
}

export const DEFAULT_GROUND_PLANE_SIZE = 20;
const MAX_GROUND_PLANE_SIZE = 240;
const GROUND_PLANE_SIZE_STEP = 2;
const GROUND_PLANE_MARGIN = 2;
const GROUND_PLANE_ORIGIN_LOCK_MARGIN = 2;
const GROUND_PLANE_SCALE_FACTOR = 1.45;
const GROUND_PLANE_FADE_DISTANCE_FACTOR = 100;
const GROUND_PLANE_FADE_FROM = 1;

function roundUpToStep(value: number, step: number): number {
  return Math.ceil(value / step) * step;
}

function roundToNearestStep(value: number, step: number): number {
  return Math.round(value / step) * step;
}

function resolveGroundPlaneAxisCenter(min: number, max: number): number {
  if (
    min <= GROUND_PLANE_ORIGIN_LOCK_MARGIN
    && max >= -GROUND_PLANE_ORIGIN_LOCK_MARGIN
  ) {
    return 0;
  }

  return roundToNearestStep((min + max) * 0.5, GROUND_PLANE_SIZE_STEP);
}

export function resolveGroundPlaneLayout(bounds: THREE.Box3 | null): GroundPlaneLayout {
  if (!bounds || bounds.isEmpty()) {
    return {
      centerX: 0,
      centerY: 0,
      size: DEFAULT_GROUND_PLANE_SIZE,
      fadeDistance: DEFAULT_GROUND_PLANE_SIZE * GROUND_PLANE_FADE_DISTANCE_FACTOR,
      fadeFrom: GROUND_PLANE_FADE_FROM,
    };
  }

  const maxFootprintExtentFromOrigin = Math.max(
    Math.abs(bounds.min.x),
    Math.abs(bounds.max.x),
    Math.abs(bounds.min.y),
    Math.abs(bounds.max.y),
  );
  const defaultHalfSize = DEFAULT_GROUND_PLANE_SIZE * 0.5;
  const rawSize = maxFootprintExtentFromOrigin <= defaultHalfSize
    ? DEFAULT_GROUND_PLANE_SIZE
    : (maxFootprintExtentFromOrigin * 2 * GROUND_PLANE_SCALE_FACTOR) + GROUND_PLANE_MARGIN;
  const size = Math.min(
    MAX_GROUND_PLANE_SIZE,
    roundUpToStep(rawSize, GROUND_PLANE_SIZE_STEP),
  );

  return {
    centerX: resolveGroundPlaneAxisCenter(bounds.min.x, bounds.max.x),
    centerY: resolveGroundPlaneAxisCenter(bounds.min.y, bounds.max.y),
    size,
    fadeDistance: size * GROUND_PLANE_FADE_DISTANCE_FACTOR,
    fadeFrom: GROUND_PLANE_FADE_FROM,
  };
}

export function areGroundPlaneLayoutsEqual(
  left: GroundPlaneLayout,
  right: GroundPlaneLayout,
  epsilon = 1e-3,
): boolean {
  return Math.abs(left.centerX - right.centerX) <= epsilon
    && Math.abs(left.centerY - right.centerY) <= epsilon
    && Math.abs(left.size - right.size) <= epsilon
    && Math.abs(left.fadeDistance - right.fadeDistance) <= epsilon
    && Math.abs(left.fadeFrom - right.fadeFrom) <= epsilon;
}
