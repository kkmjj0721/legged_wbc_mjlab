import * as THREE from 'three';

export const DEFAULT_WORKSPACE_ORBIT_PAN_TUNING = {
  closeRangeDistanceFactor: 0.18,
  minDistanceFloorFactor: 4,
  maxBoost: 3,
  farDistanceFactor: 6,
  localFarDistanceFloorFactor: 4,
  minFarPanScale: 0.18,
  minZoomScale: 0.35,
} as const;

interface ResolveWorkspaceOrbitPanSpeedOptions {
  basePanSpeed: number;
  camera: THREE.Camera;
  target: THREE.Vector3;
  sceneBounds?: THREE.Box3 | null;
  minDistance?: number;
  maxBoost?: number;
}

interface ResolveWorkspaceOrbitZoomSpeedOptions {
  baseZoomSpeed: number;
  camera: THREE.Camera;
  target: THREE.Vector3;
  sceneBounds?: THREE.Box3 | null;
  minDistance?: number;
}

function resolveWorkspaceOrbitSceneDiagonal(sceneBounds: THREE.Box3 | null | undefined) {
  const sceneDiagonal = sceneBounds?.getSize(new THREE.Vector3()).length() ?? 0;
  return Number.isFinite(sceneDiagonal) && sceneDiagonal > 0 ? sceneDiagonal : 0;
}

function resolveWorkspaceOrbitNavigationSceneScale(
  sceneBounds: THREE.Box3 | null | undefined,
) {
  const sceneDiagonal = resolveWorkspaceOrbitSceneDiagonal(sceneBounds);
  if (sceneDiagonal <= 1) {
    return sceneDiagonal;
  }

  // Robot workspaces can range from sub-meter hands to very long assemblies.
  // A square-root scale keeps navigation tuning responsive without letting a
  // long model linearly amplify close inspection panning.
  return Math.sqrt(sceneDiagonal);
}

function resolveWorkspaceOrbitPanDistanceFloor(
  sceneBounds: THREE.Box3 | null | undefined,
  minDistance: number | undefined,
) {
  const sceneScale = resolveWorkspaceOrbitNavigationSceneScale(sceneBounds);
  const distanceFromScene =
    sceneScale * DEFAULT_WORKSPACE_ORBIT_PAN_TUNING.closeRangeDistanceFactor;
  const distanceFromControls = Number.isFinite(minDistance)
    ? Math.max(0, Number(minDistance)) * DEFAULT_WORKSPACE_ORBIT_PAN_TUNING.minDistanceFloorFactor
    : 0;

  return Math.max(distanceFromScene, distanceFromControls);
}

function resolveWorkspaceOrbitPanFarDistanceCeiling(
  sceneBounds: THREE.Box3 | null | undefined,
  camera: THREE.Camera,
  distanceFloor: number,
) {
  if (!sceneBounds || sceneBounds.isEmpty()) {
    return null;
  }

  const sceneScale = resolveWorkspaceOrbitNavigationSceneScale(sceneBounds);
  if (sceneScale <= 0 || distanceFloor <= 0) {
    return null;
  }

  const cameraDistanceToBounds = sceneBounds.distanceToPoint(camera.position);
  if (!Number.isFinite(cameraDistanceToBounds) || cameraDistanceToBounds < 0) {
    return null;
  }

  const sceneCeiling = sceneScale * DEFAULT_WORKSPACE_ORBIT_PAN_TUNING.farDistanceFactor;
  const localCeiling =
    cameraDistanceToBounds +
    distanceFloor * DEFAULT_WORKSPACE_ORBIT_PAN_TUNING.localFarDistanceFloorFactor;

  return Math.max(distanceFloor, Math.min(sceneCeiling, localCeiling));
}

function resolveWorkspaceOrbitEffectivePanDistance({
  camera,
  target,
  sceneBounds,
  minDistance,
}: Pick<
  ResolveWorkspaceOrbitPanSpeedOptions,
  'camera' | 'target' | 'sceneBounds' | 'minDistance'
>) {
  const distance = camera.position.distanceTo(target);
  if (!Number.isFinite(distance) || distance <= 0) {
    return null;
  }

  const distanceFloor = resolveWorkspaceOrbitPanDistanceFloor(sceneBounds, minDistance);
  const flooredDistance = Math.max(distance, distanceFloor);
  const farDistanceCeiling = resolveWorkspaceOrbitPanFarDistanceCeiling(
    sceneBounds,
    camera,
    distanceFloor,
  );

  if (farDistanceCeiling === null || distance <= farDistanceCeiling) {
    return flooredDistance;
  }

  const dampedDistance =
    farDistanceCeiling +
    (distance - farDistanceCeiling) * DEFAULT_WORKSPACE_ORBIT_PAN_TUNING.minFarPanScale;
  return Math.max(distanceFloor, dampedDistance);
}

export function resolveWorkspaceOrbitPanSpeed({
  basePanSpeed,
  camera,
  target,
  sceneBounds,
  minDistance,
  maxBoost = DEFAULT_WORKSPACE_ORBIT_PAN_TUNING.maxBoost,
}: ResolveWorkspaceOrbitPanSpeedOptions) {
  if (!(camera instanceof THREE.PerspectiveCamera)) {
    return basePanSpeed;
  }

  const distance = camera.position.distanceTo(target);
  if (!Number.isFinite(distance) || distance <= 0) {
    return basePanSpeed;
  }

  const effectiveDistance = resolveWorkspaceOrbitEffectivePanDistance({
    camera,
    target,
    sceneBounds,
    minDistance,
  });
  if (effectiveDistance === null) {
    return basePanSpeed;
  }

  // OrbitControls scales perspective panning with camera-target distance. Use a
  // bounded effective distance so close inspection is not sticky and distant
  // pivots in large scenes do not make panning overshoot.
  const speedScale = THREE.MathUtils.clamp(
    effectiveDistance / distance,
    DEFAULT_WORKSPACE_ORBIT_PAN_TUNING.minFarPanScale,
    maxBoost,
  );
  return basePanSpeed * speedScale;
}

export function resolveWorkspaceOrbitZoomSpeed({
  baseZoomSpeed,
  camera,
  target,
  sceneBounds,
  minDistance,
}: ResolveWorkspaceOrbitZoomSpeedOptions) {
  if (!(camera instanceof THREE.PerspectiveCamera)) {
    return baseZoomSpeed;
  }

  const distance = camera.position.distanceTo(target);
  if (!Number.isFinite(distance) || distance <= 0) {
    return baseZoomSpeed;
  }

  const effectiveDistance = resolveWorkspaceOrbitEffectivePanDistance({
    camera,
    target,
    sceneBounds,
    minDistance,
  });
  if (effectiveDistance === null) {
    return baseZoomSpeed;
  }

  const speedScale = THREE.MathUtils.clamp(
    effectiveDistance / distance,
    DEFAULT_WORKSPACE_ORBIT_PAN_TUNING.minZoomScale,
    1,
  );
  return baseZoomSpeed * speedScale;
}
