import * as THREE from 'three';
import {
  AXIS_NAMES,
  ROTATE_BACK_HIT_SCORE_PENALTY,
  ROTATE_BACK_HIT_THRESHOLD_SCALE,
  ROTATE_SCREEN_HOVER_MIN_RADIUS_PX,
  ROTATE_SCREEN_HOVER_PADDING_PX,
  THICK_ROTATE_ARC_RADIUS,
  THICK_TRANSLATE_SHAFT_RADIUS,
  TRANSLATE_ARROW_BASE_RADIUS,
  TRANSLATE_GAP_BRIDGE_RADIUS,
  TRANSLATE_SCREEN_HOVER_PADDING_PX,
  TRANSLATE_SCREEN_HOVER_SAMPLE_COUNT,
  getGizmoRoot,
  type VisibleControlHit,
} from './gizmoCore';
import { preferVisibleControlHit } from './ownership';

const getAxisFromObjectBranch = (
  object: THREE.Object3D | null | undefined,
): 'X' | 'Y' | 'Z' | null => {
  let current: THREE.Object3D | null | undefined = object;

  while (current) {
    if (
      AXIS_NAMES.has(
        (current as THREE.Object3D & { userData?: Record<string, unknown> }).userData?.urdfAxis,
      )
    ) {
      return current.userData.urdfAxis as 'X' | 'Y' | 'Z';
    }

    if (AXIS_NAMES.has(current.name)) {
      return current.name as 'X' | 'Y' | 'Z';
    }

    current = current.parent;
  }

  return null;
};

const VISIBLE_AXIS_CACHE_POINTER_EPSILON = 0.0025;

type VisibleAxisCacheEntry = {
  axis: 'X' | 'Y' | 'Z' | null;
  x: number;
  y: number;
};

const shouldUseVisibleHitFallback = (controls: any) =>
  controls?.userData?.urdfUseVisibleHitFallback === true;

const getCachedVisibleAxisHit = (
  controls: any,
  pointer: { x: number; y: number } | null | undefined,
): 'X' | 'Y' | 'Z' | null => {
  const cached = controls?.userData?.urdfLastVisibleAxisHit as VisibleAxisCacheEntry | undefined;
  if (!cached || !pointer) return null;

  if (
    Math.abs(cached.x - pointer.x) > VISIBLE_AXIS_CACHE_POINTER_EPSILON ||
    Math.abs(cached.y - pointer.y) > VISIBLE_AXIS_CACHE_POINTER_EPSILON
  ) {
    return null;
  }

  return cached.axis;
};

const cacheVisibleAxisHit = (
  controls: any,
  pointer: { x: number; y: number } | null | undefined,
  axis: 'X' | 'Y' | 'Z' | null,
) => {
  if (!pointer) return;

  controls.userData = {
    ...controls.userData,
    urdfLastVisibleAxisHit: {
      axis,
      x: pointer.x,
      y: pointer.y,
    } satisfies VisibleAxisCacheEntry,
  };
};

const getPointerScreenPosition = (pointer: { x: number; y: number }, rect: DOMRect) =>
  new THREE.Vector2((pointer.x + 1) * 0.5 * rect.width, (1 - pointer.y) * 0.5 * rect.height);

const projectWorldPointToScreen = (
  worldPoint: THREE.Vector3,
  camera: THREE.Camera,
  rect: DOMRect,
  target: THREE.Vector2,
) => {
  const projected = worldPoint.clone().project(camera);
  if (
    !Number.isFinite(projected.x) ||
    !Number.isFinite(projected.y) ||
    projected.z < -1 ||
    projected.z > 1
  ) {
    return null;
  }

  target.set((projected.x + 1) * 0.5 * rect.width, (1 - projected.y) * 0.5 * rect.height);
  return target;
};

const getTranslateHandleLocalSamples = (handle: THREE.Mesh, axis: 'X' | 'Y' | 'Z') => {
  const geometry = handle.geometry as THREE.BufferGeometry;
  const geometryVersion = `${geometry.uuid}:${axis}:${TRANSLATE_SCREEN_HOVER_SAMPLE_COUNT}`;

  if (
    handle.userData.urdfTranslateScreenSampleVersion === geometryVersion &&
    Array.isArray(handle.userData.urdfTranslateScreenSamples)
  ) {
    return handle.userData.urdfTranslateScreenSamples as THREE.Vector3[];
  }

  if (!geometry.boundingBox) {
    geometry.computeBoundingBox();
  }

  const boundingBox = geometry.boundingBox;
  if (!boundingBox) {
    handle.userData.urdfTranslateScreenSamples = [new THREE.Vector3()];
    handle.userData.urdfTranslateScreenSampleVersion = geometryVersion;
    return handle.userData.urdfTranslateScreenSamples as THREE.Vector3[];
  }

  const center = boundingBox.getCenter(new THREE.Vector3());
  const axisKey = axis.toLowerCase() as 'x' | 'y' | 'z';
  const start = boundingBox.min[axisKey];
  const end = boundingBox.max[axisKey];
  const sampleCount = Number(TRANSLATE_SCREEN_HOVER_SAMPLE_COUNT);
  const samples: THREE.Vector3[] = [];

  for (let index = 0; index < sampleCount; index += 1) {
    const alpha = sampleCount === 1 ? 0.5 : index / (sampleCount - 1);
    const point = center.clone();
    point[axisKey] = THREE.MathUtils.lerp(start, end, alpha);
    samples.push(point);
  }

  handle.userData.urdfTranslateScreenSamples = samples;
  handle.userData.urdfTranslateScreenSampleVersion = geometryVersion;
  return samples;
};

const TRANSLATE_AXIS_PERP_COMPONENTS: Record<
  'X' | 'Y' | 'Z',
  readonly ['x' | 'y' | 'z', 'x' | 'y' | 'z']
> = {
  X: ['y', 'z'],
  Y: ['x', 'z'],
  Z: ['x', 'y'],
};

const getTranslateHandleLocalRadius = (handle: THREE.Object3D, alpha: number) => {
  if (handle.userData?.urdfTranslateTip) {
    const direction = handle.userData?.urdfTranslateDirection === -1 ? -1 : 1;
    const normalizedAlpha = direction === -1 ? alpha : 1 - alpha;
    return TRANSLATE_ARROW_BASE_RADIUS * Math.max(0, normalizedAlpha);
  }

  if (handle.userData?.urdfTranslateGapBridge) {
    return TRANSLATE_GAP_BRIDGE_RADIUS;
  }

  return THICK_TRANSLATE_SHAFT_RADIUS;
};

const distancePointToSegment2D = (
  point: THREE.Vector2,
  start: THREE.Vector2,
  end: THREE.Vector2,
) => {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSq = dx * dx + dy * dy;

  if (lengthSq <= 1e-8) {
    return point.distanceTo(start);
  }

  const t = THREE.MathUtils.clamp(
    ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSq,
    0,
    1,
  );

  const projectedX = start.x + dx * t;
  const projectedY = start.y + dy * t;
  return Math.hypot(point.x - projectedX, point.y - projectedY);
};

export const resolveVisibleTranslateHit = (
  controls: any,
  pointer: { x: number; y: number },
  axisFilter?: 'X' | 'Y' | 'Z',
): VisibleControlHit | null => {
  if (!controls || controls.mode !== 'translate' || !controls.camera || !controls.domElement) {
    return null;
  }

  const gizmo = getGizmoRoot(controls);
  const translateGroup = gizmo?.gizmo?.translate as THREE.Object3D | undefined;
  if (!translateGroup) return null;

  const rect = controls.domElement.getBoundingClientRect?.();
  if (!rect || rect.width <= 0 || rect.height <= 0) return null;

  const pointerPx = getPointerScreenPosition(pointer, rect);
  const projectedPoint = new THREE.Vector2();
  const projectedOffsetA = new THREE.Vector2();
  const projectedOffsetB = new THREE.Vector2();
  const worldSamplePoint = new THREE.Vector3();
  const worldOffsetPoint = new THREE.Vector3();
  const cameraWorldPosition = new THREE.Vector3();
  controls.camera.getWorldPosition(cameraWorldPosition);
  let bestHit: VisibleControlHit | null = null;

  translateGroup.traverse((node) => {
    const handle = node as THREE.Mesh;
    const axis = getAxisFromObjectBranch(handle);
    if (!axis || !handle.visible || !handle.isMesh) return;
    if (axisFilter && axis !== axisFilter) return;
    if (
      !handle.userData?.urdfTranslateShaft &&
      !handle.userData?.urdfTranslateTip &&
      !handle.userData?.urdfTranslateGapBridge
    ) {
      return;
    }

    const perpendicularComponents = TRANSLATE_AXIS_PERP_COMPONENTS[axis];
    if (!perpendicularComponents) return;

    const localSamples = getTranslateHandleLocalSamples(handle, axis);
    let handleBestScore = Number.POSITIVE_INFINITY;
    let handleBestDistance = Number.POSITIVE_INFINITY;

    for (let index = 0; index < localSamples.length; index += 1) {
      const localSample = localSamples[index];
      const alpha = localSamples.length === 1 ? 0.5 : index / (localSamples.length - 1);
      const localRadius = getTranslateHandleLocalRadius(handle, alpha);
      if (localRadius <= 0) continue;

      worldSamplePoint.copy(localSample).applyMatrix4(handle.matrixWorld);
      const projected = projectWorldPointToScreen(
        worldSamplePoint,
        controls.camera,
        rect,
        projectedPoint,
      );
      if (!projected) continue;

      const distancePx = projected.distanceTo(pointerPx);
      worldOffsetPoint.copy(localSample);
      worldOffsetPoint[perpendicularComponents[0]] += localRadius;
      worldOffsetPoint.applyMatrix4(handle.matrixWorld);
      const projectedRadiusA = projectWorldPointToScreen(
        worldOffsetPoint,
        controls.camera,
        rect,
        projectedOffsetA,
      );

      worldOffsetPoint.copy(localSample);
      worldOffsetPoint[perpendicularComponents[1]] += localRadius;
      worldOffsetPoint.applyMatrix4(handle.matrixWorld);
      const projectedRadiusB = projectWorldPointToScreen(
        worldOffsetPoint,
        controls.camera,
        rect,
        projectedOffsetB,
      );

      const radiusPx = Math.max(
        projectedRadiusA ? projectedRadiusA.distanceTo(projected) : 0,
        projectedRadiusB ? projectedRadiusB.distanceTo(projected) : 0,
      );
      if (radiusPx <= 0) continue;

      const thresholdPx = radiusPx + TRANSLATE_SCREEN_HOVER_PADDING_PX;
      if (distancePx > thresholdPx) continue;

      const normalizedScore = distancePx / thresholdPx;
      const sampleDistance = cameraWorldPosition.distanceTo(worldSamplePoint);

      if (
        normalizedScore < handleBestScore ||
        (Math.abs(normalizedScore - handleBestScore) <= 1e-6 && sampleDistance < handleBestDistance)
      ) {
        handleBestScore = normalizedScore;
        handleBestDistance = sampleDistance;
      }
    }

    if (!Number.isFinite(handleBestScore) || !Number.isFinite(handleBestDistance)) {
      return;
    }

    const candidateHit: VisibleControlHit = {
      owner: 'translate',
      axis,
      renderOrder: handle.renderOrder ?? 0,
      distance: handleBestDistance,
      score: handleBestScore,
    };
    bestHit = preferVisibleControlHit(bestHit, candidateHit);
  });

  return bestHit;
};

const resolveTranslateAxisByScreenDistance = (
  controls: any,
  pointer: { x: number; y: number },
  axisFilter?: 'X' | 'Y' | 'Z',
): 'X' | 'Y' | 'Z' | null =>
  resolveVisibleTranslateHit(controls, pointer, axisFilter)?.axis ?? null;

export const resolveVisibleRotateHit = (
  controls: any,
  pointer: { x: number; y: number },
  axisFilter?: 'X' | 'Y' | 'Z',
): VisibleControlHit | null => {
  if (!controls || controls.mode !== 'rotate' || !controls.camera || !controls.domElement)
    return null;

  const gizmo = getGizmoRoot(controls);
  const rotateGroup = gizmo?.gizmo?.rotate as THREE.Object3D | undefined;
  if (!rotateGroup) return null;

  const rect = controls.domElement.getBoundingClientRect?.();
  if (!rect || rect.width <= 0 || rect.height <= 0) return null;

  const pointerPx = getPointerScreenPosition(pointer, rect);
  const projectedStart = new THREE.Vector2();
  const projectedEnd = new THREE.Vector2();
  const projectedMid = new THREE.Vector2();
  const projectedOffsetA = new THREE.Vector2();
  const projectedOffsetB = new THREE.Vector2();
  const worldStart = new THREE.Vector3();
  const worldEnd = new THREE.Vector3();
  const worldMid = new THREE.Vector3();
  const worldOffset = new THREE.Vector3();
  const cameraWorldPosition = new THREE.Vector3();
  const cameraRight = new THREE.Vector3();
  const cameraUp = new THREE.Vector3();

  controls.camera.getWorldPosition(cameraWorldPosition);
  cameraRight.setFromMatrixColumn(controls.camera.matrixWorld, 0).normalize();
  cameraUp.setFromMatrixColumn(controls.camera.matrixWorld, 1).normalize();

  let bestHit: VisibleControlHit | null = null;
  rotateGroup.traverse((node) => {
    const handle = node as THREE.Mesh;
    const axis = getAxisFromObjectBranch(handle);
    if (!axis || !handle.visible || !handle.isMesh) return;
    if (axisFilter && axis !== axisFilter) return;
    if (!handle.userData?.urdfRotateArcMesh) return;
    const isBackArc = handle.userData?.urdfRotateArcLayer === 'back';

    const centerline = Array.isArray(handle.userData?.urdfRotateCenterlinePoints)
      ? (handle.userData.urdfRotateCenterlinePoints as THREE.Vector3[])
      : null;
    if (!centerline || centerline.length < 2) return;

    const parameters = (
      handle.geometry as THREE.BufferGeometry & {
        parameters?: { tube?: number };
      }
    ).parameters;
    const localTubeRadius =
      typeof parameters?.tube === 'number' && parameters.tube > 0
        ? parameters.tube
        : THICK_ROTATE_ARC_RADIUS;

    let handleBestScore = Number.POSITIVE_INFINITY;
    let handleBestDistance = Number.POSITIVE_INFINITY;

    for (let index = 0; index < centerline.length - 1; index += 1) {
      worldStart.copy(centerline[index]).applyMatrix4(handle.matrixWorld);
      worldEnd.copy(centerline[index + 1]).applyMatrix4(handle.matrixWorld);

      const startProjected = projectWorldPointToScreen(
        worldStart,
        controls.camera,
        rect,
        projectedStart,
      );
      const endProjected = projectWorldPointToScreen(worldEnd, controls.camera, rect, projectedEnd);
      if (!startProjected || !endProjected) continue;

      const distancePx = distancePointToSegment2D(pointerPx, startProjected, endProjected);

      worldMid.lerpVectors(worldStart, worldEnd, 0.5);
      const midProjected = projectWorldPointToScreen(worldMid, controls.camera, rect, projectedMid);
      if (!midProjected) continue;

      worldOffset.copy(worldMid).addScaledVector(cameraRight, localTubeRadius);
      const offsetProjectedA = projectWorldPointToScreen(
        worldOffset,
        controls.camera,
        rect,
        projectedOffsetA,
      );

      worldOffset.copy(worldMid).addScaledVector(cameraUp, localTubeRadius);
      const offsetProjectedB = projectWorldPointToScreen(
        worldOffset,
        controls.camera,
        rect,
        projectedOffsetB,
      );

      const radiusPx = Math.max(
        offsetProjectedA ? offsetProjectedA.distanceTo(midProjected) : 0,
        offsetProjectedB ? offsetProjectedB.distanceTo(midProjected) : 0,
        ROTATE_SCREEN_HOVER_MIN_RADIUS_PX,
      );
      const thresholdPx =
        (radiusPx + ROTATE_SCREEN_HOVER_PADDING_PX) *
        (isBackArc ? ROTATE_BACK_HIT_THRESHOLD_SCALE : 1);
      if (distancePx > thresholdPx) continue;

      const normalizedScore =
        distancePx / thresholdPx + (isBackArc ? ROTATE_BACK_HIT_SCORE_PENALTY : 0);
      const sampleDistance = cameraWorldPosition.distanceTo(worldMid);

      if (
        normalizedScore < handleBestScore ||
        (Math.abs(normalizedScore - handleBestScore) <= 1e-6 && sampleDistance < handleBestDistance)
      ) {
        handleBestScore = normalizedScore;
        handleBestDistance = sampleDistance;
      }
    }

    if (!Number.isFinite(handleBestScore) || !Number.isFinite(handleBestDistance)) {
      return;
    }

    const candidateHit: VisibleControlHit = {
      owner: 'rotate',
      axis,
      renderOrder: handle.renderOrder ?? 0,
      distance: handleBestDistance,
      score: handleBestScore,
    };
    bestHit = preferVisibleControlHit(bestHit, candidateHit);
  });

  return bestHit;
};

const resolveVisibleAxisHit = (
  controls: any,
  pointer: { x: number; y: number },
): 'X' | 'Y' | 'Z' | null => {
  if (!controls || !pointer) return null;
  if (controls.mode === 'translate') {
    return resolveTranslateAxisByScreenDistance(controls, pointer);
  }
  if (controls.mode === 'rotate') {
    return resolveVisibleRotateHit(controls, pointer)?.axis ?? null;
  }
  return null;
};

export const patchVisibleHoverHitFallback = (controls: any) => {
  if (!controls || controls.userData?.urdfVisibleHoverHitFallbackPatched) return;

  const originalPointerHover =
    typeof controls.pointerHover === 'function' ? controls.pointerHover.bind(controls) : null;

  if (!originalPointerHover) return;

  controls.pointerHover = (pointer: { x: number; y: number; button?: number } | null) => {
    originalPointerHover(pointer);

    if (!shouldUseVisibleHitFallback(controls)) {
      return;
    }

    if (controls.object === undefined || controls.dragging === true) return;
    if (!pointer) return;
    if (controls.mode !== 'translate' && controls.mode !== 'rotate') return;

    // Hover should only react to the currently visible gizmo geometry,
    // not the oversized invisible picker volume used for dragging.
    const visibleAxis = resolveVisibleAxisHit(controls, pointer);
    if (visibleAxis) {
      controls.axis = visibleAxis;
    }
    // When the visible hit test finds nothing, keep the axis set by the
    // stock raycaster above so that pointerDown can still initiate a drag.
    cacheVisibleAxisHit(controls, pointer, visibleAxis);
  };

  controls.userData = {
    ...controls.userData,
    urdfVisibleHoverHitFallbackPatched: true,
  };
};

export const patchVisiblePointerDownFallback = (controls: any) => {
  if (!controls || controls.userData?.urdfVisiblePointerDownFallbackPatched) return;

  const originalPointerDown =
    typeof controls.pointerDown === 'function' ? controls.pointerDown.bind(controls) : null;

  if (!originalPointerDown) return;

  controls.pointerDown = (pointer: { x: number; y: number; button?: number } | null) => {
    if (!shouldUseVisibleHitFallback(controls)) {
      originalPointerDown(pointer);
      return;
    }

    if (
      controls.object !== undefined &&
      controls.dragging !== true &&
      (pointer == null || pointer.button === undefined || pointer.button === 0) &&
      (controls.mode === 'translate' || controls.mode === 'rotate')
    ) {
      const visibleAxis =
        getCachedVisibleAxisHit(controls, pointer) ??
        (pointer ? resolveVisibleAxisHit(controls, pointer) : null);

      if (visibleAxis) {
        controls.axis = visibleAxis;
        cacheVisibleAxisHit(controls, pointer, visibleAxis);
      } else if (controls.axis === null) {
        // No visible handle is under the pointer and the stock raycaster did
        // not already select an axis, so do not let the hidden picker start a
        // drag from stale cached state.
        return;
      }
    }

    originalPointerDown(pointer);
  };

  controls.userData = {
    ...controls.userData,
    urdfVisiblePointerDownFallbackPatched: true,
  };
};
