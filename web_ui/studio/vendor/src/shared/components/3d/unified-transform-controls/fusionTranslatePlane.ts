import * as THREE from 'three';

export type FusionTranslateAxisName = 'X' | 'Y' | 'Z';
export type FusionTranslatePlaneName = 'XY' | 'YZ' | 'XZ';

export const FUSION_TRANSLATE_PLANE_OFFSET = 0.245;
export const FUSION_TRANSLATE_PLANE_SIZE = 0.155;
export const FUSION_TRANSLATE_PLANE_PICKER_SCALE = 1.35;
export const FUSION_TRANSLATE_CENTER_RADIUS = 0.075;
export const FUSION_TRANSLATE_CENTER_PICKER_RADIUS = 0.118;

const AXIS_UNIT: Record<FusionTranslateAxisName, THREE.Vector3> = {
  X: new THREE.Vector3(1, 0, 0),
  Y: new THREE.Vector3(0, 1, 0),
  Z: new THREE.Vector3(0, 0, 1),
};

export const getFusionTranslateAxisUnit = (axis: FusionTranslateAxisName) =>
  AXIS_UNIT[axis].clone();

export const getFusionTranslatePlaneAxes = (
  plane: FusionTranslatePlaneName,
): [FusionTranslateAxisName, FusionTranslateAxisName] => {
  if (plane === 'XY') return ['X', 'Y'];
  if (plane === 'YZ') return ['Y', 'Z'];
  return ['X', 'Z'];
};

export const getFusionTranslatePlaneNormalAxis = (
  plane: FusionTranslatePlaneName,
): FusionTranslateAxisName => {
  if (plane === 'XY') return 'Z';
  if (plane === 'YZ') return 'X';
  return 'Y';
};

export const getFusionTranslatePlaneCenter = (
  plane: FusionTranslatePlaneName,
  offset = FUSION_TRANSLATE_PLANE_OFFSET,
) => {
  const [firstAxis, secondAxis] = getFusionTranslatePlaneAxes(plane);
  return getFusionTranslateAxisUnit(firstAxis)
    .multiplyScalar(offset)
    .add(getFusionTranslateAxisUnit(secondAxis).multiplyScalar(offset));
};

export const getFusionTranslatePlaneCorners = ({
  offset = FUSION_TRANSLATE_PLANE_OFFSET,
  plane,
  size = FUSION_TRANSLATE_PLANE_SIZE,
}: {
  offset?: number;
  plane: FusionTranslatePlaneName;
  size?: number;
}) => {
  const [firstAxis, secondAxis] = getFusionTranslatePlaneAxes(plane);
  const first = getFusionTranslateAxisUnit(firstAxis);
  const second = getFusionTranslateAxisUnit(secondAxis);
  const center = getFusionTranslatePlaneCenter(plane, offset);
  const halfSize = size * 0.5;

  return [
    center.clone().addScaledVector(first, -halfSize).addScaledVector(second, -halfSize),
    center.clone().addScaledVector(first, halfSize).addScaledVector(second, -halfSize),
    center.clone().addScaledVector(first, halfSize).addScaledVector(second, halfSize),
    center.clone().addScaledVector(first, -halfSize).addScaledVector(second, halfSize),
  ] as const;
};

export const createFusionTranslatePlaneGeometry = ({
  offset,
  plane,
  scale = 1,
  size,
}: {
  offset?: number;
  plane: FusionTranslatePlaneName;
  scale?: number;
  size?: number;
}) => {
  const corners = getFusionTranslatePlaneCorners({
    offset,
    plane,
    size: (size ?? FUSION_TRANSLATE_PLANE_SIZE) * scale,
  });
  const positions = [
    corners[0],
    corners[1],
    corners[2],
    corners[0],
    corners[2],
    corners[3],
  ].flatMap((point) => [point.x, point.y, point.z]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeBoundingSphere();
  return geometry;
};

export const createFusionTranslatePlaneOutlineGeometry = ({
  offset,
  plane,
  size,
}: {
  offset?: number;
  plane: FusionTranslatePlaneName;
  size?: number;
}) => {
  const corners = getFusionTranslatePlaneCorners({ offset, plane, size });
  const positions = [
    corners[0],
    corners[1],
    corners[1],
    corners[2],
    corners[2],
    corners[3],
    corners[3],
    corners[0],
  ].flatMap((point) => [point.x, point.y, point.z]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeBoundingSphere();
  return geometry;
};

export const getFusionTranslatePlaneDragPlane = ({
  origin,
  plane,
  spaceQuaternion,
}: {
  origin: THREE.Vector3;
  plane: FusionTranslatePlaneName;
  spaceQuaternion: THREE.Quaternion;
}) => {
  const normal = getFusionTranslateAxisUnit(getFusionTranslatePlaneNormalAxis(plane))
    .applyQuaternion(spaceQuaternion)
    .normalize();
  return new THREE.Plane().setFromNormalAndCoplanarPoint(normal, origin);
};

export const getFusionTranslateCenterDragPlane = ({
  cameraDirection,
  origin,
}: {
  cameraDirection: THREE.Vector3;
  origin: THREE.Vector3;
}) => {
  const normal = cameraDirection.clone();
  if (normal.lengthSq() < 1e-10) normal.set(0, 0, 1);
  return new THREE.Plane().setFromNormalAndCoplanarPoint(normal.normalize(), origin);
};

export const resolveFusionTranslatePlanarDelta = ({
  axesWorld,
  intersection,
  snap,
  startIntersection,
}: {
  axesWorld: readonly [THREE.Vector3, THREE.Vector3];
  intersection: THREE.Vector3;
  snap?: number | null;
  startIntersection: THREE.Vector3;
}) => {
  const delta = intersection.clone().sub(startIntersection);
  const result = new THREE.Vector3();

  for (const axis of axesWorld) {
    let distance = delta.dot(axis);
    if (snap && snap > 0) {
      distance = Math.round(distance / snap) * snap;
    }
    result.addScaledVector(axis, distance);
  }

  return result;
};
