import * as THREE from 'three';

export type FusionRotateAxisName = 'X' | 'Y' | 'Z';

export const FUSION_ROTATE_ARC_RADIUS = 0.54;
export const FUSION_ROTATE_FULL_CIRCLE = Math.PI * 2;
export const FUSION_ROTATE_FRONT_ARC_SPAN = Math.PI;
export const FUSION_ROTATE_E_RING_RADIUS = 0.66;
export const FUSION_ROTATE_TRACKBALL_RADIUS = 0.095;
export const FUSION_ROTATE_TUBE_RADIAL_SEGMENTS = 20;
export const FUSION_ROTATE_FRONT_ARC_TUBULAR_SEGMENTS = 96;
export const FUSION_ROTATE_FULL_RING_TUBULAR_SEGMENTS = 160;
export const FUSION_ROTATE_TRACKBALL_SENSITIVITY = 1.45;

const AXIS_UNIT: Record<FusionRotateAxisName, THREE.Vector3> = {
  X: new THREE.Vector3(1, 0, 0),
  Y: new THREE.Vector3(0, 1, 0),
  Z: new THREE.Vector3(0, 0, 1),
};

export const getFusionRotateArcRadius = () => FUSION_ROTATE_ARC_RADIUS;

export const getFusionRotateArcPoint = (
  axis: FusionRotateAxisName,
  angle: number,
  radius = FUSION_ROTATE_ARC_RADIUS,
) => {
  const cos = Math.cos(angle) * radius;
  const sin = Math.sin(angle) * radius;

  if (axis === 'X') return new THREE.Vector3(0, cos, sin);
  if (axis === 'Y') return new THREE.Vector3(cos, 0, sin);
  return new THREE.Vector3(cos, sin, 0);
};

export const getFusionRotateFrontArcAngles = (centerAngle = 0) => ({
  start: centerAngle - FUSION_ROTATE_FRONT_ARC_SPAN * 0.5,
  end: centerAngle + FUSION_ROTATE_FRONT_ARC_SPAN * 0.5,
});

export const getFusionRotateFrontArcCenterAngle = (
  axis: FusionRotateAxisName,
  cameraVectorLocal: THREE.Vector3,
) => {
  const projected = cameraVectorLocal.clone();
  if (axis === 'X') {
    projected.x = 0;
    if (projected.lengthSq() < 1e-10) return 0;
    projected.normalize();
    return Math.atan2(projected.z, projected.y);
  }
  if (axis === 'Y') {
    projected.y = 0;
    if (projected.lengthSq() < 1e-10) return 0;
    projected.normalize();
    return Math.atan2(projected.z, projected.x);
  }

  projected.z = 0;
  if (projected.lengthSq() < 1e-10) return 0;
  projected.normalize();
  return Math.atan2(projected.y, projected.x);
};

export const getFusionRotateFrontArcQuaternion = (
  axis: FusionRotateAxisName,
  centerAngle: number,
) => {
  const signedAngle = axis === 'Y' ? -centerAngle : centerAngle;
  return new THREE.Quaternion().setFromAxisAngle(AXIS_UNIT[axis], signedAngle);
};

export const getFusionRotateScreenQuaternion = (cameraDirectionWorld: THREE.Vector3) => {
  const normal = cameraDirectionWorld.clone();
  if (normal.lengthSq() < 1e-10) return new THREE.Quaternion();
  return new THREE.Quaternion().setFromUnitVectors(
    AXIS_UNIT.Z,
    normal.normalize(),
  );
};

export const createFusionRotateTubeGeometry = ({
  axis,
  closed,
  endAngle,
  radius = FUSION_ROTATE_ARC_RADIUS,
  radialSegments = FUSION_ROTATE_TUBE_RADIAL_SEGMENTS,
  startAngle,
  tubeRadius,
  tubularSegments,
}: {
  axis: FusionRotateAxisName;
  closed: boolean;
  endAngle: number;
  radius?: number;
  radialSegments?: number;
  startAngle: number;
  tubeRadius: number;
  tubularSegments: number;
}) => {
  const samples = Math.max(8, closed ? tubularSegments : tubularSegments + 1);
  const points: THREE.Vector3[] = [];

  for (let index = 0; index < samples; index += 1) {
    const alpha = closed ? index / samples : index / (samples - 1);
    points.push(
      getFusionRotateArcPoint(
        axis,
        THREE.MathUtils.lerp(startAngle, endAngle, alpha),
        radius,
      ),
    );
  }

  return new THREE.TubeGeometry(
    new THREE.CatmullRomCurve3(points, closed),
    tubularSegments,
    tubeRadius,
    radialSegments,
    closed,
  );
};

export const createFusionRotateFrontArcGeometry = (
  axis: FusionRotateAxisName,
  tubeRadius: number,
  radius = FUSION_ROTATE_ARC_RADIUS,
) => {
  const arc = getFusionRotateFrontArcAngles(0);
  return createFusionRotateTubeGeometry({
    axis,
    closed: false,
    endAngle: arc.end,
    radius,
    startAngle: arc.start,
    tubeRadius,
    tubularSegments: FUSION_ROTATE_FRONT_ARC_TUBULAR_SEGMENTS,
  });
};

export const createFusionRotateFullRingGeometry = (
  axis: FusionRotateAxisName,
  tubeRadius: number,
  radius = FUSION_ROTATE_ARC_RADIUS,
) =>
  createFusionRotateTubeGeometry({
    axis,
    closed: true,
    endAngle: FUSION_ROTATE_FULL_CIRCLE,
    radius,
    startAngle: 0,
    tubeRadius,
    tubularSegments: FUSION_ROTATE_FULL_RING_TUBULAR_SEGMENTS,
  });

export const resolveFusionTrackballQuaternion = ({
  cameraRightWorld,
  cameraUpWorld,
  deltaWorld,
  parentWorldQuaternionInv,
  radius,
  startQuaternion,
}: {
  cameraRightWorld: THREE.Vector3;
  cameraUpWorld: THREE.Vector3;
  deltaWorld: THREE.Vector3;
  parentWorldQuaternionInv: THREE.Quaternion;
  radius: number;
  startQuaternion: THREE.Quaternion;
}) => {
  const safeRadius = Math.max(radius, 1e-6);
  const yaw = deltaWorld.dot(cameraRightWorld) / safeRadius;
  const pitch = -deltaWorld.dot(cameraUpWorld) / safeRadius;
  const parentUp = cameraUpWorld.clone().applyQuaternion(parentWorldQuaternionInv).normalize();
  const parentRight = cameraRightWorld.clone().applyQuaternion(parentWorldQuaternionInv).normalize();
  const yawQuaternion = new THREE.Quaternion().setFromAxisAngle(
    parentUp,
    yaw * FUSION_ROTATE_TRACKBALL_SENSITIVITY,
  );
  const pitchQuaternion = new THREE.Quaternion().setFromAxisAngle(
    parentRight,
    pitch * FUSION_ROTATE_TRACKBALL_SENSITIVITY,
  );

  return yawQuaternion
    .multiply(pitchQuaternion)
    .multiply(startQuaternion)
    .normalize();
};
