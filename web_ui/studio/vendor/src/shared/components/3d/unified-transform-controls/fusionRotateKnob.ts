import * as THREE from 'three';

import {
  THICK_ROTATE_ARC_RADIUS,
  THICK_TRANSLATE_SHAFT_RADIUS,
} from './gizmoCore';

export type FusionRotateAxisName = 'X' | 'Y' | 'Z';

export const FUSION_ROTATE_ARC_RADIUS = 0.54;
export const FUSION_ROTATE_ARC_RADII: Record<FusionRotateAxisName, number> = {
  X: FUSION_ROTATE_ARC_RADIUS,
  Y: FUSION_ROTATE_ARC_RADIUS,
  Z: FUSION_ROTATE_ARC_RADIUS,
};
export const FUSION_ROTATE_ARC_MAX_RADIUS = Math.max(
  ...Object.values(FUSION_ROTATE_ARC_RADII),
);
export const FUSION_ROTATE_FULL_CIRCLE = Math.PI * 2;

const createFusionRotateArcAngles = (
  knobDegrees: number,
) => {
  return {
    start: 0,
    end: FUSION_ROTATE_FULL_CIRCLE,
    knob: THREE.MathUtils.degToRad(knobDegrees),
  };
};

export const getFusionRotateArcRadius = (axis: FusionRotateAxisName) =>
  FUSION_ROTATE_ARC_RADII[axis];

export const FUSION_ROTATE_ARC_ANGLES: Record<
  FusionRotateAxisName,
  { start: number; end: number; knob: number }
> = {
  X: createFusionRotateArcAngles(45),
  Y: createFusionRotateArcAngles(140),
  Z: createFusionRotateArcAngles(300),
};

export const FUSION_ROTATE_ARC_START = FUSION_ROTATE_ARC_ANGLES.X.start;
export const FUSION_ROTATE_ARC_END = FUSION_ROTATE_ARC_ANGLES.X.end;
export const FUSION_ROTATE_KNOB_ANGLE = FUSION_ROTATE_ARC_ANGLES.X.knob;

export const getFusionRotateArcAngles = (axis: FusionRotateAxisName) =>
  FUSION_ROTATE_ARC_ANGLES[axis];

export const getFusionRotateKnobAngle = (axis: FusionRotateAxisName) =>
  getFusionRotateArcAngles(axis).knob;

export const getFusionRotateArcPoint = (
  axis: FusionRotateAxisName,
  angle: number,
  radius = getFusionRotateArcRadius(axis),
) => {
  const cos = Math.cos(angle) * radius;
  const sin = Math.sin(angle) * radius;

  if (axis === 'X') return new THREE.Vector3(0, cos, sin);
  if (axis === 'Y') return new THREE.Vector3(cos, 0, sin);
  return new THREE.Vector3(cos, sin, 0);
};

export const resolveFusionRotateKnobAngle = ({
  activeAxis,
  axis,
  baseAngle,
  rotationAngle = 0,
}: {
  activeAxis: FusionRotateAxisName | null;
  axis: FusionRotateAxisName;
  baseAngle?: number;
  rotationAngle?: number;
}) => {
  const resolvedBaseAngle = baseAngle ?? getFusionRotateKnobAngle(axis);
  return activeAxis === axis ? resolvedBaseAngle - rotationAngle : resolvedBaseAngle;
};

export const FUSION_TRANSLATE_ARC_CLEARANCE = 0.04;

export const resolveFusionTranslateShaftStart = ({
  rotateArcTubeRadius = THICK_ROTATE_ARC_RADIUS,
  translateShaftRadius = THICK_TRANSLATE_SHAFT_RADIUS,
}: {
  rotateArcTubeRadius?: number;
  translateShaftRadius?: number;
} = {}) =>
  FUSION_ROTATE_ARC_MAX_RADIUS +
  rotateArcTubeRadius +
  translateShaftRadius +
  FUSION_TRANSLATE_ARC_CLEARANCE;
