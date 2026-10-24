const TAU = Math.PI * 2;

export function wrapContinuousJointAngle(angle: number): number {
  if (!Number.isFinite(angle)) {
    return angle;
  }

  let wrapped = angle % TAU;

  if (wrapped > Math.PI) {
    wrapped -= TAU;
  }

  if (wrapped <= -Math.PI) {
    wrapped += TAU;
  }

  return wrapped;
}

export function unwrapContinuousJointAngle(angle: number, referenceAngle: number): number {
  if (!Number.isFinite(angle) || !Number.isFinite(referenceAngle)) {
    return angle;
  }

  let unwrapped = angle;

  while (unwrapped - referenceAngle > Math.PI) {
    unwrapped -= TAU;
  }

  while (unwrapped - referenceAngle < -Math.PI) {
    unwrapped += TAU;
  }

  return unwrapped;
}
