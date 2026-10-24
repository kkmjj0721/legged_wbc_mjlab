type ViewerJointLike = {
  angle?: number;
  jointValue?: number;
  name?: string;
};

type ViewerJointMap = Record<string, ViewerJointLike>;

function toFiniteNumber(value: unknown): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export function resolveViewerJointKey(
  joints: ViewerJointMap | null | undefined,
  jointNameOrId: string | null | undefined,
): string | null {
  if (!joints || !jointNameOrId) {
    return null;
  }

  if (jointNameOrId in joints) {
    return jointNameOrId;
  }

  const resolvedEntry = Object.entries(joints).find(([, joint]) => joint?.name === jointNameOrId);
  return resolvedEntry?.[0] ?? null;
}

export function normalizeViewerJointAngleState(
  joints: ViewerJointMap | null | undefined,
  jointAngles: Record<string, number> | null | undefined,
): Record<string, number> {
  if (!joints || !jointAngles) {
    return {};
  }

  return Object.entries(jointAngles).reduce<Record<string, number>>((normalized, [jointNameOrId, angle]) => {
    const numericAngle = toFiniteNumber(angle);
    if (numericAngle === null) {
      return normalized;
    }

    const resolvedKey = resolveViewerJointKey(joints, jointNameOrId);
    if (!resolvedKey) {
      return normalized;
    }

    const shouldOverride = jointNameOrId === resolvedKey || normalized[resolvedKey] === undefined;
    if (shouldOverride) {
      normalized[resolvedKey] = numericAngle;
    }

    return normalized;
  }, {});
}

export function resolveViewerJointAngleValue(
  jointAngles: Record<string, number> | null | undefined,
  jointKey: string,
  joint: ViewerJointLike | null | undefined,
  fallback = 0,
): number {
  const exactAngle = toFiniteNumber(jointAngles?.[jointKey]);
  if (exactAngle !== null) {
    return exactAngle;
  }

  const namedAngle = joint?.name ? toFiniteNumber(jointAngles?.[joint.name]) : null;
  if (namedAngle !== null) {
    return namedAngle;
  }

  const jointAngle = toFiniteNumber(joint?.angle ?? joint?.jointValue);
  if (jointAngle !== null) {
    return jointAngle;
  }

  return fallback;
}
