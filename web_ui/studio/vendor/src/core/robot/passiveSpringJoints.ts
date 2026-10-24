import type { UrdfJoint } from '@/types';

export const PASSIVE_SPRING_EFFORT_EPSILON = 1e-9;
export const HARD_PASSIVE_SPRING_STIFFNESS_THRESHOLD = 1;

export function isUnactuatedJoint(joint: Pick<UrdfJoint, 'limit'>): boolean {
  const effort = joint.limit?.effort;
  return (
    typeof effort !== 'number' ||
    !Number.isFinite(effort) ||
    Math.abs(effort) <= PASSIVE_SPRING_EFFORT_EPSILON
  );
}

export function isPassiveSpringJoint(joint: Pick<UrdfJoint, 'dynamics' | 'limit'>): boolean {
  const stiffness = joint.dynamics?.stiffness;
  return (
    typeof stiffness === 'number' &&
    Number.isFinite(stiffness) &&
    stiffness > 0 &&
    isUnactuatedJoint(joint)
  );
}

export function isHardPassiveSpringJoint(joint: Pick<UrdfJoint, 'dynamics' | 'limit'>): boolean {
  const stiffness = joint.dynamics?.stiffness;
  return (
    typeof stiffness === 'number' &&
    Number.isFinite(stiffness) &&
    stiffness > HARD_PASSIVE_SPRING_STIFFNESS_THRESHOLD &&
    isUnactuatedJoint(joint)
  );
}

export function resolveMjcfPassiveSpringJointMetadata({
  stiffness,
  hasActuator,
}: {
  stiffness: number | undefined;
  hasActuator: boolean;
}): { mjcfPassiveSpringJoint?: boolean; mjcfHardPassiveSpringJoint?: boolean } {
  if (typeof stiffness !== 'number' || !Number.isFinite(stiffness)) {
    return {};
  }

  const passive = stiffness > 0 && !hasActuator;
  return {
    mjcfPassiveSpringJoint: passive,
    mjcfHardPassiveSpringJoint:
      passive && stiffness > HARD_PASSIVE_SPRING_STIFFNESS_THRESHOLD,
  };
}
