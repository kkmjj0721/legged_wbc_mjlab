import * as THREE from 'three';

import {
  solveClosedLoopMotionCompensation,
  type LinkIkPositionSolveRequest,
  type LinkIkSolveFailureReason,
} from '@/core/robot';
import type { JointQuaternion, RobotState } from '@/types';

// Ignore sub-pixel proxy jitter so the drag loop only solves when the user
// actually moves the handle to a meaningfully new target.
export const LINK_IK_TARGET_EPSILON_SQ = 1e-8;
export const LINK_IK_PREVIEW_MAX_ITERATIONS = 6;
export const LINK_IK_PREVIEW_POSITION_TOLERANCE = 2e-3;
export const LINK_IK_PREVIEW_STALL_TOLERANCE = 1e-4;
export const LINK_IK_PREVIEW_COORDINATE_PAIR_MAX_DISTANCE = 2;
export const LINK_IK_PREVIEW_MAX_ANGLE_STEP = 0.025;
export const LINK_IK_PREVIEW_MAX_QUATERNION_STEP_RADIANS = 0.025;
export const LINK_IK_COMMIT_EPSILON = 1e-6;
export const LINK_IK_PREVIEW_COMMIT_EPSILON = 1e-5;

export interface LinkIkDragKinematicState {
  angles: Record<string, number>;
  quaternions: Record<string, JointQuaternion>;
}

function hasMeaningfulNumberDelta(
  previous: number | undefined,
  next: number | undefined,
  epsilon: number,
): boolean {
  if (typeof previous !== 'number' || typeof next !== 'number') {
    return previous !== next;
  }

  return Math.abs(previous - next) > epsilon;
}

function hasMeaningfulQuaternionDelta(
  previous: JointQuaternion | undefined,
  next: JointQuaternion | undefined,
  epsilon: number,
): boolean {
  if (!previous || !next) {
    return previous !== next;
  }

  return (
    hasMeaningfulNumberDelta(previous.x, next.x, epsilon) ||
    hasMeaningfulNumberDelta(previous.y, next.y, epsilon) ||
    hasMeaningfulNumberDelta(previous.z, next.z, epsilon) ||
    hasMeaningfulNumberDelta(previous.w, next.w, epsilon)
  );
}

export function createEmptyLinkIkDragKinematicState(): LinkIkDragKinematicState {
  return {
    angles: {},
    quaternions: {},
  };
}

export function cloneLinkIkDragKinematicState(
  state: Partial<LinkIkDragKinematicState> | null | undefined,
): LinkIkDragKinematicState {
  return {
    angles: { ...(state?.angles ?? {}) },
    quaternions: { ...(state?.quaternions ?? {}) },
  };
}

export function hasLinkIkKinematicStateChanges(
  state: Partial<LinkIkDragKinematicState> | null | undefined,
): boolean {
  return Boolean(
    state &&
    (Object.keys(state.angles ?? {}).length > 0 || Object.keys(state.quaternions ?? {}).length > 0),
  );
}

export function hasRestorableLinkIkPreviewKinematicState(
  state: Partial<LinkIkDragKinematicState> | null | undefined,
): boolean {
  return hasLinkIkKinematicStateChanges(state);
}

export function resolveLinkIkCommittedStateEpsilon(preview: boolean): number {
  return preview ? LINK_IK_PREVIEW_COMMIT_EPSILON : LINK_IK_COMMIT_EPSILON;
}

export function shouldAcceptLinkIkSolveState({
  seedState,
  nextState,
  preview,
  converged,
  failureReason,
}: {
  seedState: Partial<LinkIkDragKinematicState> | null | undefined;
  nextState: Partial<LinkIkDragKinematicState> | null | undefined;
  preview: boolean;
  converged: boolean;
  failureReason?: LinkIkSolveFailureReason;
}): boolean {
  if (converged || failureReason !== 'stalled') {
    return true;
  }

  return hasLinkIkKinematicStateChanges(
    diffLinkIkDragKinematicState(seedState, nextState, resolveLinkIkCommittedStateEpsilon(preview)),
  );
}

export function diffLinkIkDragKinematicState(
  previousState: Partial<LinkIkDragKinematicState> | null | undefined,
  nextState: Partial<LinkIkDragKinematicState> | null | undefined,
  epsilon = LINK_IK_COMMIT_EPSILON,
): LinkIkDragKinematicState {
  const delta = createEmptyLinkIkDragKinematicState();
  const previousAngles = previousState?.angles ?? {};
  const nextAngles = nextState?.angles ?? {};
  const previousQuaternions = previousState?.quaternions ?? {};
  const nextQuaternions = nextState?.quaternions ?? {};

  Object.entries(nextAngles).forEach(([jointId, nextAngle]) => {
    if (hasMeaningfulNumberDelta(previousAngles[jointId], nextAngle, epsilon)) {
      delta.angles[jointId] = nextAngle;
    }
  });

  Object.entries(nextQuaternions).forEach(([jointId, nextQuaternion]) => {
    if (hasMeaningfulQuaternionDelta(previousQuaternions[jointId], nextQuaternion, epsilon)) {
      delta.quaternions[jointId] = nextQuaternion;
    }
  });

  return delta;
}

function limitNumberStep(previous: number | undefined, next: number, maxStep: number): number {
  if (!Number.isFinite(next) || previous === undefined || !Number.isFinite(previous)) {
    return next;
  }

  const delta = next - previous;
  if (Math.abs(delta) <= maxStep) {
    return next;
  }

  return previous + Math.sign(delta) * maxStep;
}

function limitQuaternionStep(
  previous: JointQuaternion | undefined,
  next: JointQuaternion,
  maxStepRadians: number,
): JointQuaternion {
  if (!previous) {
    return next;
  }

  const previousQuaternion = new THREE.Quaternion(previous.x, previous.y, previous.z, previous.w);
  const nextQuaternion = new THREE.Quaternion(next.x, next.y, next.z, next.w);
  const normalizedNextQuaternion = nextQuaternion.clone().normalize();
  const normalizedPreviousQuaternion = previousQuaternion.clone().normalize();

  let dot = normalizedPreviousQuaternion.dot(normalizedNextQuaternion);
  const shortestPathTarget =
    dot < 0
      ? new THREE.Quaternion(
          -normalizedNextQuaternion.x,
          -normalizedNextQuaternion.y,
          -normalizedNextQuaternion.z,
          -normalizedNextQuaternion.w,
        )
      : normalizedNextQuaternion;
  dot = Math.min(1, Math.max(-1, Math.abs(dot)));

  const deltaAngle = 2 * Math.acos(dot);
  if (!Number.isFinite(deltaAngle) || deltaAngle <= maxStepRadians) {
    return {
      x: shortestPathTarget.x,
      y: shortestPathTarget.y,
      z: shortestPathTarget.z,
      w: shortestPathTarget.w,
    };
  }

  const t = Math.min(1, maxStepRadians / deltaAngle);
  const limitedQuaternion = normalizedPreviousQuaternion.slerp(shortestPathTarget, t);
  return {
    x: limitedQuaternion.x,
    y: limitedQuaternion.y,
    z: limitedQuaternion.z,
    w: limitedQuaternion.w,
  };
}

export function limitLinkIkPreviewKinematicStateStep(
  previousState: Partial<LinkIkDragKinematicState> | null | undefined,
  nextState: Partial<LinkIkDragKinematicState> | null | undefined,
  {
    maxAngleStep = LINK_IK_PREVIEW_MAX_ANGLE_STEP,
    maxQuaternionStepRadians = LINK_IK_PREVIEW_MAX_QUATERNION_STEP_RADIANS,
    limitedJointIds,
  }: {
    maxAngleStep?: number;
    maxQuaternionStepRadians?: number;
    limitedJointIds?: ReadonlySet<string>;
  } = {},
): LinkIkDragKinematicState {
  const limitedState = createEmptyLinkIkDragKinematicState();
  const previousAngles = previousState?.angles ?? {};
  const previousQuaternions = previousState?.quaternions ?? {};

  Object.entries(nextState?.angles ?? {}).forEach(([jointId, nextAngle]) => {
    if (limitedJointIds && !limitedJointIds.has(jointId)) {
      limitedState.angles[jointId] = nextAngle;
      return;
    }

    limitedState.angles[jointId] = limitNumberStep(
      previousAngles[jointId],
      nextAngle,
      maxAngleStep,
    );
  });

  Object.entries(nextState?.quaternions ?? {}).forEach(([jointId, nextQuaternion]) => {
    if (limitedJointIds && !limitedJointIds.has(jointId)) {
      limitedState.quaternions[jointId] = nextQuaternion;
      return;
    }

    limitedState.quaternions[jointId] = limitQuaternionStep(
      previousQuaternions[jointId],
      nextQuaternion,
      maxQuaternionStepRadians,
    );
  });

  return limitedState;
}

function limitLinkIkPreviewKinematicStateStepForJoints(
  previousState: Partial<LinkIkDragKinematicState> | null | undefined,
  nextState: Partial<LinkIkDragKinematicState> | null | undefined,
  limitedJointIds: ReadonlySet<string>,
  {
    maxAngleStep = LINK_IK_PREVIEW_MAX_ANGLE_STEP,
    maxQuaternionStepRadians = LINK_IK_PREVIEW_MAX_QUATERNION_STEP_RADIANS,
  }: {
    maxAngleStep?: number;
    maxQuaternionStepRadians?: number;
  } = {},
): LinkIkDragKinematicState {
  const limitedState = createEmptyLinkIkDragKinematicState();
  const previousAngles = previousState?.angles ?? {};
  const previousQuaternions = previousState?.quaternions ?? {};

  Object.entries(nextState?.angles ?? {}).forEach(([jointId, nextAngle]) => {
    if (!limitedJointIds.has(jointId)) {
      return;
    }

    limitedState.angles[jointId] = limitNumberStep(
      previousAngles[jointId],
      nextAngle,
      maxAngleStep,
    );
  });

  Object.entries(nextState?.quaternions ?? {}).forEach(([jointId, nextQuaternion]) => {
    if (!limitedJointIds.has(jointId)) {
      return;
    }

    limitedState.quaternions[jointId] = limitQuaternionStep(
      previousQuaternions[jointId],
      nextQuaternion,
      maxQuaternionStepRadians,
    );
  });

  return limitedState;
}

export function resolveClosedLoopAwareLinkIkPreviewState({
  baseRobot,
  previousState,
  nextSolveState,
  limitedJointIds,
  maxAngleStep = LINK_IK_PREVIEW_MAX_ANGLE_STEP,
  maxQuaternionStepRadians = LINK_IK_PREVIEW_MAX_QUATERNION_STEP_RADIANS,
}: {
  baseRobot: Pick<RobotState, 'links' | 'joints' | 'rootLinkId' | 'closedLoopConstraints'>;
  previousState: Partial<LinkIkDragKinematicState> | null | undefined;
  nextSolveState: Partial<LinkIkDragKinematicState> | null | undefined;
  limitedJointIds?: ReadonlySet<string>;
  maxAngleStep?: number;
  maxQuaternionStepRadians?: number;
}): LinkIkDragKinematicState {
  if (!limitedJointIds || limitedJointIds.size === 0) {
    return limitLinkIkPreviewKinematicStateStep(previousState, nextSolveState, {
      maxAngleStep,
      maxQuaternionStepRadians,
    });
  }

  const limitedActiveState = limitLinkIkPreviewKinematicStateStepForJoints(
    previousState,
    nextSolveState,
    limitedJointIds,
    {
      maxAngleStep,
      maxQuaternionStepRadians,
    },
  );

  if (!baseRobot.closedLoopConstraints?.length) {
    return limitedActiveState;
  }

  const compensation = solveClosedLoopMotionCompensation(baseRobot, {
    angles: limitedActiveState.angles,
    quaternions: limitedActiveState.quaternions,
    lockedJointIds: [...limitedJointIds],
    maxIterations: 12,
    tolerance: 1e-5,
  });

  return {
    angles: {
      ...limitedActiveState.angles,
      ...compensation.angles,
    },
    quaternions: {
      ...limitedActiveState.quaternions,
      ...compensation.quaternions,
    },
  };
}

export function hasMeaningfulLinkIkTargetDelta(
  previousWorldPosition: THREE.Vector3 | null,
  nextWorldPosition: THREE.Vector3,
): boolean {
  if (!previousWorldPosition) {
    return true;
  }

  return previousWorldPosition.distanceToSquared(nextWorldPosition) > LINK_IK_TARGET_EPSILON_SQ;
}

export function shouldScheduleLinkIkPreviewSolve({
  pendingTargetWorldPosition,
  lastSolvedTargetWorldPosition,
  nextTargetWorldPosition,
  hasMeaningfulDragMotion,
}: {
  pendingTargetWorldPosition: THREE.Vector3 | null;
  lastSolvedTargetWorldPosition: THREE.Vector3 | null;
  nextTargetWorldPosition: THREE.Vector3;
  hasMeaningfulDragMotion: boolean;
}): boolean {
  if (!hasMeaningfulDragMotion) {
    return false;
  }

  if (pendingTargetWorldPosition) {
    return hasMeaningfulLinkIkTargetDelta(pendingTargetWorldPosition, nextTargetWorldPosition);
  }

  if (lastSolvedTargetWorldPosition) {
    return hasMeaningfulLinkIkTargetDelta(lastSolvedTargetWorldPosition, nextTargetWorldPosition);
  }

  return true;
}

export function resolveLinkIkSolveRequestOptions(
  preview: boolean,
):
  | Pick<
      LinkIkPositionSolveRequest,
      'coordinatePairMaxDistance' | 'maxIterations' | 'positionTolerance' | 'stallTolerance'
    >
  | undefined {
  if (!preview) {
    return undefined;
  }

  return {
    coordinatePairMaxDistance: LINK_IK_PREVIEW_COORDINATE_PAIR_MAX_DISTANCE,
    maxIterations: LINK_IK_PREVIEW_MAX_ITERATIONS,
    positionTolerance: LINK_IK_PREVIEW_POSITION_TOLERANCE,
    stallTolerance: LINK_IK_PREVIEW_STALL_TOLERANCE,
  };
}
