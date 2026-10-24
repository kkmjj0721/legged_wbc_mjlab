/**
 * MJCF keyframe -> initial pose
 *
 * Decodes a MuJoCo <keyframe> qpos array into per-joint initial values and
 * applies them onto a built RobotState. Pure logic extracted from
 * mjcfParser.ts: walks the body tree to map each joint to its qpos address,
 * picks the "home"/first usable keyframe, then writes free/ball/scalar joint
 * poses back onto the robot.
 */

import { RobotState } from '@/types';

import { toRPYObjectFromQuat } from './mjcfMath';
import { type MJCFModelKeyframe } from './mjcfModel';

type MJCFQposJointKind = 'free' | 'ball' | 'scalar';

interface MJCFQposJointBinding {
  jointName: string;
  kind: MJCFQposJointKind;
  qposAddress: number;
  qposSize: number;
}

/** Minimal joint shape needed to size and classify a qpos slot. */
interface QposJointSource {
  name: string;
  type: string;
}

/** Minimal (recursive) body shape needed to walk qpos joint bindings. */
interface QposBodySource {
  joints: QposJointSource[];
  children: QposBodySource[];
}

function getJointQposSize(joint: QposJointSource): number {
  switch (joint.type.toLowerCase()) {
    case 'free':
      return 7;
    case 'ball':
      return 4;
    case 'hinge':
    case 'slide':
      return 1;
    default:
      return 0;
  }
}

function getQposJointKind(joint: QposJointSource): MJCFQposJointKind {
  switch (joint.type.toLowerCase()) {
    case 'free':
      return 'free';
    case 'ball':
      return 'ball';
    default:
      return 'scalar';
  }
}

function collectQposJointBindings(
  body: QposBodySource,
  bindings: MJCFQposJointBinding[] = [],
  qposCursor = { value: 0 },
): MJCFQposJointBinding[] {
  body.joints.forEach((joint) => {
    const qposSize = getJointQposSize(joint);
    if (qposSize <= 0) {
      return;
    }

    bindings.push({
      jointName: joint.name,
      kind: getQposJointKind(joint),
      qposAddress: qposCursor.value,
      qposSize,
    });
    qposCursor.value += qposSize;
  });

  body.children.forEach((child) => collectQposJointBindings(child, bindings, qposCursor));
  return bindings;
}

function getExpectedQposLength(bindings: MJCFQposJointBinding[]): number {
  return bindings.reduce(
    (length, binding) => Math.max(length, binding.qposAddress + binding.qposSize),
    0,
  );
}

function selectInitialPoseKeyframe(
  keyframes: MJCFModelKeyframe[],
  expectedQposLength: number,
): MJCFModelKeyframe | null {
  if (expectedQposLength <= 0) {
    return null;
  }

  const usableKeyframes = keyframes.filter((keyframe) => {
    return keyframe.qpos && keyframe.qpos.length >= expectedQposLength;
  });
  return (
    usableKeyframes.find((keyframe) => keyframe.name?.trim().toLowerCase() === 'home') ??
    usableKeyframes[0] ??
    null
  );
}

function readFiniteQposValue(qpos: number[], index: number): number | null {
  const value = qpos[index];
  return Number.isFinite(value) ? value! : null;
}

function readFiniteQposQuaternion(
  qpos: number[],
  address: number,
): { w: number; x: number; y: number; z: number } | null {
  const w = readFiniteQposValue(qpos, address);
  const x = readFiniteQposValue(qpos, address + 1);
  const y = readFiniteQposValue(qpos, address + 2);
  const z = readFiniteQposValue(qpos, address + 3);
  if (w == null || x == null || y == null || z == null) {
    return null;
  }

  const length = Math.hypot(w, x, y, z);
  if (length <= 1e-12) {
    return null;
  }

  return {
    w: w / length,
    x: x / length,
    y: y / length,
    z: z / length,
  };
}

export function applyInitialPoseKeyframe(
  robot: Pick<RobotState, 'joints'>,
  worldBody: QposBodySource,
  keyframes: MJCFModelKeyframe[],
): void {
  if (keyframes.length === 0) {
    return;
  }

  const qposBindings = collectQposJointBindings(worldBody);
  const keyframe = selectInitialPoseKeyframe(keyframes, getExpectedQposLength(qposBindings));
  const qpos = keyframe?.qpos;
  if (!qpos) {
    return;
  }

  qposBindings.forEach((binding) => {
    const joint = robot.joints[binding.jointName];
    if (!joint) {
      return;
    }

    if (binding.kind === 'free') {
      const x = readFiniteQposValue(qpos, binding.qposAddress);
      const y = readFiniteQposValue(qpos, binding.qposAddress + 1);
      const z = readFiniteQposValue(qpos, binding.qposAddress + 2);
      const quaternion = readFiniteQposQuaternion(qpos, binding.qposAddress + 3);
      if (x == null || y == null || z == null || !quaternion) {
        return;
      }

      joint.origin = {
        xyz: { x, y, z },
        rpy: toRPYObjectFromQuat(quaternion) ?? joint.origin.rpy,
      };
      return;
    }

    if (binding.kind === 'ball') {
      const quaternion = readFiniteQposQuaternion(qpos, binding.qposAddress);
      if (!quaternion) {
        return;
      }

      joint.quaternion = {
        x: quaternion.x,
        y: quaternion.y,
        z: quaternion.z,
        w: quaternion.w,
      };
      return;
    }

    const value = readFiniteQposValue(qpos, binding.qposAddress);
    if (value != null) {
      joint.angle = value;
    }
  });
}
