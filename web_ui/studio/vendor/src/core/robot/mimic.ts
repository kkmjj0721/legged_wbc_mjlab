import type { RobotData, UrdfJoint } from '@/types';

import type { JointAngleOverrideMap } from './kinematics';
import { resolveJointKey } from './identity';

const MIMIC_MULTIPLIER_EPSILON = 1e-9;

export interface MimicJointAngleTargets {
  driverJointId: string;
  driverAngle: number;
  angles: JointAngleOverrideMap;
  lockedJointIds: string[];
}

function getMimicMultiplier(joint: Pick<UrdfJoint, 'mimic'>): number {
  return Number.isFinite(joint.mimic?.multiplier) ? joint.mimic!.multiplier! : 1;
}

function getMimicOffset(joint: Pick<UrdfJoint, 'mimic'>): number {
  return Number.isFinite(joint.mimic?.offset) ? joint.mimic!.offset! : 0;
}

function buildFollowerJointMap(
  joints: Record<string, UrdfJoint>,
): Map<string, Array<{ jointId: string; joint: UrdfJoint }>> {
  const followersByDriver = new Map<string, Array<{ jointId: string; joint: UrdfJoint }>>();

  Object.entries(joints).forEach(([jointId, joint]) => {
    const driverJointId = resolveJointKey(joints, joint.mimic?.joint);
    if (!driverJointId) {
      return;
    }

    const existingFollowers = followersByDriver.get(driverJointId) ?? [];
    existingFollowers.push({ jointId, joint });
    followersByDriver.set(driverJointId, existingFollowers);
  });

  return followersByDriver;
}

export function resolveMimicJointAngleTargets(
  robot: Pick<RobotData, 'joints'>,
  selectedJointId: string,
  selectedAngle: number,
): MimicJointAngleTargets {
  const selectedJoint = robot.joints[selectedJointId];
  if (!selectedJoint) {
    return {
      driverJointId: selectedJointId,
      driverAngle: selectedAngle,
      angles: { [selectedJointId]: selectedAngle },
      lockedJointIds: [selectedJointId],
    };
  }

  let driverJointId = selectedJointId;
  let driverAngle = selectedAngle;
  const visitedDrivers = new Set<string>();

  while (!visitedDrivers.has(driverJointId)) {
    visitedDrivers.add(driverJointId);

    const currentJoint = robot.joints[driverJointId];
    const nextDriverJointId = resolveJointKey(robot.joints, currentJoint?.mimic?.joint);
    if (!currentJoint?.mimic || !nextDriverJointId) {
      break;
    }

    const multiplier = getMimicMultiplier(currentJoint);
    if (Math.abs(multiplier) <= MIMIC_MULTIPLIER_EPSILON) {
      break;
    }

    driverAngle = (driverAngle - getMimicOffset(currentJoint)) / multiplier;
    driverJointId = nextDriverJointId;
  }

  const followersByDriver = buildFollowerJointMap(robot.joints);
  const angles: JointAngleOverrideMap = { [driverJointId]: driverAngle };
  const lockedJointIds = new Set<string>([driverJointId]);
  const propagationQueue = [{ jointId: driverJointId, angle: driverAngle }];

  while (propagationQueue.length > 0) {
    const current = propagationQueue.shift();
    if (!current) {
      continue;
    }

    const followerEntries = followersByDriver.get(current.jointId) ?? [];
    followerEntries.forEach(({ jointId, joint }) => {
      if (lockedJointIds.has(jointId)) {
        return;
      }

      const followerAngle = current.angle * getMimicMultiplier(joint) + getMimicOffset(joint);
      angles[jointId] = followerAngle;
      lockedJointIds.add(jointId);
      propagationQueue.push({ jointId, angle: followerAngle });
    });
  }

  return {
    driverJointId,
    driverAngle,
    angles,
    lockedJointIds: [...lockedJointIds],
  };
}
