import * as THREE from 'three';

import {
  JointType,
  type RobotClosedLoopConstraint,
  type RobotData,
  type UrdfJoint,
  type UrdfLink,
  type UrdfOrigin,
  type UrdfVisual,
  type Vector3,
} from '@/types';

import {
  computeLinkWorldMatrices,
  createOriginMatrix,
  getParentJointByChildLink,
} from './kinematics';

const IDENTITY_MATRIX = new THREE.Matrix4().identity();
const SUPPORTED_DYNAMIC_REROOT_JOINT_TYPES = new Set<JointType>([
  JointType.FIXED,
  JointType.REVOLUTE,
  JointType.CONTINUOUS,
  JointType.PRISMATIC,
  JointType.FLOATING,
]);

function normalizeZero(value: number): number {
  return Object.is(value, -0) || Math.abs(value) < Number.EPSILON ? 0 : value;
}

function cloneVector3(vector: Vector3): Vector3 {
  return {
    x: Number.isFinite(vector?.x) ? vector.x : 0,
    y: Number.isFinite(vector?.y) ? vector.y : 0,
    z: Number.isFinite(vector?.z) ? vector.z : 0,
  };
}

function decomposeMatrixToOrigin(matrix: THREE.Matrix4, includeQuaternion = false): UrdfOrigin {
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  const euler = new THREE.Euler(0, 0, 0, 'ZYX');

  matrix.decompose(position, quaternion, scale);
  euler.setFromQuaternion(quaternion, 'ZYX');

  return {
    xyz: {
      x: normalizeZero(position.x),
      y: normalizeZero(position.y),
      z: normalizeZero(position.z),
    },
    rpy: {
      r: normalizeZero(euler.x),
      p: normalizeZero(euler.y),
      y: normalizeZero(euler.z),
    },
    ...(includeQuaternion
      ? {
          quatXyzw: {
            x: normalizeZero(quaternion.x),
            y: normalizeZero(quaternion.y),
            z: normalizeZero(quaternion.z),
            w: normalizeZero(quaternion.w),
          },
        }
      : {}),
  };
}

function multiplyOriginTransform(
  leftMatrix: THREE.Matrix4,
  origin: UrdfOrigin | undefined,
): UrdfOrigin {
  const nextMatrix = leftMatrix.clone().multiply(createOriginMatrix(origin));

  return decomposeMatrixToOrigin(nextMatrix, Boolean(origin?.quatXyzw));
}

function transformLocalPoint(transform: THREE.Matrix4, point: Vector3): Vector3 {
  const nextPoint = new THREE.Vector3(point.x, point.y, point.z).applyMatrix4(transform);
  return {
    x: normalizeZero(nextPoint.x),
    y: normalizeZero(nextPoint.y),
    z: normalizeZero(nextPoint.z),
  };
}

function cloneVisual(visual: UrdfVisual, transform?: THREE.Matrix4): UrdfVisual {
  return {
    ...visual,
    dimensions: { ...visual.dimensions },
    authoredMaterials: visual.authoredMaterials?.map((material) => ({ ...material })),
    mjcfHfield: visual.mjcfHfield
      ? {
          ...visual.mjcfHfield,
          size: visual.mjcfHfield.size ? { ...visual.mjcfHfield.size } : undefined,
          elevation: visual.mjcfHfield.elevation ? [...visual.mjcfHfield.elevation] : undefined,
        }
      : undefined,
    origin: transform
      ? multiplyOriginTransform(transform, visual.origin)
      : {
          ...visual.origin,
          xyz: cloneVector3(visual.origin.xyz),
          rpy: { ...visual.origin.rpy },
          ...(visual.origin.quatXyzw ? { quatXyzw: { ...visual.origin.quatXyzw } } : {}),
        },
  };
}

function cloneLinkWithFrameAdjustment(link: UrdfLink, transform?: THREE.Matrix4): UrdfLink {
  return {
    ...link,
    visual: cloneVisual(link.visual, transform),
    visualBodies: link.visualBodies?.map((body) => cloneVisual(body, transform)) ?? [],
    collision: cloneVisual(link.collision, transform),
    collisionBodies: link.collisionBodies?.map((body) => cloneVisual(body, transform)) ?? [],
    inertial: link.inertial
      ? {
          ...link.inertial,
          origin: link.inertial.origin
            ? transform
              ? multiplyOriginTransform(transform, link.inertial.origin)
              : {
                  ...link.inertial.origin,
                  xyz: cloneVector3(link.inertial.origin.xyz),
                  rpy: { ...link.inertial.origin.rpy },
                  ...(link.inertial.origin.quatXyzw
                    ? { quatXyzw: { ...link.inertial.origin.quatXyzw } }
                    : {}),
                }
            : undefined,
          inertia: { ...link.inertial.inertia },
        }
      : undefined,
  };
}

function cloneJointOrigin(origin: UrdfOrigin): UrdfOrigin {
  return {
    ...origin,
    xyz: cloneVector3(origin.xyz),
    rpy: { ...origin.rpy },
    ...(origin.quatXyzw ? { quatXyzw: { ...origin.quatXyzw } } : {}),
  };
}

function cloneJoint(joint: UrdfJoint): UrdfJoint {
  return {
    ...joint,
    origin: cloneJointOrigin(joint.origin),
    axis: joint.axis ? cloneVector3(joint.axis) : undefined,
    limit: joint.limit ? { ...joint.limit } : undefined,
    dynamics: { ...joint.dynamics },
    hardware: { ...joint.hardware },
    mimic: joint.mimic ? { ...joint.mimic } : undefined,
    calibration: joint.calibration ? { ...joint.calibration } : undefined,
    safetyController: joint.safetyController ? { ...joint.safetyController } : undefined,
    quaternion: joint.quaternion ? { ...joint.quaternion } : undefined,
  };
}

function findPathJointIdsFromRoot(
  robot: RobotData,
  targetRootLinkId: string,
  componentId: string,
): string[] {
  if (!robot.links[targetRootLinkId]) {
    throw new Error(
      `Cannot reroot assembly component "${componentId}" because link "${targetRootLinkId}" does not exist`,
    );
  }

  const parentJointByChild = getParentJointByChildLink(robot);
  const jointIds: string[] = [];
  const visitedLinkIds = new Set<string>();
  let currentLinkId = targetRootLinkId;

  while (currentLinkId !== robot.rootLinkId) {
    if (visitedLinkIds.has(currentLinkId)) {
      throw new Error(
        `Cannot reroot assembly component "${componentId}" because the link path to root contains a cycle at "${currentLinkId}"`,
      );
    }
    visitedLinkIds.add(currentLinkId);

    const parentJoint = parentJointByChild.get(currentLinkId);
    if (!parentJoint) {
      throw new Error(
        `Cannot reroot assembly component "${componentId}" because link "${targetRootLinkId}" is disconnected from root "${robot.rootLinkId}"`,
      );
    }

    jointIds.push(parentJoint.id);
    currentLinkId = parentJoint.parentLinkId;
  }

  return jointIds;
}

function negateAxis(axis: Vector3 | undefined): Vector3 {
  const normalizedAxis = axis ? cloneVector3(axis) : { x: 0, y: 0, z: 1 };
  return {
    x: normalizeZero(-normalizedAxis.x),
    y: normalizeZero(-normalizedAxis.y),
    z: normalizeZero(-normalizedAxis.z),
  };
}

function reversePathJoint(
  joint: UrdfJoint,
  frameAdjustmentByLinkId: Map<string, THREE.Matrix4>,
  componentId: string,
): UrdfJoint {
  if (!SUPPORTED_DYNAMIC_REROOT_JOINT_TYPES.has(joint.type)) {
    throw new Error(
      `Cannot reroot assembly component "${componentId}" through unsupported joint "${joint.id}" of type "${joint.type}"`,
    );
  }

  const parentFrameAdjustment = frameAdjustmentByLinkId.get(joint.childLinkId) ?? IDENTITY_MATRIX;
  const nextJoint = cloneJoint(joint);
  nextJoint.parentLinkId = joint.childLinkId;
  nextJoint.childLinkId = joint.parentLinkId;
  nextJoint.origin = decomposeMatrixToOrigin(parentFrameAdjustment, false);

  if (
    joint.type === JointType.REVOLUTE ||
    joint.type === JointType.CONTINUOUS ||
    joint.type === JointType.PRISMATIC
  ) {
    nextJoint.axis = negateAxis(joint.axis);
  }

  return nextJoint;
}

function cloneConstraintWithFrameAdjustments(
  constraint: RobotClosedLoopConstraint,
  frameAdjustmentByLinkId: Map<string, THREE.Matrix4>,
): RobotClosedLoopConstraint {
  const nextConstraint: RobotClosedLoopConstraint = {
    ...constraint,
    anchorWorld: cloneVector3(constraint.anchorWorld),
    anchorLocalA: cloneVector3(constraint.anchorLocalA),
    anchorLocalB: cloneVector3(constraint.anchorLocalB),
    source: constraint.source ? { ...constraint.source } : undefined,
  };

  const linkATransform = frameAdjustmentByLinkId.get(constraint.linkAId);
  if (linkATransform) {
    nextConstraint.anchorLocalA = transformLocalPoint(linkATransform, nextConstraint.anchorLocalA);
  }

  const linkBTransform = frameAdjustmentByLinkId.get(constraint.linkBId);
  if (linkBTransform) {
    nextConstraint.anchorLocalB = transformLocalPoint(linkBTransform, nextConstraint.anchorLocalB);
  }

  return nextConstraint;
}

function recomputeConstraintAnchorWorlds(
  robot: RobotData,
): RobotClosedLoopConstraint[] | undefined {
  if (!robot.closedLoopConstraints?.length) {
    return robot.closedLoopConstraints;
  }

  const linkWorldMatrices = computeLinkWorldMatrices(robot);
  return robot.closedLoopConstraints.map((constraint) => {
    const linkAMatrix = linkWorldMatrices[constraint.linkAId];
    if (!linkAMatrix) {
      return constraint;
    }

    return {
      ...constraint,
      anchorWorld: transformLocalPoint(linkAMatrix, constraint.anchorLocalA),
    };
  });
}

export function rerootAssemblyComponentRobot(
  robot: RobotData,
  targetRootLinkId: string,
  componentId: string,
): RobotData {
  if (targetRootLinkId === robot.rootLinkId) {
    return robot;
  }

  const pathJointIds = findPathJointIdsFromRoot(robot, targetRootLinkId, componentId);
  if (pathJointIds.length === 0) {
    return robot;
  }

  const pathJointIdSet = new Set(pathJointIds);
  const frameAdjustmentByLinkId = new Map<string, THREE.Matrix4>([
    [targetRootLinkId, IDENTITY_MATRIX.clone()],
  ]);

  pathJointIds.forEach((jointId) => {
    const joint = robot.joints[jointId];
    if (!joint) {
      throw new Error(
        `Cannot reroot assembly component "${componentId}" because joint "${jointId}" does not exist`,
      );
    }

    if (!SUPPORTED_DYNAMIC_REROOT_JOINT_TYPES.has(joint.type)) {
      throw new Error(
        `Cannot reroot assembly component "${componentId}" through unsupported joint "${joint.id}" of type "${joint.type}"`,
      );
    }

    frameAdjustmentByLinkId.set(
      joint.parentLinkId,
      createOriginMatrix(joint.origin).clone().invert(),
    );
  });

  const links: RobotData['links'] = {};
  Object.entries(robot.links).forEach(([linkId, link]) => {
    const frameAdjustment = frameAdjustmentByLinkId.get(linkId);
    links[linkId] = cloneLinkWithFrameAdjustment(link, frameAdjustment);
  });

  const joints: RobotData['joints'] = {};
  Object.entries(robot.joints).forEach(([jointId, joint]) => {
    if (pathJointIdSet.has(jointId)) {
      joints[jointId] = reversePathJoint(joint, frameAdjustmentByLinkId, componentId);
      return;
    }

    const nextJoint = cloneJoint(joint);
    const parentFrameAdjustment = frameAdjustmentByLinkId.get(joint.parentLinkId);
    if (parentFrameAdjustment) {
      nextJoint.origin = multiplyOriginTransform(parentFrameAdjustment, joint.origin);
    }
    joints[jointId] = nextJoint;
  });

  const rerootedRobot: RobotData = {
    ...robot,
    links,
    joints,
    rootLinkId: targetRootLinkId,
    closedLoopConstraints: robot.closedLoopConstraints?.map((constraint) =>
      cloneConstraintWithFrameAdjustments(constraint, frameAdjustmentByLinkId),
    ),
  };

  return {
    ...rerootedRobot,
    closedLoopConstraints: recomputeConstraintAnchorWorlds(rerootedRobot),
  };
}
