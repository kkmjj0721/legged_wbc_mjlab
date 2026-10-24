/**
 * Robot Data Builders
 * Factory functions for creating robot components with sensible defaults
 */

import type { UrdfLink, UrdfJoint, RobotState, UrdfOrigin } from '@/types';
import { DEFAULT_LINK, DEFAULT_JOINT, GeometryType, JointType } from '@/types';
import {
  LINK_ID_PREFIX,
  JOINT_ID_PREFIX,
  DEFAULT_ROBOT_NAME,
  DEFAULT_VISUAL_COLOR,
  DEFAULT_JOINT_OFFSET_Z,
} from './constants';
import { resolveLinkRenderableBounds } from './assemblyPlacement';

/**
 * Generate a unique ID with the given prefix
 */
export const generateId = (prefix: string): string => {
  return `${prefix}${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
};

/**
 * Generate a unique link ID
 */
export const generateLinkId = (): string => generateId(LINK_ID_PREFIX);

/**
 * Generate a unique joint ID
 */
export const generateJointId = (): string => generateId(JOINT_ID_PREFIX);

function cloneOrigin(origin: UrdfOrigin): UrdfOrigin;
function cloneOrigin(origin: undefined): undefined;
function cloneOrigin(origin: UrdfOrigin | undefined): UrdfOrigin | undefined;
function cloneOrigin(origin?: UrdfOrigin): UrdfOrigin | undefined {
  return origin
    ? {
        xyz: { ...origin.xyz },
        rpy: { ...origin.rpy },
        ...(origin.quatXyzw ? { quatXyzw: { ...origin.quatXyzw } } : {}),
      }
    : undefined;
}

function resolveAttachedChildGeometryOriginZ(
  geometry: Pick<UrdfLink['visual'], 'type' | 'dimensions'> | undefined,
): number {
  if (!geometry) {
    return 0;
  }

  switch (geometry.type) {
    case GeometryType.BOX:
      return Math.max((geometry.dimensions?.z ?? 0) * 0.5, 0);
    case GeometryType.CYLINDER:
    case GeometryType.CAPSULE:
      return Math.max((geometry.dimensions?.y ?? 0) * 0.5, 0);
    default:
      return 0;
  }
}

function createOffsetOrigin(origin: UrdfOrigin, offsetZ: number): UrdfOrigin;
function createOffsetOrigin(origin: undefined, offsetZ: number): undefined;
function createOffsetOrigin(origin: UrdfOrigin | undefined, offsetZ: number): UrdfOrigin | undefined;
function createOffsetOrigin(origin: UrdfOrigin | undefined, offsetZ: number): UrdfOrigin | undefined {
  if (!origin) {
    return origin;
  }

  return {
    ...origin,
    xyz: {
      ...origin.xyz,
      z: offsetZ,
    },
  };
}

export function resolveDefaultChildJointOrigin(
  parentLink: UrdfLink | undefined,
  yOffset: number = 0,
): UrdfOrigin {
  const parentTopZ = parentLink ? resolveLinkRenderableBounds(parentLink)?.bounds.max.z : undefined;
  const zOffset = Number.isFinite(parentTopZ) ? Number(parentTopZ) : DEFAULT_JOINT_OFFSET_Z;

  return {
    xyz: { x: 0, y: yOffset, z: zOffset },
    rpy: { r: 0, p: 0, y: 0 },
  };
}

/**
 * Create a new link with default values
 */
export const createLink = (options: Partial<UrdfLink> = {}): UrdfLink => {
  const id = options.id || generateLinkId();
  return {
    ...DEFAULT_LINK,
    id,
    name: options.name || id,
    visual: {
      ...DEFAULT_LINK.visual,
      color: DEFAULT_VISUAL_COLOR,
      ...options.visual,
    },
    visualBodies: options.visualBodies?.map((body) => ({ ...body })) || [],
    collision: {
      ...DEFAULT_LINK.collision,
      ...options.collision,
    },
    collisionBodies: options.collisionBodies?.map((body) => ({ ...body })) || [],
    inertial: {
      ...DEFAULT_LINK.inertial,
      ...options.inertial,
    },
    visible: options.visible ?? true,
    ...options,
  };
};

/**
 * Create a child link whose default geometry extends away from the parent joint
 * instead of centering the joint inside the link body.
 */
export const createAttachedChildLink = (options: Partial<UrdfLink> = {}): UrdfLink => {
  const link = createLink(options);
  const visualOriginZ = resolveAttachedChildGeometryOriginZ(link.visual);
  const collisionOriginZ = resolveAttachedChildGeometryOriginZ(link.collision);
  const inertialOriginZ = visualOriginZ || collisionOriginZ;

  return {
    ...link,
    visual: options.visual?.origin
      ? link.visual
      : {
          ...link.visual,
          origin: createOffsetOrigin(link.visual.origin, visualOriginZ),
        },
    collision: options.collision?.origin
      ? link.collision
      : {
          ...link.collision,
          origin: createOffsetOrigin(link.collision.origin, collisionOriginZ),
        },
    inertial: link.inertial
      ? options.inertial?.origin
        ? link.inertial
        : {
            ...link.inertial,
            origin: createOffsetOrigin(link.inertial.origin, inertialOriginZ),
          }
      : link.inertial,
  };
};

/**
 * Create a new joint with default values
 */
export const createJoint = (
  options: Partial<UrdfJoint> & {
    parentLinkId: string;
    childLinkId: string;
  },
): UrdfJoint => {
  const {
    id: optionId,
    name: optionName,
    type,
    parentLinkId,
    childLinkId,
    origin,
    axis,
    limit,
    dynamics,
    hardware,
    ...jointOptions
  } = options;
  const id = optionId || generateJointId();
  const hasLimit = Object.prototype.hasOwnProperty.call(options, 'limit');
  return {
    ...DEFAULT_JOINT,
    ...jointOptions,
    id,
    name: optionName || id,
    type: type || JointType.REVOLUTE,
    parentLinkId,
    childLinkId,
    origin: {
      xyz: { x: 0, y: 0, z: 0.5 },
      rpy: { r: 0, p: 0, y: 0 },
      ...origin,
    },
    axis: axis || { x: 0, y: 0, z: 1 },
    limit: hasLimit ? (limit ? { ...DEFAULT_JOINT.limit, ...limit } : undefined) : { ...DEFAULT_JOINT.limit },
    dynamics: {
      ...DEFAULT_JOINT.dynamics,
      ...dynamics,
    },
    hardware: {
      ...DEFAULT_JOINT.hardware,
      ...hardware,
    },
  };
};

/**
 * Create a new empty robot state with a single root link
 */
export const createEmptyRobot = (name: string = DEFAULT_ROBOT_NAME): RobotState => {
  const rootLinkId = generateLinkId();
  const rootLink = createAttachedChildLink({
    id: rootLinkId,
    name: 'base_link',
  });

  return {
    name,
    links: { [rootLinkId]: rootLink },
    joints: {},
    rootLinkId,
    selection: { type: 'link', id: rootLinkId },
  };
};

/**
 * Add a child link and joint to an existing robot
 */
export const addChildToRobot = (
  robot: RobotState,
  parentLinkId: string,
  linkOptions: Partial<UrdfLink> = {},
  jointOptions: Partial<UrdfJoint> = {},
): RobotState => {
  // Calculate offset based on existing siblings
  const siblings = Object.values(robot.joints).filter((j) => j.parentLinkId === parentLinkId);
  const yOffset = siblings.length * 0.5;
  const parentLink = robot.links[parentLinkId];

  const newLink = createAttachedChildLink({
    name: `link_${Object.keys(robot.links).length + 1}`,
    ...linkOptions,
  });

  const newJoint = createJoint({
    name: `joint_${Object.keys(robot.joints).length + 1}`,
    parentLinkId,
    childLinkId: newLink.id,
    origin: resolveDefaultChildJointOrigin(parentLink, yOffset),
    ...jointOptions,
  });

  return {
    ...robot,
    links: { ...robot.links, [newLink.id]: newLink },
    joints: { ...robot.joints, [newJoint.id]: newJoint },
    selection: { type: 'joint', id: newJoint.id },
  };
};

/**
 * Clone a link with a new ID
 */
export const cloneLink = (link: UrdfLink, newId?: string): UrdfLink => {
  const id = newId || generateLinkId();
  return {
    ...link,
    id,
    name: `${link.name}_copy`,
    visual: {
      ...link.visual,
      origin: cloneOrigin(link.visual.origin),
      dimensions: { ...link.visual.dimensions },
      authoredMaterials: link.visual.authoredMaterials?.map((material) => ({ ...material })),
    },
    visualBodies:
      link.visualBodies?.map((body) => ({
        ...body,
        origin: cloneOrigin(body.origin),
        dimensions: { ...body.dimensions },
        authoredMaterials: body.authoredMaterials?.map((material) => ({ ...material })),
      })) || [],
    collision: {
      ...link.collision,
      origin: cloneOrigin(link.collision.origin),
      dimensions: { ...link.collision.dimensions },
    },
    collisionBodies:
      link.collisionBodies?.map((body) => ({
        ...body,
        origin: cloneOrigin(body.origin),
        dimensions: { ...body.dimensions },
      })) || [],
    inertial: link.inertial
      ? {
          ...link.inertial,
          origin: cloneOrigin(link.inertial.origin),
          inertia: { ...link.inertial.inertia },
        }
      : undefined,
  };
};

/**
 * Clone a joint with a new ID and updated link references
 */
export const cloneJoint = (
  joint: UrdfJoint,
  newParentLinkId: string,
  newChildLinkId: string,
  newId?: string,
): UrdfJoint => {
  const id = newId || generateJointId();
  return {
    ...joint,
    id,
    name: `${joint.name}_copy`,
    parentLinkId: newParentLinkId,
    childLinkId: newChildLinkId,
    origin: cloneOrigin(joint.origin)!,
    axis: joint.axis ? { ...joint.axis } : undefined,
    limit: joint.limit ? { ...joint.limit } : undefined,
    dynamics: { ...joint.dynamics },
    hardware: { ...joint.hardware },
  };
};
