import {
  GeometryType,
  IMPORTED_EXTERNAL_FRAME_LINK_TYPE,
  type RobotState,
  type UrdfLink,
} from '@/types';
import { getVisualGeometryEntries } from './visualBodies';

const SYNTHETIC_JOINT_STAGE_MARKER = '__joint_stage_';

function hasRenderableGeometry(link: UrdfLink | undefined): boolean {
  if (!link) {
    return false;
  }

  if (getVisualGeometryEntries(link).length > 0) {
    return true;
  }

  if (link.collision.type !== GeometryType.NONE) {
    return true;
  }

  return (link.collisionBodies || []).some((body) => body.type !== GeometryType.NONE);
}

export function isSyntheticJointStageLink(link: UrdfLink | undefined): boolean {
  if (!link) {
    return false;
  }

  const linkName = (link.name || '').trim();
  if (!linkName.includes(SYNTHETIC_JOINT_STAGE_MARKER)) {
    return false;
  }

  if ((link.inertial?.mass || 0) > 0) {
    return false;
  }

  return !hasRenderableGeometry(link);
}

export function isSyntheticWorldRoot(robot: RobotState, linkId: string): boolean {
  if (robot.rootLinkId !== linkId) {
    return false;
  }

  const link = robot.links[linkId];
  if (!link) {
    return false;
  }

  if ((link.name || '').trim().toLowerCase() !== 'world') {
    return false;
  }

  if ((link.inertial?.mass || 0) > 0) {
    return false;
  }

  if (hasRenderableGeometry(link)) {
    return false;
  }

  return Object.values(robot.joints).some((joint) => joint.parentLinkId === linkId);
}

function isImportedExternalFrameRoot(robot: RobotState, linkId: string): boolean {
  if (robot.rootLinkId !== linkId) {
    return false;
  }

  const link = robot.links[linkId];
  if (!link || link.type !== IMPORTED_EXTERNAL_FRAME_LINK_TYPE) {
    return false;
  }

  if ((link.inertial?.mass || 0) > 0 || hasRenderableGeometry(link)) {
    return false;
  }

  return Object.values(robot.joints).some((joint) => joint.parentLinkId === linkId);
}

function isTransparentRootFrame(robot: RobotState, linkId: string): boolean {
  return isSyntheticWorldRoot(robot, linkId) || isImportedExternalFrameRoot(robot, linkId);
}

export function isTransparentDisplayLink(robot: RobotState, linkId: string): boolean {
  const link = robot.links[linkId];
  if (!link) {
    return false;
  }

  return isTransparentRootFrame(robot, linkId) || isSyntheticJointStageLink(link);
}

export function getTreeDisplayRootLinkIds(robot: RobotState): string[] {
  if (!isTransparentRootFrame(robot, robot.rootLinkId)) {
    return robot.links[robot.rootLinkId] ? [robot.rootLinkId] : [];
  }

  const childRootIds = Object.values(robot.joints)
    .filter((joint) => joint.parentLinkId === robot.rootLinkId)
    .map((joint) => joint.childLinkId)
    .filter((linkId, index, ids) => ids.indexOf(linkId) === index && Boolean(robot.links[linkId]));

  return childRootIds.length > 0 ? childRootIds : [robot.rootLinkId];
}

export function getPrimaryTreeDisplayRootLinkId(robot: RobotState): string | null {
  return getTreeDisplayRootLinkIds(robot)[0] ?? null;
}

export function getTreeRenderRootLinkIds(robot: RobotState): string[] {
  if (isTransparentRootFrame(robot, robot.rootLinkId)) {
    return robot.links[robot.rootLinkId] ? [robot.rootLinkId] : [];
  }

  return getTreeDisplayRootLinkIds(robot);
}

export function getPrimaryTreeRenderRootLinkId(robot: RobotState): string | null {
  return getTreeRenderRootLinkIds(robot)[0] ?? null;
}
