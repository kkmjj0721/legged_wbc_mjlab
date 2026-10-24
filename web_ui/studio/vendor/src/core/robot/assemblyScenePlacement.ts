import {
  DEFAULT_JOINT,
  DEFAULT_LINK,
  GeometryType,
  JointType,
  entityRefKey,
  type AssemblyState,
  type AssemblyTransform,
  type BridgeEntityRef,
  type RobotData,
  type UrdfJoint,
  type UrdfLink,
} from '@/types';

import type { AssemblySceneProjection } from './assemblySceneProjection';
import { cloneAssemblyTransform } from './assemblyTransformUtils';

export type AssemblyComponentSceneTransformTarget =
  | {
      readonly kind: 'component-root';
      readonly componentId: string;
      readonly runtimeJointId: string;
    }
  | {
      readonly kind: 'bridge';
      readonly componentId: string;
      readonly bridgeId: string;
      readonly runtimeJointId: string;
    };

/**
 * Renderer-ready placement derived from canonical workspace state.
 *
 * Assembly placement stays outside RobotData so a stable scene root can apply
 * it exactly once. Assembled scenes receive private fixed joints only for
 * connected-graph roots; bridged child placement remains owned by bridge
 * origins. Private IDs are deliberately absent from the projection maps.
 */
export interface AssemblyScenePlacement {
  readonly robotData: RobotData;
  readonly renderStrategy: AssemblySceneProjection['renderStrategy'];
  readonly assemblyTransform: AssemblyTransform;
  readonly directComponentId: string | null;
  readonly directComponentTransform: AssemblyTransform | null;
  readonly componentTransformTargets: ReadonlyMap<
    string,
    AssemblyComponentSceneTransformTarget
  >;
}

function createSceneRootLink(id: string): UrdfLink {
  return {
    ...structuredClone(DEFAULT_LINK),
    id,
    name: id,
    visual: {
      ...structuredClone(DEFAULT_LINK.visual),
      type: GeometryType.NONE,
      dimensions: { x: 0, y: 0, z: 0 },
    },
    collision: {
      ...structuredClone(DEFAULT_LINK.collision),
      type: GeometryType.NONE,
      dimensions: { x: 0, y: 0, z: 0 },
    },
    inertial: {
      mass: 0,
      origin: { xyz: { x: 0, y: 0, z: 0 }, rpy: { r: 0, p: 0, y: 0 } },
      inertia: { ixx: 0, ixy: 0, ixz: 0, iyy: 0, iyz: 0, izz: 0 },
    },
  };
}

function createSceneRootJoint(
  id: string,
  parentLinkId: string,
  childLinkId: string,
  transform: AssemblyTransform,
): UrdfJoint {
  return {
    ...structuredClone(DEFAULT_JOINT),
    id,
    name: id,
    type: JointType.FIXED,
    parentLinkId,
    childLinkId,
    origin: {
      xyz: { ...transform.position },
      rpy: { ...transform.rotation },
    },
    axis: undefined,
    limit: undefined,
  };
}

function allocateSyntheticId(usedIds: Set<string>, preferredId: string): string {
  let id = preferredId;
  let suffix = 1;
  while (usedIds.has(id)) {
    id = `${preferredId}__scene${suffix}`;
    suffix += 1;
  }
  usedIds.add(id);
  return id;
}

function collectTopLevelRootLinkIds(robot: RobotData): string[] {
  const childLinkIds = new Set(Object.values(robot.joints).map((joint) => joint.childLinkId));
  return Object.keys(robot.links).filter((linkId) => !childLinkIds.has(linkId));
}

function getVisibleIncomingBridges(
  workspace: AssemblyState,
  componentId: string,
): AssemblyState['bridges'][string][] {
  return Object.values(workspace.bridges).filter((bridge) => {
    const parent = workspace.components[bridge.parentComponentId];
    const child = workspace.components[bridge.childComponentId];
    return (
      bridge.childComponentId === componentId &&
      parent?.visible !== false &&
      child?.visible !== false
    );
  });
}

function resolveComponentIdForGlobalLink(
  projection: AssemblySceneProjection,
  linkId: string,
): string | null {
  const ref = projection.globalToEntityRef.get(linkId);
  return ref?.type === 'link' ? ref.componentId : null;
}

function createAssembledScenePlacement(
  workspace: AssemblyState,
  projection: AssemblySceneProjection,
): AssemblyScenePlacement {
  const robot = projection.robotData;
  const topLevelRootLinkIds = collectTopLevelRootLinkIds(robot);
  const usedIds = new Set([
    ...Object.keys(robot.links),
    ...Object.keys(robot.joints),
    ...Array.from(projection.globalToEntityRef.keys()),
  ]);
  const worldRootLinkId = allocateSyntheticId(usedIds, '__workspace_scene_root__');
  const links: RobotData['links'] = {
    ...robot.links,
    [worldRootLinkId]: createSceneRootLink(worldRootLinkId),
  };
  const joints: RobotData['joints'] = { ...robot.joints };
  const componentTransformTargets = new Map<string, AssemblyComponentSceneTransformTarget>();

  topLevelRootLinkIds.forEach((rootLinkId) => {
    const componentId = resolveComponentIdForGlobalLink(projection, rootLinkId);
    if (!componentId) {
      throw new Error(
        `Cannot place assembled scene root "${rootLinkId}" because it has no projected owner`,
      );
    }

    const target = projection.componentRootTargets.get(componentId);
    if (!target) {
      throw new Error(
        `Cannot place assembled scene root "${rootLinkId}" because component "${componentId}" is missing`,
      );
    }

    const runtimeJointId = allocateSyntheticId(
      usedIds,
      `__workspace_component_root_joint__${componentId}`,
    );
    joints[runtimeJointId] = createSceneRootJoint(
      runtimeJointId,
      worldRootLinkId,
      rootLinkId,
      target.componentTransform,
    );
    componentTransformTargets.set(componentId, {
      kind: 'component-root',
      componentId,
      runtimeJointId,
    });
  });

  projection.componentRootTargets.forEach((_target, componentId) => {
    if (componentTransformTargets.has(componentId)) {
      return;
    }

    const incomingBridgeTargets = getVisibleIncomingBridges(workspace, componentId).flatMap(
      (bridge) => {
        const ref: BridgeEntityRef = { type: 'bridge', bridgeId: bridge.id };
        const runtimeJointId = projection.entityRefKeyToGlobal.get(entityRefKey(ref));
        return runtimeJointId && joints[runtimeJointId]
          ? [{ bridge, runtimeJointId }]
          : [];
      },
    );
    if (incomingBridgeTargets.length !== 1) {
      return;
    }
    const { bridge, runtimeJointId } = incomingBridgeTargets[0]!;
    componentTransformTargets.set(componentId, {
      kind: 'bridge',
      componentId,
      bridgeId: bridge.id,
      runtimeJointId,
    });
  });

  return {
    robotData: {
      ...robot,
      links,
      joints,
      rootLinkId: worldRootLinkId,
    },
    renderStrategy: projection.renderStrategy,
    assemblyTransform: cloneAssemblyTransform(workspace.transform),
    directComponentId: null,
    directComponentTransform: null,
    componentTransformTargets,
  };
}

export function createAssemblyScenePlacement(
  workspace: AssemblyState,
  projection: AssemblySceneProjection,
): AssemblyScenePlacement {
  if (projection.renderStrategy === 'assembled-scene') {
    return createAssembledScenePlacement(workspace, projection);
  }

  const [target] = projection.componentRootTargets.values();
  return {
    robotData: projection.robotData,
    renderStrategy: projection.renderStrategy,
    assemblyTransform: cloneAssemblyTransform(workspace.transform),
    directComponentId: target?.componentId ?? null,
    directComponentTransform: target
      ? cloneAssemblyTransform(target.componentTransform)
      : null,
    componentTransformTargets: new Map(),
  };
}
