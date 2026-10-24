import type {
  AssemblyState,
  JointEntityRef,
  LinkEntityRef,
  RobotData,
  WorkspaceSelection,
} from '@/types';

import { createAssemblySceneProjection } from './assemblySceneProjection';

export type WorkspaceInspectableEntityRef = LinkEntityRef | JointEntityRef;

export interface WorkspaceRobotDataTarget {
  robotData: RobotData;
  scope: 'component' | 'assembly';
  componentId: string | null;
  resolveSnapshotEntityRef: (
    type: WorkspaceInspectableEntityRef['type'],
    snapshotEntityId: string,
  ) => WorkspaceInspectableEntityRef | null;
}

function getSelectedComponentId(selection: WorkspaceSelection): string | null {
  if (!selection) {
    return null;
  }

  switch (selection.entity.type) {
    case 'component':
    case 'link':
    case 'joint':
    case 'tendon':
      return selection.entity.componentId;
    case 'assembly':
    case 'bridge':
      return null;
  }
}

function createComponentTarget(
  workspace: AssemblyState,
  componentId: string,
): WorkspaceRobotDataTarget | null {
  const component = workspace.components[componentId];
  if (!component) {
    return null;
  }

  return {
    robotData: component.robot,
    scope: 'component',
    componentId,
    resolveSnapshotEntityRef(type, snapshotEntityId) {
      const exists = type === 'link'
        ? component.robot.links[snapshotEntityId] !== undefined
        : component.robot.joints[snapshotEntityId] !== undefined;
      return exists
        ? { type, componentId, entityId: snapshotEntityId }
        : null;
    },
  };
}

/** Resolve the immutable RobotData view appropriate for a canonical selection. */
export function resolveWorkspaceRobotDataTarget(
  workspace: AssemblyState,
  selection: WorkspaceSelection,
): WorkspaceRobotDataTarget {
  const selectedComponentId = getSelectedComponentId(selection);
  if (selectedComponentId) {
    const componentTarget = createComponentTarget(workspace, selectedComponentId);
    if (componentTarget) {
      return componentTarget;
    }
  }

  const components = Object.values(workspace.components);
  const requestsAssemblyScope =
    selection?.entity.type === 'assembly' || selection?.entity.type === 'bridge';
  if (!requestsAssemblyScope && components.length === 1 && components[0]) {
    return createComponentTarget(workspace, components[0].id)!;
  }

  const projection = createAssemblySceneProjection(workspace);
  return {
    robotData: projection.robotData,
    scope: 'assembly',
    componentId: null,
    resolveSnapshotEntityRef(type, snapshotEntityId) {
      const ref = projection.globalToEntityRef.get(snapshotEntityId);
      return ref?.type === type ? ref : null;
    },
  };
}
