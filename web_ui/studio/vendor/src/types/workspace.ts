import type { AssemblyState } from './robot';
import type { InteractionHelperKind } from './ui';

export interface AssemblyEntityRef {
  readonly type: 'assembly';
}

export interface ComponentEntityRef {
  readonly type: 'component';
  readonly componentId: string;
}

export interface BridgeEntityRef {
  readonly type: 'bridge';
  readonly bridgeId: string;
}

export interface LinkEntityRef {
  readonly type: 'link';
  readonly componentId: string;
  readonly entityId: string;
}

export interface JointEntityRef {
  readonly type: 'joint';
  readonly componentId: string;
  readonly entityId: string;
}

export interface TendonEntityRef {
  readonly type: 'tendon';
  readonly componentId: string;
  readonly entityId: string;
}

export type ComponentRobotEntityRef = LinkEntityRef | JointEntityRef | TendonEntityRef;

/** Canonical workspace identity. Entity ownership is always explicit. */
export type EntityRef =
  | AssemblyEntityRef
  | ComponentEntityRef
  | BridgeEntityRef
  | ComponentRobotEntityRef;

/**
 * Canonical workspace selection. `null` is the only empty-selection value, so
 * renderer details cannot survive after their owning entity is cleared.
 */
export type WorkspaceSelection =
  | {
      readonly entity: EntityRef;
      readonly subType?: 'visual' | 'collision';
      readonly objectIndex?: number;
      readonly helperKind?: InteractionHelperKind;
      readonly highlightObjectId?: number;
    }
  | null;

/** Compare canonical entity identities without parsing or constructing global IDs. */
export function areEntityRefsEqual(
  left: EntityRef | null | undefined,
  right: EntityRef | null | undefined,
): boolean {
  if (left === right) {
    return true;
  }
  if (!left || !right || left.type !== right.type) {
    return false;
  }

  switch (left.type) {
    case 'assembly':
      return right.type === 'assembly';
    case 'component':
      return right.type === 'component' && left.componentId === right.componentId;
    case 'bridge':
      return right.type === 'bridge' && left.bridgeId === right.bridgeId;
    case 'link':
    case 'joint':
    case 'tendon':
      return (
        right.type === left.type &&
        left.componentId === right.componentId &&
        left.entityId === right.entityId
      );
  }
}

/**
 * Stable, collision-free identity for maps keyed by EntityRef. JSON tuple
 * encoding remains unambiguous even when IDs contain separators.
 */
export function entityRefKey(ref: EntityRef): string {
  switch (ref.type) {
    case 'assembly':
      return JSON.stringify(['assembly']);
    case 'component':
      return JSON.stringify(['component', ref.componentId]);
    case 'bridge':
      return JSON.stringify(['bridge', ref.bridgeId]);
    case 'link':
    case 'joint':
    case 'tendon':
      return JSON.stringify([ref.type, ref.componentId, ref.entityId]);
  }
}

export interface WorkspaceActivityEntry {
  id: string;
  timestamp: string;
  label: string;
}

/** Serializable canonical history shared by the workspace store and archive. */
export interface WorkspaceHistory {
  past: AssemblyState[];
  future: AssemblyState[];
  activity: WorkspaceActivityEntry[];
}
