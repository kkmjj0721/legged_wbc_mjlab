import * as THREE from 'three';

import { asRuntimeObject3D } from './runtimeRobotTypes';

export type RuntimeGeometrySubType = 'visual' | 'collision';

function normalizeLinkName(candidate: unknown): string | null {
  if (typeof candidate !== 'string') {
    return null;
  }

  const trimmed = candidate.trim();
  return trimmed.length > 0 ? trimmed : null;
}

export function getSyntheticGeomParentName(candidate: string | null | undefined): string | null {
  const trimmed = normalizeLinkName(candidate);
  if (!trimmed) {
    return null;
  }

  const match = trimmed.match(/^(.*)_geom_\d+$/);
  return match?.[1]?.trim() || null;
}

export function getGeometryObjectIndexUserDataKey(
  subType: RuntimeGeometrySubType,
): 'visualObjectIndex' | 'collisionObjectIndex' {
  return subType === 'collision' ? 'collisionObjectIndex' : 'visualObjectIndex';
}

function isTaggedCollisionObject(object: THREE.Object3D | null): boolean {
  const runtimeObject = object ? asRuntimeObject3D(object) : null;

  return Boolean(
    runtimeObject &&
    (runtimeObject.isURDFCollider ||
      runtimeObject.userData?.isCollisionGroup === true ||
      runtimeObject.userData?.isCollisionMesh === true ||
      runtimeObject.userData?.isCollision === true ||
      runtimeObject.userData?.geometryRole === 'collision'),
  );
}

function isTaggedVisualObject(object: THREE.Object3D | null): boolean {
  const runtimeObject = object ? asRuntimeObject3D(object) : null;

  return Boolean(
    runtimeObject &&
    (runtimeObject.isURDFVisual ||
      runtimeObject.userData?.isVisualGroup === true ||
      runtimeObject.userData?.isVisualMesh === true ||
      runtimeObject.userData?.isVisual === true ||
      runtimeObject.userData?.geometryRole === 'visual'),
  );
}

function matchesGeometryRole(
  object: THREE.Object3D | null,
  subType: RuntimeGeometrySubType,
): boolean {
  return subType === 'collision' ? isTaggedCollisionObject(object) : isTaggedVisualObject(object);
}

export function subtreeContainsGeometryRole(
  object: THREE.Object3D,
  subType: RuntimeGeometrySubType,
): boolean {
  if (matchesGeometryRole(object, subType)) {
    return true;
  }

  let found = false;
  object.traverse((child) => {
    if (found || child === object) {
      return;
    }

    if (matchesGeometryRole(child, subType)) {
      found = true;
    }
  });

  return found;
}

function readGeometryObjectIndex(
  object: THREE.Object3D | null,
  subType: RuntimeGeometrySubType,
): number | null {
  const key = getGeometryObjectIndexUserDataKey(subType);
  const value = object?.userData?.[key];
  const numeric = typeof value === 'number' ? value : Number(value);
  return Number.isInteger(numeric) && numeric >= 0 ? numeric : null;
}

export function resolveGeometryObjectIndexFromMetadata(
  object: THREE.Object3D | null,
  linkObject: THREE.Object3D,
  subType: RuntimeGeometrySubType,
): number | null {
  let current = object;

  while (current && current !== linkObject) {
    const indexedObject = readGeometryObjectIndex(current, subType);
    if (indexedObject !== null) {
      return indexedObject;
    }
    current = current.parent;
  }

  return null;
}

function rootContainsSelectionMetadata(
  root: THREE.Object3D,
  semanticLinkName: string,
  subType: RuntimeGeometrySubType,
  objectIndex: number | null,
): boolean {
  let matched = false;

  root.traverse((child) => {
    if (matched) {
      return;
    }

    if (!matchesGeometryRole(child, subType)) {
      return;
    }

    const childLinkName = normalizeLinkName(child.userData?.parentLinkName);
    if (childLinkName !== semanticLinkName) {
      return;
    }

    if (objectIndex === null) {
      matched = true;
      return;
    }

    matched = readGeometryObjectIndex(child, subType) === objectIndex;
  });

  return matched;
}

export function resolveRuntimeGeometryRoot(
  linkObject: THREE.Object3D,
  semanticLinkName: string,
  subType: RuntimeGeometrySubType,
  objectIndex = 0,
): THREE.Object3D | null {
  const directGeometryRoots = linkObject.children.filter((child) => {
    const runtimeChild = asRuntimeObject3D(child);

    if (runtimeChild.isURDFJoint || runtimeChild.isURDFLink) {
      return false;
    }

    return subtreeContainsGeometryRole(child, subType);
  });

  if (directGeometryRoots.length === 0) {
    return null;
  }

  const exactMatches = directGeometryRoots.filter((root) =>
    rootContainsSelectionMetadata(root, semanticLinkName, subType, objectIndex),
  );
  if (exactMatches.length > 0) {
    return exactMatches[0] ?? null;
  }

  const semanticMatches = directGeometryRoots.filter((root) =>
    rootContainsSelectionMetadata(root, semanticLinkName, subType, null),
  );
  if (semanticMatches.length > 0) {
    return semanticMatches[objectIndex] ?? semanticMatches[0] ?? null;
  }

  return directGeometryRoots[objectIndex] ?? directGeometryRoots[0] ?? null;
}
