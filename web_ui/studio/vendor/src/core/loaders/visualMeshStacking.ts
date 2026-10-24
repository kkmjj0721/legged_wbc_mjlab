import * as THREE from 'three';

import { cloneMaterialWithCoplanarOffset } from './coplanarMaterialOffset';

const POSITION_TOLERANCE = 1e-4;
const QUATERNION_TOLERANCE = 1e-4;
const BOUNDS_CENTER_TOLERANCE = 1e-4;
const BOUNDS_MIN_SIZE = 1e-5;
const BOUNDS_MIN_OVERLAP_RATIO = 0.5;
const BOUNDS_MAX_CENTER_SHIFT_RATIO = 0.25;
const BOUNDS_MAX_AXIS_SIZE_RATIO = 1.2;
const BOUNDS_MAX_VOLUME_RATIO = 1.5;

export interface CoincidentVisualRootEntry {
  root: THREE.Object3D;
  stableId: string | number;
}

export interface StackCoincidentVisualRootsOptions {
  space?: 'local' | 'world';
}

interface VisualRootDescriptor {
  entry: CoincidentVisualRootEntry;
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  bounds: THREE.Box3 | null;
  boundsCenter: THREE.Vector3 | null;
  boundsSize: THREE.Vector3 | null;
}

function buildWorldBounds(root: THREE.Object3D): THREE.Box3 | null {
  let bounds: THREE.Box3 | null = null;
  root.updateWorldMatrix(true, true);

  root.traverse((child: any) => {
    if (!child?.isMesh) {
      return;
    }

    const geometry = child.geometry;
    if (!(geometry instanceof THREE.BufferGeometry)) {
      return;
    }

    geometry.computeBoundingBox?.();
    if (!geometry.boundingBox) {
      return;
    }

    const childBounds = geometry.boundingBox.clone().applyMatrix4(child.matrixWorld);
    if (bounds) {
      bounds.union(childBounds);
      return;
    }

    bounds = childBounds;
  });

  return bounds;
}

function buildVisualRootDescriptor(
  entry: CoincidentVisualRootEntry,
  space: 'local' | 'world' = 'local',
): VisualRootDescriptor {
  const position = space === 'world'
    ? entry.root.getWorldPosition(new THREE.Vector3())
    : entry.root.position.clone();
  const quaternion = space === 'world'
    ? entry.root.getWorldQuaternion(new THREE.Quaternion())
    : entry.root.quaternion.clone();
  const bounds = buildWorldBounds(entry.root);
  const boundsCenter = bounds ? bounds.getCenter(new THREE.Vector3()) : null;
  const boundsSize = bounds ? bounds.getSize(new THREE.Vector3()) : null;

  return {
    entry,
    position,
    quaternion,
    bounds,
    boundsCenter,
    boundsSize,
  };
}

function transformsMatchWithinTolerance(
  left: VisualRootDescriptor,
  right: VisualRootDescriptor,
): boolean {
  const leftPosition = left.position;
  const rightPosition = right.position;

  for (const axis of ['x', 'y', 'z'] as const) {
    if (Math.abs((leftPosition[axis] ?? 0) - (rightPosition[axis] ?? 0)) > POSITION_TOLERANCE) {
      return false;
    }
  }

  const leftQuaternion = left.quaternion;
  const rightQuaternion = right.quaternion;

  let directDelta = 0;
  let negatedDelta = 0;
  for (const axis of ['x', 'y', 'z', 'w'] as const) {
    directDelta = Math.max(directDelta, Math.abs((leftQuaternion[axis] ?? 0) - (rightQuaternion[axis] ?? 0)));
    negatedDelta = Math.max(negatedDelta, Math.abs((leftQuaternion[axis] ?? 0) + (rightQuaternion[axis] ?? 0)));
  }

  return Math.min(directDelta, negatedDelta) <= QUATERNION_TOLERANCE;
}

function boundsMatchWithinTolerance(
  left: VisualRootDescriptor,
  right: VisualRootDescriptor,
): boolean {
  if (!left.bounds || !right.bounds || !left.boundsCenter || !right.boundsCenter || !left.boundsSize || !right.boundsSize) {
    return true;
  }

  let leftVolume = 1;
  let rightVolume = 1;
  for (const axis of ['x', 'y', 'z'] as const) {
    const leftSize = left.boundsSize[axis] ?? 0;
    const rightSize = right.boundsSize[axis] ?? 0;
    const minSize = Math.min(leftSize, rightSize);
    const maxSize = Math.max(leftSize, rightSize);
    leftVolume *= Math.max(leftSize, BOUNDS_MIN_SIZE);
    rightVolume *= Math.max(rightSize, BOUNDS_MIN_SIZE);
    const centerDelta = Math.abs((left.boundsCenter[axis] ?? 0) - (right.boundsCenter[axis] ?? 0));
    const centerTolerance = Math.max(BOUNDS_CENTER_TOLERANCE, minSize * BOUNDS_MAX_CENTER_SHIFT_RATIO);
    if (centerDelta > centerTolerance) {
      return false;
    }

    if (maxSize <= BOUNDS_MIN_SIZE) {
      continue;
    }

    const overlap = Math.min(left.bounds.max[axis], right.bounds.max[axis])
      - Math.max(left.bounds.min[axis], right.bounds.min[axis]);
    if (overlap < -BOUNDS_CENTER_TOLERANCE) {
      return false;
    }

    if (minSize > BOUNDS_MIN_SIZE && (overlap / minSize) < BOUNDS_MIN_OVERLAP_RATIO) {
      return false;
    }

    if (minSize > BOUNDS_MIN_SIZE && (maxSize / minSize) > BOUNDS_MAX_AXIS_SIZE_RATIO) {
      return false;
    }
  }

  const minVolume = Math.min(leftVolume, rightVolume);
  const maxVolume = Math.max(leftVolume, rightVolume);
  if (minVolume > BOUNDS_MIN_SIZE && (maxVolume / minVolume) > BOUNDS_MAX_VOLUME_RATIO) {
    return false;
  }

  return true;
}

function visualRootsMatchWithinTolerance(
  left: VisualRootDescriptor,
  right: VisualRootDescriptor,
): boolean {
  return transformsMatchWithinTolerance(left, right)
    && boundsMatchWithinTolerance(left, right);
}

function applyVisualStackIndex(root: THREE.Object3D, stackIndex: number): void {
  root.userData = {
    ...(root.userData ?? {}),
    visualStackIndex: stackIndex,
  };

  if (stackIndex <= 0) {
    return;
  }

  root.traverse((child: any) => {
    if (!child?.isMesh) {
      return;
    }

    child.renderOrder = Math.max(Number(child.renderOrder) || 0, stackIndex);
    child.userData = {
      ...(child.userData ?? {}),
      visualStackIndex: stackIndex,
    };

    if (Array.isArray(child.material)) {
      child.material = child.material.map((material: THREE.Material) =>
        cloneMaterialWithCoplanarOffset(material, stackIndex),
      );
      return;
    }

    if (child.material) {
      child.material = cloneMaterialWithCoplanarOffset(child.material as THREE.Material, stackIndex);
    }
  });
}

export function stackCoincidentVisualRoots(
  entries: CoincidentVisualRootEntry[],
  options: StackCoincidentVisualRootsOptions = {},
): void {
  const { space = 'local' } = options;
  const descriptors = entries.map((entry) => buildVisualRootDescriptor(entry, space));
  const groups: VisualRootDescriptor[][] = [];

  descriptors.forEach((descriptor) => {
    const existing = groups.find((group) => visualRootsMatchWithinTolerance(group[0], descriptor));
    if (existing) {
      existing.push(descriptor);
      return;
    }

    groups.push([descriptor]);
  });

  groups.forEach((group) => {
    if (group.length < 2) {
      applyVisualStackIndex(group[0].entry.root, 0);
      return;
    }

    group
      .slice()
      .sort((left, right) => String(left.entry.stableId).localeCompare(String(right.entry.stableId)))
      .forEach((descriptor, index) => {
        applyVisualStackIndex(descriptor.entry.root, index);
      });
  });
}
