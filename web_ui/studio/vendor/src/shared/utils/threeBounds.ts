import * as THREE from 'three';

import { asRuntimeObject3D } from '@/shared/components/3d/runtimeRobotTypes';

interface LowestMeshZOptions {
  includeInvisible?: boolean;
  includeVisual?: boolean;
  includeCollision?: boolean;
}

interface VisibleMeshBoundsOptions {
  includeInvisible?: boolean;
  includeGroundPlaneHelpers?: boolean;
}

type MeshRole = 'visual' | 'collision' | 'unknown';
const GROUND_PLANE_BOUND_OBJECT_NAMES = new Set(['ReferenceGrid', 'GroundShadowPlane']);

function refreshWorldMatrices(root: THREE.Object3D): void {
  // Nested component pivots are often measured directly instead of from the scene
  // root. Refresh ancestors first so world-space bounds are based on current parent
  // transforms rather than stale matrixWorld values.
  root.updateWorldMatrix(true, true);
}

function isRuntimeUrdfNode(object: THREE.Object3D | null): boolean {
  const runtimeObject = object ? asRuntimeObject3D(object) : null;

  return Boolean(runtimeObject?.isURDFJoint || runtimeObject?.isURDFLink);
}

function shouldIgnoreBoundsAncestor(
  object: THREE.Object3D | null,
  options?: { includeGroundPlaneHelpers?: boolean },
): boolean {
  let current: THREE.Object3D | null = object;

  while (current) {
    const isGroundPlaneHelper = Boolean(
      options?.includeGroundPlaneHelpers && GROUND_PLANE_BOUND_OBJECT_NAMES.has(current.name),
    );

    if (
      current.userData?.isGizmo ||
      (current.name?.startsWith('__') && !isRuntimeUrdfNode(current)) ||
      ((current.userData?.isHelper || current.userData?.excludeFromSceneBounds) &&
        !isGroundPlaneHelper)
    ) {
      return true;
    }

    current = current.parent;
  }

  return false;
}

function getMeshRole(mesh: THREE.Mesh): MeshRole {
  let current: THREE.Object3D | null = mesh;

  while (current) {
    const runtimeCurrent = asRuntimeObject3D(current);

    if (
      runtimeCurrent.isURDFCollider ||
      current.userData?.isCollisionMesh === true ||
      current.userData?.geometryRole === 'collision'
    ) {
      return 'collision';
    }

    if (current.userData?.isVisualMesh === true || current.userData?.geometryRole === 'visual') {
      return 'visual';
    }

    current = current.parent;
  }

  return 'unknown';
}

/**
 * Compute the lowest world-space Z from mesh bounding boxes.
 * Invalid/NaN bounding boxes are skipped to avoid poisoning the result.
 */
export function getLowestMeshZ(root: THREE.Object3D, options?: LowestMeshZOptions): number | null {
  const includeInvisible = options?.includeInvisible ?? true;
  const includeVisual = options?.includeVisual ?? true;
  const includeCollision = options?.includeCollision ?? true;
  const worldBox = new THREE.Box3();
  let lowestZ = Number.POSITIVE_INFINITY;

  refreshWorldMatrices(root);

  const visitNode = (obj: THREE.Object3D) => {
    if (shouldIgnoreBoundsAncestor(obj)) return;
    if (!(obj as THREE.Mesh).isMesh) return;

    const mesh = obj as THREE.Mesh;
    if (!mesh.geometry) return;

    const meshRole = getMeshRole(mesh);
    if (meshRole === 'collision' && !includeCollision) return;
    if ((meshRole === 'visual' || meshRole === 'unknown') && !includeVisual) return;

    if (!mesh.geometry.boundingBox) {
      mesh.geometry.computeBoundingBox();
    }

    const localBox = mesh.geometry.boundingBox;
    if (!localBox) return;

    worldBox.copy(localBox).applyMatrix4(mesh.matrixWorld);

    if (!Number.isFinite(worldBox.min.z) || !Number.isFinite(worldBox.max.z)) {
      return;
    }

    lowestZ = Math.min(lowestZ, worldBox.min.z);
  };

  // NOTE:
  // `obj.visible` only reflects local visibility. Using `traverseVisible` ensures
  // meshes under hidden parents (e.g. hidden collision groups) are excluded.
  if (includeInvisible) {
    root.traverse(visitNode);
  } else {
    root.traverseVisible(visitNode);
  }

  return Number.isFinite(lowestZ) ? lowestZ : null;
}

/**
 * Move an object's local Z so its lowest rendered mesh point sits on a target world-space Z plane.
 *
 * Notes:
 * - This mutates `root.position`.
 * - The translation is resolved in parent-local space so rotated/scaled ancestors still move the
 *   child by the requested world-space Z delta.
 */
export function alignObjectLowestPointToZ(
  root: THREE.Object3D,
  targetZ = 0,
  options?: LowestMeshZOptions,
): number | null {
  const minZ = getLowestMeshZ(root, options);
  if (minZ === null) {
    return null;
  }

  const deltaWorldZ = targetZ - minZ;
  if (deltaWorldZ === 0) {
    refreshWorldMatrices(root);
    return targetZ;
  }

  if (root.parent) {
    const localOrigin = root.parent.worldToLocal(new THREE.Vector3(0, 0, 0));
    const localTarget = root.parent.worldToLocal(new THREE.Vector3(0, 0, deltaWorldZ));
    root.position.add(localTarget.sub(localOrigin));
  } else {
    root.position.z += deltaWorldZ;
  }

  refreshWorldMatrices(root);
  return targetZ;
}

/**
 * Compute the visible mesh bounds for a scene subtree while skipping helper,
 * gizmo, and infrastructure meshes that should not influence camera clipping.
 */
export function computeVisibleMeshBounds(
  root: THREE.Object3D,
  options?: VisibleMeshBoundsOptions,
): THREE.Box3 | null {
  const includeInvisible = options?.includeInvisible ?? false;
  const includeGroundPlaneHelpers = options?.includeGroundPlaneHelpers ?? false;
  const bounds = new THREE.Box3();
  const worldBox = new THREE.Box3();
  let hasBounds = false;

  refreshWorldMatrices(root);

  const visitNode = (obj: THREE.Object3D) => {
    if (shouldIgnoreBoundsAncestor(obj, { includeGroundPlaneHelpers })) {
      return;
    }

    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) {
      return;
    }

    if (!mesh.geometry.boundingBox) {
      mesh.geometry.computeBoundingBox();
    }

    const localBox = mesh.geometry.boundingBox;
    if (!localBox) {
      return;
    }

    worldBox.copy(localBox).applyMatrix4(mesh.matrixWorld);
    if (
      !Number.isFinite(worldBox.min.x) ||
      !Number.isFinite(worldBox.min.y) ||
      !Number.isFinite(worldBox.min.z) ||
      !Number.isFinite(worldBox.max.x) ||
      !Number.isFinite(worldBox.max.y) ||
      !Number.isFinite(worldBox.max.z)
    ) {
      return;
    }

    if (!hasBounds) {
      bounds.copy(worldBox);
      hasBounds = true;
      return;
    }

    bounds.union(worldBox);
  };

  if (includeInvisible) {
    root.traverse(visitNode);
  } else {
    root.traverseVisible(visitNode);
  }

  return hasBounds ? bounds : null;
}
