import * as THREE from 'three';

const WORLD_UP = new THREE.Vector3(0, 0, 1);
const LOCAL_AXES = [
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(0, 1, 0),
  new THREE.Vector3(0, 0, 1),
] as const;

const pooledBoundsSize = new THREE.Vector3();
const pooledWorldPosition = new THREE.Vector3();
const pooledWorldQuaternion = new THREE.Quaternion();
const pooledWorldScale = new THREE.Vector3();
const pooledThinAxisWorld = new THREE.Vector3();

const HOVER_SUPPORT_SURFACE_NAME_RE = /\b(floor|ground|groundplane|plane|terrain|stage)\b/i;
const HOVER_SUPPORT_SURFACE_UP_ALIGNMENT = 0.82;
const HOVER_SUPPORT_SURFACE_MIN_FOOTPRINT = 0.14;
const HOVER_SUPPORT_SURFACE_MIN_FLATNESS = 4;
const HOVER_SUPPORT_SURFACE_MIN_NAMED_FOOTPRINT = 0.04;
const HOVER_SUPPORT_SURFACE_MAX_THICKNESS = 0.2;

function getAncestorNameHint(object: THREE.Object3D | null): boolean {
  let current: THREE.Object3D | null = object;

  while (current) {
    if (HOVER_SUPPORT_SURFACE_NAME_RE.test(String(current.name || ''))) {
      return true;
    }
    current = current.parent;
  }

  return false;
}

function isPlaneGeometryType(geometry: THREE.BufferGeometry | null | undefined): boolean {
  const geometryType = String(geometry?.type || '');
  return geometryType === 'PlaneGeometry' || geometryType === 'PlaneBufferGeometry';
}

function getThinAxisIndex(size: THREE.Vector3): 0 | 1 | 2 {
  if (size.x <= size.y && size.x <= size.z) {
    return 0;
  }

  if (size.y <= size.x && size.y <= size.z) {
    return 1;
  }

  return 2;
}

function isHorizontalSupportSurface(mesh: THREE.Mesh): boolean {
  const geometry = mesh.geometry;
  if (!geometry) {
    return false;
  }

  if (!geometry.boundingBox) {
    geometry.computeBoundingBox();
  }
  if (!geometry.boundingBox) {
    return false;
  }

  geometry.boundingBox.getSize(pooledBoundsSize);
  mesh.matrixWorld.decompose(
    pooledWorldPosition,
    pooledWorldQuaternion,
    pooledWorldScale,
  );

  const scaledSize = [
    Math.abs(pooledBoundsSize.x * pooledWorldScale.x),
    Math.abs(pooledBoundsSize.y * pooledWorldScale.y),
    Math.abs(pooledBoundsSize.z * pooledWorldScale.z),
  ] as const;
  const thinAxisIndex = getThinAxisIndex(new THREE.Vector3(...scaledSize));
  const thinAxisThickness = scaledSize[thinAxisIndex];
  const footprintAxes = scaledSize.filter((_, index) => index !== thinAxisIndex);
  const [footprintA = 0, footprintB = 0] = footprintAxes;
  const footprint = Math.max(footprintA, footprintB);
  const footprintArea = footprintA * footprintB;
  const namedSurface = getAncestorNameHint(mesh);

  pooledThinAxisWorld.copy(LOCAL_AXES[thinAxisIndex]).applyQuaternion(pooledWorldQuaternion).normalize();
  const upAlignment = Math.abs(pooledThinAxisWorld.dot(WORLD_UP));
  if (upAlignment < HOVER_SUPPORT_SURFACE_UP_ALIGNMENT) {
    return false;
  }

  if (isPlaneGeometryType(geometry)) {
    return namedSurface || footprintArea >= HOVER_SUPPORT_SURFACE_MIN_FOOTPRINT;
  }

  if (thinAxisThickness > HOVER_SUPPORT_SURFACE_MAX_THICKNESS) {
    return false;
  }

  const flatness = footprint / Math.max(thinAxisThickness, 1e-3);
  if (namedSurface) {
    return footprintArea >= HOVER_SUPPORT_SURFACE_MIN_NAMED_FOOTPRINT
      && flatness >= HOVER_SUPPORT_SURFACE_MIN_FLATNESS;
  }

  return footprintArea >= HOVER_SUPPORT_SURFACE_MIN_FOOTPRINT
    && flatness >= HOVER_SUPPORT_SURFACE_MIN_FLATNESS * 2;
}

export function isHoverSupportSurface(object: THREE.Object3D | null): boolean {
  let current: THREE.Object3D | null = object;

  while (current) {
    if (current.userData?.hoverSupportSurface === true) {
      return true;
    }

    if ((current as THREE.Mesh).isMesh && isHorizontalSupportSurface(current as THREE.Mesh)) {
      return true;
    }

    current = current.parent;
  }

  return false;
}
