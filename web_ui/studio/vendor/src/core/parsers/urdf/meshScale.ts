export interface MeshScaleLike {
  x?: number;
  y?: number;
  z?: number;
}

const DEFAULT_MESH_SCALE = 1;
const IDENTITY_MESH_SCALE = Object.freeze({ x: 1, y: 1, z: 1 });

function normalizeScaleComponent(value: number | undefined): number {
  return Number.isFinite(value) ? Number(value) : DEFAULT_MESH_SCALE;
}

export function normalizeMeshScale(scale?: MeshScaleLike): { x: number; y: number; z: number } {
  if (!scale) {
    return { ...IDENTITY_MESH_SCALE };
  }

  return {
    x: normalizeScaleComponent(scale.x),
    y: normalizeScaleComponent(scale.y),
    z: normalizeScaleComponent(scale.z),
  };
}

export function isIdentityMeshScale(scale?: MeshScaleLike): boolean {
  const normalized = normalizeMeshScale(scale);
  return normalized.x === 1 && normalized.y === 1 && normalized.z === 1;
}

export function formatUrdfMeshScaleAttribute(
  scale: MeshScaleLike | undefined,
  formatNumber: (value: number) => string,
): string {
  const normalized = normalizeMeshScale(scale);
  if (isIdentityMeshScale(normalized)) {
    return '';
  }

  return ` scale="${formatNumber(normalized.x)} ${formatNumber(normalized.y)} ${formatNumber(normalized.z)}"`;
}
