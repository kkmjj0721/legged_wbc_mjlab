import type { AssemblyTransform } from '@/types';

const TRANSFORM_EPSILON = 1e-9;

export const IDENTITY_ASSEMBLY_TRANSFORM: AssemblyTransform = Object.freeze({
  position: Object.freeze({ x: 0, y: 0, z: 0 }),
  rotation: Object.freeze({ r: 0, p: 0, y: 0 }),
});

export function cloneAssemblyTransform(transform?: AssemblyTransform | null): AssemblyTransform {
  if (!transform) {
    return {
      position: { ...IDENTITY_ASSEMBLY_TRANSFORM.position },
      rotation: { ...IDENTITY_ASSEMBLY_TRANSFORM.rotation },
    };
  }

  return {
    position: {
      x: Number.isFinite(transform.position?.x) ? transform.position.x : 0,
      y: Number.isFinite(transform.position?.y) ? transform.position.y : 0,
      z: Number.isFinite(transform.position?.z) ? transform.position.z : 0,
    },
    rotation: {
      r: Number.isFinite(transform.rotation?.r) ? transform.rotation.r : 0,
      p: Number.isFinite(transform.rotation?.p) ? transform.rotation.p : 0,
      y: Number.isFinite(transform.rotation?.y) ? transform.rotation.y : 0,
    },
  };
}

export function isIdentityAssemblyTransform(transform?: AssemblyTransform | null): boolean {
  if (!transform) {
    return true;
  }

  const normalized = cloneAssemblyTransform(transform);
  return (
    Math.abs(normalized.position.x) <= TRANSFORM_EPSILON &&
    Math.abs(normalized.position.y) <= TRANSFORM_EPSILON &&
    Math.abs(normalized.position.z) <= TRANSFORM_EPSILON &&
    Math.abs(normalized.rotation.r) <= TRANSFORM_EPSILON &&
    Math.abs(normalized.rotation.p) <= TRANSFORM_EPSILON &&
    Math.abs(normalized.rotation.y) <= TRANSFORM_EPSILON
  );
}
