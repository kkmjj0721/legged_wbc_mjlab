import type * as THREE from 'three';

/**
 * Returns true if the material is protected from disposal or modification
 * because it is a shared singleton or collision overlay material.
 */
export function isProtectedMaterial(material: THREE.Material): boolean {
  return Boolean(
    (material as any).userData?.isSharedMaterial || (material as any).userData?.isCollisionMaterial,
  );
}

/**
 * Marks a material as a shared singleton that should not be disposed
 * or have its properties (opacity, etc.) modified by scene sync logic.
 */
export function markMaterialAsShared(material: THREE.Material): void {
  material.userData.isSharedMaterial = true;
}

/**
 * Marks a material as a collision overlay material.
 * Collision materials are protected from disposal and visual property overrides.
 */
export function markMaterialAsCollision(material: THREE.Material): void {
  material.userData.isCollisionMaterial = true;
}
