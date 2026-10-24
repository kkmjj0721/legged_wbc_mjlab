import * as THREE from 'three';

import { disposeMaterial } from '@/shared/utils/three/dispose';

import { SHARED_MATERIALS } from './sharedMaterials';

export function disposeReplacedMaterials(
  material: THREE.Material | THREE.Material[] | undefined,
  disposedMaterials: Set<THREE.Material>,
  disposeTextures: boolean,
): void {
  if (!material) return;

  const mats = Array.isArray(material) ? material : [material];
  for (const mat of mats) {
    if (!mat || disposedMaterials.has(mat) || SHARED_MATERIALS.has(mat)) continue;
    disposeMaterial(mat, disposeTextures, SHARED_MATERIALS);
    disposedMaterials.add(mat);
  }
}
