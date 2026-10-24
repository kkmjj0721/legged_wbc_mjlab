import type * as THREE from 'three';

import {
  cloneMaterialWithCoplanarOffset as cloneMaterialWithCoplanarOffsetImpl,
  isCoplanarOffsetMaterial as isCoplanarOffsetMaterialImpl,
  markMaterialAsCoplanarOffset as markMaterialAsCoplanarOffsetImpl,
  mitigateCoplanarMaterialZFighting as mitigateCoplanarMaterialZFightingImpl,
} from './coplanarMaterialOffset.shared.js';

export type CoplanarGroupStackAssignment = {
  groupIndex: number;
  materialIndex: number;
  stackIndex: number;
};

export type CoplanarMaterialAnalysis = {
  adjustedMaterialIndices: number[];
  groupStackAssignments: CoplanarGroupStackAssignment[];
  duplicateTriangleCount: number;
  nearCoplanarTriangleCount: number;
};

export type CoplanarMaterialOffsetResult = CoplanarMaterialAnalysis & {
  adjustedMaterialCount: number;
};

export const cloneMaterialWithCoplanarOffset:
  <T extends THREE.Material>(material: T, stackIndex?: number) => T =
    cloneMaterialWithCoplanarOffsetImpl;

export const mitigateCoplanarMaterialZFighting:
  (mesh: THREE.Mesh) => CoplanarMaterialOffsetResult =
    mitigateCoplanarMaterialZFightingImpl;

export const isCoplanarOffsetMaterial:
  (material: THREE.Material | null | undefined) => boolean =
    isCoplanarOffsetMaterialImpl;

export const markMaterialAsCoplanarOffset:
  <T extends THREE.Material>(material: T) => T =
    markMaterialAsCoplanarOffsetImpl;
