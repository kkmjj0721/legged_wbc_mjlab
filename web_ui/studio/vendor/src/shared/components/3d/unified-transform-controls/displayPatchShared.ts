import * as THREE from 'three';
import {
  AXIS_NAMES,
  DISPLAY_BEHAVIOR_PATCH_VERSION,
  GIZMO_BASE_RENDER_ORDER,
} from './gizmoCore';

export const removeHandlesByNames = (group: THREE.Object3D | undefined, names: Set<string>) => {
  if (!group) return;

  const nodesToRemove: THREE.Object3D[] = [];
  const removedNodes = new Set<THREE.Object3D>();
  group.traverse((node) => {
    if (node === group) return;
    if (!names.has(node.name)) return;
    if ((node as any).isLine || (node as any).isMesh || (node as any).isObject3D) {
      nodesToRemove.push(node);
      removedNodes.add(node);
    }
  });

  const survivingGeometries = new Set<THREE.BufferGeometry>();
  const survivingMaterials = new Set<THREE.Material>();
  group.traverse((node) => {
    if (removedNodes.has(node)) return;
    const geometry = (node as THREE.Mesh).geometry as THREE.BufferGeometry | undefined;
    const material = (node as THREE.Mesh & { material?: THREE.Material | THREE.Material[] }).material;
    if (geometry) survivingGeometries.add(geometry);
    if (Array.isArray(material)) {
      material.forEach((entry) => {
        if (entry) survivingMaterials.add(entry);
      });
    } else if (material) {
      survivingMaterials.add(material);
    }
  });

  for (const node of nodesToRemove) {
    node.parent?.remove(node);
    node.traverse((entry) => {
      const geometry = (entry as THREE.Mesh).geometry as THREE.BufferGeometry | undefined;
      const material = (entry as THREE.Mesh & { material?: THREE.Material | THREE.Material[] }).material;
      if (geometry && !survivingGeometries.has(geometry)) {
        geometry.dispose();
      }
      if (Array.isArray(material)) {
        material.forEach((item) => {
          if (item && !survivingMaterials.has(item)) item.dispose();
        });
      } else if (material && !survivingMaterials.has(material)) {
        material.dispose();
      }
    });
  }
};

const disposeObjectResources = (object: THREE.Object3D) => {
  object.traverse((node) => {
    const geometry = (node as THREE.Mesh).geometry as THREE.BufferGeometry | undefined;
    const material = (node as THREE.Mesh & { material?: THREE.Material | THREE.Material[] }).material;

    geometry?.dispose?.();

    if (Array.isArray(material)) {
      material.forEach((entry) => entry?.dispose?.());
      return;
    }

    material?.dispose?.();
  });
};

export const removeGeneratedHandles = (
  group: THREE.Object3D | undefined,
  predicate: (node: THREE.Object3D) => boolean
) => {
  if (!group) return;

  const nodesToRemove: THREE.Object3D[] = [];
  group.traverse((node) => {
    if (node === group) return;
    if (!predicate(node)) return;
    nodesToRemove.push(node);
  });

  for (const node of nodesToRemove) {
    node.parent?.remove(node);
    disposeObjectResources(node);
  }
};

export const getDisplayBehaviorPatchKey = (thicknessScale: number) =>
  `${DISPLAY_BEHAVIOR_PATCH_VERSION}:${thicknessScale.toFixed(3)}`;

export const getAxisComponentKey = (axisName: 'X' | 'Y' | 'Z') =>
  axisName.toLowerCase() as 'x' | 'y' | 'z';

export const cloneAxisColorMaterial = (sourceMaterial: THREE.Material | null) => {
  const color = new THREE.Color(0xffffff);
  if (sourceMaterial && 'color' in sourceMaterial && sourceMaterial.color instanceof THREE.Color) {
    color.copy(sourceMaterial.color);
  }

  const material = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 1,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });

  material.userData = {
    ...material.userData,
    urdfBaseColor: color.clone(),
  };

  return material;
};

export const hideStockAxisLines = (group: THREE.Object3D | undefined) => {
  if (!group) return;

  group.traverse((node) => {
    const line = node as THREE.Line;
    if (!line.isLine) return;
    if (!AXIS_NAMES.has(line.name)) return;

    line.userData.urdfHideStockAxisLine = true;
    line.visible = false;
  });
};

export const replaceMeshGeometry = (mesh: THREE.Mesh, nextGeometry: THREE.BufferGeometry) => {
  const previousGeometry = mesh.geometry as THREE.BufferGeometry | undefined;
  mesh.geometry = nextGeometry;
  previousGeometry?.dispose?.();
};

/** Ensure every mesh in gizmo groups renders above scene geometry (e.g. grid). */
export const enforceGizmoRenderPriority = (gizmo: any) => {
  const groups = [
    gizmo?.gizmo?.translate,
    gizmo?.gizmo?.rotate,
    gizmo?.picker?.translate,
    gizmo?.picker?.rotate,
    gizmo?.helper?.translate,
    gizmo?.helper?.rotate,
  ];

  for (const group of groups) {
    if (!group) continue;
    (group as THREE.Object3D).traverse((node: THREE.Object3D) => {
      const preferredRenderOrder =
        typeof node.userData?.urdfRenderOrder === 'number'
          ? node.userData.urdfRenderOrder
          : (Number.isFinite(node.renderOrder) && node.renderOrder > 0
              ? node.renderOrder
              : GIZMO_BASE_RENDER_ORDER);

      node.renderOrder = preferredRenderOrder;
      if (typeof node.userData?.urdfRenderOrder !== 'number') {
        node.userData = {
          ...node.userData,
          urdfRenderOrder: preferredRenderOrder,
        };
      }

      const material = (node as THREE.Mesh & { material?: THREE.Material | THREE.Material[] }).material;
      if (!material) return;
      const materials = Array.isArray(material) ? material : [material];
      for (const mat of materials) {
        mat.depthTest = false;
        mat.depthWrite = false;
      }
    });
  }
};
