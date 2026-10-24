import { use, useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';

import { GENERATED_OBJ_MATERIAL_USER_DATA_KEY } from '@/core/loaders/objModelData';
import { applyObjMaterialLibrariesToObject } from '@/core/loaders/objMaterialUtils';
import {
  createObjectFromSerializedObjDataAsync,
  loadSerializedObjModelData,
} from '@/core/loaders/objParseWorkerBridge';
import { createMainThreadYieldController } from '@/core/utils/yieldToMainThread';
import {
  addMainThreadMaterialPerformance,
  durationMs,
  readHighResolutionEpochMs,
} from '@/core/loaders/meshLoadPerformance';
import { applyVisualMeshShadowPolicyToObject } from '@/core/utils/visualMeshShadowPolicy';
import { disposeMaterial, disposeObject3D } from '@/shared/utils/three/dispose';

import { useLoadingManager } from '../meshLoadingManager';

interface ScaleProps {
  x: number;
  y: number;
  z: number;
}

interface OBJRendererImplProps {
  url: string;
  material: THREE.Material;
  color: string;
  enableShadows?: boolean;
  assets: Record<string, string>;
  assetBaseDir?: string;
  logicalAssetPath?: string;
  preserveOriginalMaterial?: boolean;
  scale?: ScaleProps;
  onResolved?: () => void;
}

export function shouldOverrideObjPreviewMesh(
  mesh: THREE.Mesh,
  preserveOriginalMaterial = false,
): boolean {
  if (preserveOriginalMaterial) {
    return false;
  }

  if (mesh.geometry?.getAttribute?.('color')) {
    return false;
  }

  const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  return materials.every((material) => (
    material?.userData?.[GENERATED_OBJ_MATERIAL_USER_DATA_KEY] === true
  ));
}

function cloneObjPreviewMaterial(material: THREE.Material): THREE.Material {
  const cloned = material.clone();

  (
    [
      'map',
      'alphaMap',
      'aoMap',
      'bumpMap',
      'displacementMap',
      'emissiveMap',
      'metalnessMap',
      'normalMap',
      'roughnessMap',
      'specularMap',
    ] as const
  ).forEach((propertyName) => {
    const texture = (cloned as THREE.Material & Record<string, unknown>)[propertyName];
    if (!(texture instanceof THREE.Texture)) {
      return;
    }

    const clonedTexture = texture.clone();
    clonedTexture.needsUpdate = true;
    (cloned as THREE.Material & Record<string, unknown>)[propertyName] = clonedTexture;
  });

  cloned.needsUpdate = true;
  return cloned;
}

function cloneObjPreviewObject(source: THREE.Object3D): THREE.Object3D {
  const clone = source.clone(true);

  clone.traverse((child) => {
    const renderable = child as THREE.Object3D & {
      material?: THREE.Material | THREE.Material[];
    };
    if (!renderable.material) {
      return;
    }

    if (Array.isArray(renderable.material)) {
      renderable.material = renderable.material.map((entry) => cloneObjPreviewMaterial(entry));
      return;
    }

    renderable.material = cloneObjPreviewMaterial(renderable.material);
  });

  return clone;
}

function enableVertexColorMaterials(mesh: THREE.Mesh): void {
  if (!mesh.geometry?.getAttribute?.('color')) {
    return;
  }

  const currentMaterials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  let didCloneMaterial = false;
  const nextMaterials = currentMaterials.map((entry) => {
    if (
      !entry ||
      !('vertexColors' in entry) ||
      (entry as THREE.MeshPhongMaterial).vertexColors === true
    ) {
      return entry;
    }

    const cloned = cloneObjPreviewMaterial(entry);
    (cloned as THREE.MeshPhongMaterial).vertexColors = true;
    cloned.needsUpdate = true;
    didCloneMaterial = true;
    return cloned;
  });

  if (!didCloneMaterial) {
    return;
  }

  mesh.material = Array.isArray(mesh.material) ? nextMaterials : nextMaterials[0];
}

function disposeGeneratedObjMaterials(
  materialOrMaterials: THREE.Material | THREE.Material[],
): void {
  const disposedMaterials = new Set<THREE.Material>();
  const materials = Array.isArray(materialOrMaterials)
    ? materialOrMaterials
    : [materialOrMaterials];

  materials.forEach((entry) => {
    if (!entry || disposedMaterials.has(entry)) {
      return;
    }

    disposedMaterials.add(entry);
    if (entry.userData?.[GENERATED_OBJ_MATERIAL_USER_DATA_KEY] === true) {
      disposeMaterial(entry, true);
    }
  });
}

export async function loadObjPreviewScene(
  assetUrl: string,
  manager: THREE.LoadingManager,
  sourcePath?: string | null,
): Promise<THREE.Group> {
  const requestUrl = manager.resolveURL(assetUrl);
  const serializedObject = await loadSerializedObjModelData(requestUrl);
  const yieldIfNeeded = createMainThreadYieldController();
  const object = await createObjectFromSerializedObjDataAsync(serializedObject, {
    yieldIfNeeded,
  });

  const materialStartedAt = readHighResolutionEpochMs();
  await applyObjMaterialLibrariesToObject(
    object,
    serializedObject.materialLibraries,
    manager,
    sourcePath,
    { yieldIfNeeded },
  );
  addMainThreadMaterialPerformance(serializedObject.loadPerformance, durationMs(materialStartedAt));
  if (serializedObject.loadPerformance) {
    object.userData = {
      ...(object.userData ?? {}),
      meshLoadPerformance: serializedObject.loadPerformance,
    };
  }

  return object;
}

export function replaceObjPreviewMeshMaterials(
  meshes: THREE.Mesh[],
  sharedMaterial: THREE.Material,
): void {
  meshes.forEach((mesh) => {
    const previousMaterial = mesh.material;
    if (previousMaterial === sharedMaterial) {
      return;
    }

    mesh.material = sharedMaterial;
    disposeGeneratedObjMaterials(previousMaterial);
  });
}

export function disposeObjPreviewClone(
  clone: THREE.Object3D,
  sharedMaterial: THREE.Material,
): void {
  disposeObject3D(clone, true, new Set([sharedMaterial]));
}

export function OBJRendererImpl({
  url,
  material,
  assets,
  assetBaseDir,
  logicalAssetPath,
  scale,
  enableShadows = true,
  preserveOriginalMaterial = false,
  onResolved,
}: OBJRendererImplProps) {
  const sharedMaterialRef = useRef(material);
  const manager = useLoadingManager(assets, assetBaseDir);
  const loadedObject = use(
    useMemo(
      () => loadObjPreviewScene(url, manager, logicalAssetPath),
      [logicalAssetPath, manager, url],
    ),
  );
  const { clone, overrideMeshes } = useMemo(() => {
    const nextClone = cloneObjPreviewObject(loadedObject);
    const meshes: THREE.Mesh[] = [];

    nextClone.traverse((child) => {
      if (!(child as THREE.Mesh).isMesh) {
        return;
      }

      const mesh = child as THREE.Mesh;
      enableVertexColorMaterials(mesh);
      if (shouldOverrideObjPreviewMesh(mesh, preserveOriginalMaterial)) {
        meshes.push(mesh);
      }
    });

    return { clone: nextClone, overrideMeshes: meshes };
  }, [loadedObject, preserveOriginalMaterial]);

  useLayoutEffect(() => {
    sharedMaterialRef.current = material;
    replaceObjPreviewMeshMaterials(overrideMeshes, material);
    if (enableShadows) {
      applyVisualMeshShadowPolicyToObject(clone);
    }
  }, [clone, enableShadows, material, overrideMeshes]);

  useEffect(() => {
    onResolved?.();
  }, [clone, onResolved]);

  useEffect(
    () => () => {
      disposeObjPreviewClone(clone, sharedMaterialRef.current);
    },
    [clone],
  );

  const scaleArr: [number, number, number] = scale ? [scale.x, scale.y, scale.z] : [1, 1, 1];

  return (
    <group rotation={[0, 0, 0]} scale={scaleArr}>
      <primitive object={clone} />
    </group>
  );
}

export default OBJRendererImpl;
