import { use, useEffect, useLayoutEffect, useMemo } from 'react';
import * as THREE from 'three';
import { useLoadingManager } from '../meshLoadingManager';
import { cloneColladaScenePreservingRootTransform } from './colladaScene';
import { isCoplanarOffsetMaterial, markMaterialAsCoplanarOffset } from '@/core/loaders';
import { loadColladaScene } from '@/core/loaders/colladaParseWorkerBridge';
import { applyVisualMeshShadowPolicyToObject } from '@/core/utils/visualMeshShadowPolicy';
import { disposeObject3DGraph } from '@/shared/utils/three/dispose';

interface ScaleProps {
  x: number;
  y: number;
  z: number;
}

interface DAERendererImplProps {
  url: string;
  material: THREE.Material;
  enableShadows?: boolean;
  assets: Record<string, string>;
  assetBaseDir?: string;
  normalizeRoot?: boolean;
  preserveOriginalMaterial?: boolean;
  scale?: ScaleProps;
  onResolved?: () => void;
}

const ORIGINAL_MATERIAL_KEY = '__urdfStudioDaeOriginalMaterial';
const GENERATED_OVERRIDE_MATERIAL_KEY = '__urdfStudioDaeGeneratedOverrideMaterial';

function disposeGeneratedOverrideMaterials(materialOrMaterials: THREE.Material | THREE.Material[]) {
  const materials = Array.isArray(materialOrMaterials)
    ? materialOrMaterials
    : [materialOrMaterials];
  materials.forEach((nextMaterial) => {
    if (nextMaterial?.userData?.[GENERATED_OVERRIDE_MATERIAL_KEY] === true) {
      nextMaterial.dispose();
    }
  });
}

function buildOverrideMaterial(
  sourceMaterial: THREE.Material | THREE.Material[],
  baseMaterial: THREE.Material,
) {
  const sourceMaterials = Array.isArray(sourceMaterial) ? sourceMaterial : [sourceMaterial];
  const hasCoplanarOffsets = sourceMaterials.some((nextMaterial) =>
    isCoplanarOffsetMaterial(nextMaterial),
  );
  if (!hasCoplanarOffsets) {
    return baseMaterial;
  }

  const nextMaterials = sourceMaterials.map((nextMaterial) => {
    const overrideMaterial = baseMaterial.clone();
    overrideMaterial.userData = {
      ...(overrideMaterial.userData ?? {}),
      [GENERATED_OVERRIDE_MATERIAL_KEY]: true,
    };

    if (isCoplanarOffsetMaterial(nextMaterial)) {
      return markMaterialAsCoplanarOffset(overrideMaterial);
    }

    overrideMaterial.needsUpdate = true;
    return overrideMaterial;
  });

  return Array.isArray(sourceMaterial) ? nextMaterials : nextMaterials[0];
}

export function DAERendererImpl({
  url,
  material,
  enableShadows = true,
  assets,
  assetBaseDir,
  normalizeRoot = false,
  preserveOriginalMaterial = false,
  scale,
  onResolved,
}: DAERendererImplProps) {
  const manager = useLoadingManager(assets, assetBaseDir);
  const colladaScene = use(useMemo(() => loadColladaScene(url, manager), [manager, url]));
  const { clone, overrideMeshes } = useMemo(() => {
    return cloneColladaScenePreservingRootTransform(
      colladaScene,
      normalizeRoot,
      preserveOriginalMaterial,
    );
  }, [colladaScene, normalizeRoot, preserveOriginalMaterial]);

  useLayoutEffect(() => {
    if (enableShadows) {
      applyVisualMeshShadowPolicyToObject(clone);
    }

    overrideMeshes.forEach((mesh) => {
      const originalMaterial =
        (mesh.userData?.[ORIGINAL_MATERIAL_KEY] as THREE.Material | THREE.Material[] | undefined) ??
        mesh.material;
      mesh.userData = {
        ...(mesh.userData ?? {}),
        [ORIGINAL_MATERIAL_KEY]: originalMaterial,
      };

      disposeGeneratedOverrideMaterials(mesh.material);
      mesh.material = buildOverrideMaterial(originalMaterial, material);
    });

    return () => {
      overrideMeshes.forEach((mesh) => {
        disposeGeneratedOverrideMaterials(mesh.material);
      });
    };
  }, [clone, enableShadows, material, overrideMeshes]);

  useEffect(() => {
    onResolved?.();
  }, [clone, onResolved]);

  useEffect(
    () => () => {
      disposeObject3DGraph(clone);
    },
    [clone],
  );

  const scaleArr: [number, number, number] = scale ? [scale.x, scale.y, scale.z] : [1, 1, 1];

  return (
    <group scale={scaleArr}>
      <primitive object={clone} />
    </group>
  );
}

export default DAERendererImpl;
