import { useEffect, useLayoutEffect, useMemo } from 'react';
import { useLoader } from '@react-three/fiber';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { clone as cloneSkeleton } from 'three/examples/jsm/utils/SkeletonUtils.js';
import { useLoadingManager } from '../meshLoadingManager';
import { applyVisualMeshShadowPolicyToObject } from '@/core/utils/visualMeshShadowPolicy';
import { disposeObject3D, disposeObject3DGraph } from '@/shared/utils/three/dispose';

interface ScaleProps {
  x: number;
  y: number;
  z: number;
}

interface GLTFRendererImplProps {
  url: string;
  material: THREE.Material;
  enableShadows?: boolean;
  assets: Record<string, string>;
  assetBaseDir?: string;
  preserveOriginalMaterial?: boolean;
  scale?: ScaleProps;
  onResolved?: () => void;
}

const gltfUrlRefCounts = new Map<string, number>();

function retainGltfUrl(url: string): void {
  gltfUrlRefCounts.set(url, (gltfUrlRefCounts.get(url) ?? 0) + 1);
}

function releaseGltfUrl(url: string, scene: THREE.Object3D): void {
  const nextCount = (gltfUrlRefCounts.get(url) ?? 1) - 1;
  if (nextCount > 0) {
    gltfUrlRefCounts.set(url, nextCount);
    return;
  }

  gltfUrlRefCounts.delete(url);
  useLoader.clear(GLTFLoader, url);
  disposeObject3D(scene, true);
}

export function GLTFRendererImpl({
  url,
  material,
  enableShadows = true,
  assets,
  assetBaseDir,
  preserveOriginalMaterial = false,
  scale,
  onResolved,
}: GLTFRendererImplProps) {
  const manager = useLoadingManager(assets, assetBaseDir);
  const gltf = useLoader(GLTFLoader, url, (loader) => {
    loader.manager = manager;
  });

  useEffect(() => {
    retainGltfUrl(url);
    return () => releaseGltfUrl(url, gltf.scene);
  }, [gltf.scene, url]);

  const { clone, overrideMeshes } = useMemo(() => {
    const nextClone = cloneSkeleton(gltf.scene);
    const meshes: THREE.Mesh[] = [];

    nextClone.traverse((child) => {
      if (!(child as THREE.Mesh).isMesh) {
        return;
      }

      if (!preserveOriginalMaterial) {
        meshes.push(child as THREE.Mesh);
      }
    });

    return { clone: nextClone, overrideMeshes: meshes };
  }, [gltf.scene, preserveOriginalMaterial]);

  useLayoutEffect(() => {
    overrideMeshes.forEach((mesh) => {
      mesh.material = material;
    });
    if (enableShadows) {
      applyVisualMeshShadowPolicyToObject(clone);
    }
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

export default GLTFRendererImpl;
