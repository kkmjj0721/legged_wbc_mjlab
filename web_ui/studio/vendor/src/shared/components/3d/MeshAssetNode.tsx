import React, { useEffect, useMemo } from 'react';
import { useLoader } from '@react-three/fiber';
import * as THREE from 'three';

import { findAssetByPath } from '@/core/loaders/meshLoader';
import { getSourceFileDirectory } from '@/core/parsers/meshPathUtils';
import { getAssetFileExtension, isImageAssetPath } from '@/core/utils/assetFileTypes';

import { DAERenderer, GLTFRenderer, OBJRenderer, STLRenderer } from './MeshRenderers';

interface ScaleProps {
  x: number;
  y: number;
  z: number;
}

interface MeshAssetNodeProps {
  meshPath?: string | null;
  assets: Record<string, string>;
  material: THREE.Material;
  color?: string;
  enableShadows?: boolean;
  scale?: ScaleProps;
  normalizeRoot?: boolean;
  preserveOriginalMaterial?: boolean;
  onResolved?: () => void;
  missingContent?: React.ReactNode;
  unknownContent?: React.ReactNode;
}

const DEFAULT_COLOR = '#ffffff';
const imageTextureUrlRefCounts = new Map<string, number>();

function retainImageTextureUrl(url: string): void {
  imageTextureUrlRefCounts.set(url, (imageTextureUrlRefCounts.get(url) ?? 0) + 1);
}

function releaseImageTextureUrl(url: string, texture: THREE.Texture): void {
  const nextCount = (imageTextureUrlRefCounts.get(url) ?? 1) - 1;
  if (nextCount > 0) {
    imageTextureUrlRefCounts.set(url, nextCount);
    return;
  }

  imageTextureUrlRefCounts.delete(url);
  useLoader.clear(THREE.TextureLoader, url);
  texture.dispose();
}

function resolveImagePlaneScale(
  texture: THREE.Texture,
  scale?: ScaleProps,
): [number, number, number] {
  const width = scale?.x || 1;
  const fallbackHeight = scale?.y || 1;
  const depth = scale?.z || 1;
  const image = texture.image as { width?: number; height?: number } | undefined;

  if (!image?.width || !image?.height) {
    return [width, fallbackHeight, depth];
  }

  const aspectHeight = width * (image.height / image.width);
  const height = fallbackHeight === 1 ? aspectHeight : fallbackHeight;
  return [width, height, depth];
}

function ImageAssetPlane({
  url,
  material,
  color,
  enableShadows,
  scale,
  onResolved,
}: {
  url: string;
  material: THREE.Material;
  color: string;
  enableShadows?: boolean;
  scale?: ScaleProps;
  onResolved?: () => void;
}) {
  const texture = useLoader(THREE.TextureLoader, url);

  useEffect(() => {
    retainImageTextureUrl(url);
    return () => releaseImageTextureUrl(url, texture);
  }, [texture, url]);

  const texturedMaterial = useMemo(() => {
    const nextMaterial = material.clone();

    if ('map' in nextMaterial) {
      (nextMaterial as THREE.MeshStandardMaterial).map = texture;
    }
    if ('color' in nextMaterial) {
      (nextMaterial as THREE.MeshStandardMaterial).color.set(color);
    }
    if ('side' in nextMaterial) {
      nextMaterial.side = THREE.DoubleSide;
    }
    if ('transparent' in nextMaterial) {
      nextMaterial.transparent = true;
    }
    if ('alphaTest' in nextMaterial) {
      (nextMaterial as THREE.MeshStandardMaterial).alphaTest = 0.001;
    }

    nextMaterial.needsUpdate = true;
    return nextMaterial;
  }, [color, material, texture]);
  const planeScale = useMemo(() => resolveImagePlaneScale(texture, scale), [scale, texture]);

  useEffect(() => {
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    onResolved?.();
  }, [onResolved, texture]);

  useEffect(
    () => () => {
      texturedMaterial.dispose();
    },
    [texturedMaterial],
  );

  return (
    <mesh scale={planeScale} castShadow={enableShadows} receiveShadow={enableShadows}>
      <planeGeometry args={[1, 1]} />
      <primitive object={texturedMaterial} attach="material" />
    </mesh>
  );
}

export function MeshAssetNode({
  meshPath,
  assets,
  material,
  color = DEFAULT_COLOR,
  enableShadows = true,
  scale,
  normalizeRoot,
  preserveOriginalMaterial,
  onResolved,
  missingContent = null,
  unknownContent = null,
}: MeshAssetNodeProps) {
  if (!meshPath) {
    return missingContent;
  }

  const assetUrl = findAssetByPath(meshPath, assets);
  if (!assetUrl) {
    return missingContent;
  }

  const extension = getAssetFileExtension(meshPath);
  const assetBaseDir = getSourceFileDirectory(meshPath);

  if (isImageAssetPath(meshPath)) {
    return (
      <ImageAssetPlane
        url={assetUrl}
        material={material}
        color={color}
        enableShadows={enableShadows}
        scale={scale}
        onResolved={onResolved}
      />
    );
  }

  if (extension === 'stl') {
    return (
      <STLRenderer
        url={assetUrl}
        material={material}
        enableShadows={enableShadows}
        scale={scale}
        onResolved={onResolved}
      />
    );
  }

  if (extension === 'obj') {
    return (
      <OBJRenderer
        url={assetUrl}
        material={material}
        color={color}
        enableShadows={enableShadows}
        assets={assets}
        assetBaseDir={assetBaseDir}
        logicalAssetPath={meshPath}
        preserveOriginalMaterial={preserveOriginalMaterial}
        scale={scale}
        onResolved={onResolved}
      />
    );
  }

  if (extension === 'dae') {
    return (
      <DAERenderer
        url={assetUrl}
        material={material}
        enableShadows={enableShadows}
        assets={assets}
        assetBaseDir={assetBaseDir}
        normalizeRoot={normalizeRoot}
        preserveOriginalMaterial={preserveOriginalMaterial}
        scale={scale}
        onResolved={onResolved}
      />
    );
  }

  if (extension === 'gltf' || extension === 'glb') {
    return (
      <GLTFRenderer
        url={assetUrl}
        material={material}
        enableShadows={enableShadows}
        assets={assets}
        assetBaseDir={assetBaseDir}
        preserveOriginalMaterial={preserveOriginalMaterial}
        scale={scale}
        onResolved={onResolved}
      />
    );
  }

  return unknownContent;
}
