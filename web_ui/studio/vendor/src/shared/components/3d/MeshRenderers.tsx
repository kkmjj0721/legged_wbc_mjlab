/**
 * Shared Mesh Renderer Components
 * Used by the unified editor viewer pipeline.
 */

import React, { lazy } from 'react';
import * as THREE from 'three';
export { useLoadingManager } from './meshLoadingManager';

interface ScaleProps {
  x: number;
  y: number;
  z: number;
}

const LazySTLRenderer = lazy(() => import('./renderers/STLRendererImpl'));
const LazyOBJRenderer = lazy(() => import('./renderers/OBJRendererImpl'));
const LazyDAERenderer = lazy(() => import('./renderers/DAERendererImpl'));
const LazyGLTFRenderer = lazy(() => import('./renderers/GLTFRendererImpl'));

// STL Renderer
export const STLRenderer = React.memo(
  ({
    url,
    material,
    enableShadows,
    scale,
    onResolved,
  }: {
    url: string;
    material: THREE.Material;
    enableShadows?: boolean;
    scale?: ScaleProps;
    onResolved?: () => void;
  }) => {
    return (
      <LazySTLRenderer
        url={url}
        material={material}
        enableShadows={enableShadows}
        scale={scale}
        onResolved={onResolved}
      />
    );
  },
);

// OBJ Renderer
export const OBJRenderer = React.memo(
  ({
    url,
    material,
    color,
    enableShadows,
    assets,
    assetBaseDir,
    logicalAssetPath,
    preserveOriginalMaterial,
    scale,
    onResolved,
  }: {
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
  }) => {
    return (
      <LazyOBJRenderer
        url={url}
        material={material}
        color={color}
        enableShadows={enableShadows}
        assets={assets}
        assetBaseDir={assetBaseDir}
        logicalAssetPath={logicalAssetPath}
        preserveOriginalMaterial={preserveOriginalMaterial}
        scale={scale}
        onResolved={onResolved}
      />
    );
  },
);

// DAE (Collada) Renderer
export const DAERenderer = React.memo(
  ({
    url,
    material,
    enableShadows,
    assets,
    assetBaseDir,
    normalizeRoot,
    preserveOriginalMaterial,
    scale,
    onResolved,
  }: {
    url: string;
    material: THREE.Material;
    enableShadows?: boolean;
    assets: Record<string, string>;
    assetBaseDir?: string;
    normalizeRoot?: boolean;
    preserveOriginalMaterial?: boolean;
    scale?: ScaleProps;
    onResolved?: () => void;
  }) => {
    return (
      <LazyDAERenderer
        url={url}
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
  },
);

// GLTF / GLB Renderer
export const GLTFRenderer = React.memo(
  ({
    url,
    material,
    enableShadows,
    assets,
    assetBaseDir,
    preserveOriginalMaterial,
    scale,
    onResolved,
  }: {
    url: string;
    material: THREE.Material;
    enableShadows?: boolean;
    assets: Record<string, string>;
    assetBaseDir?: string;
    preserveOriginalMaterial?: boolean;
    scale?: ScaleProps;
    onResolved?: () => void;
  }) => {
    return (
      <LazyGLTFRenderer
        url={url}
        material={material}
        enableShadows={enableShadows}
        assets={assets}
        assetBaseDir={assetBaseDir}
        preserveOriginalMaterial={preserveOriginalMaterial}
        scale={scale}
        onResolved={onResolved}
      />
    );
  },
);
