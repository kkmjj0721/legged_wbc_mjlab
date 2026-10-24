import { use, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { createGeometryFromSerializedStlData } from '@/core/loaders/stlGeometryData';
import { loadSerializedStlGeometryData } from '@/core/loaders/stlParseWorkerBridge';

interface ScaleProps {
  x: number;
  y: number;
  z: number;
}

interface STLRendererImplProps {
  url: string;
  material: THREE.Material;
  enableShadows?: boolean;
  scale?: ScaleProps;
  onResolved?: () => void;
}

export function STLRendererImpl({
  url,
  material,
  enableShadows = true,
  scale,
  onResolved,
}: STLRendererImplProps) {
  const serializedGeometry = use(useMemo(() => loadSerializedStlGeometryData(url), [url]));
  const clone = useMemo(
    () => createGeometryFromSerializedStlData(serializedGeometry),
    [serializedGeometry],
  );

  useEffect(() => {
    onResolved?.();
  }, [clone, onResolved]);

  useEffect(
    () => () => {
      clone.dispose();
    },
    [clone],
  );

  const scaleArr: [number, number, number] = scale ? [scale.x, scale.y, scale.z] : [1, 1, 1];

  return (
    <mesh
      geometry={clone}
      material={material}
      rotation={[0, 0, 0]}
      scale={scaleArr}
      castShadow={enableShadows}
      receiveShadow={enableShadows}
    />
  );
}

export default STLRendererImpl;
