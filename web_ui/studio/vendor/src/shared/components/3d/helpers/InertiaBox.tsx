/**
 * Inertia Box Visualization Component
 * Displays a semi-transparent box representing the inertia of a link
 */

import React from 'react';
import * as THREE from 'three';
import type { UrdfLink } from '@/types';
import { MathUtils as DataUtils } from '@/shared/utils/math';
import { ignoreRaycast } from '@/shared/utils/three/ignoreRaycast';

interface InertiaBoxProps {
  link: UrdfLink;
  hovered?: boolean;
  selected?: boolean;
}

export const InertiaBox = React.memo(
  ({ link, hovered = false, selected = false }: InertiaBoxProps) => {
    const outlineRef = React.useRef<THREE.LineSegments>(null);
    React.useLayoutEffect(() => {
      if (outlineRef.current) {
        outlineRef.current.raycast = ignoreRaycast;
      }
    }, []);

    const inertial = link.inertial;
    if (!inertial) return null;

    const boxData = DataUtils.computeInertiaBox(inertial);
    if (!boxData) return null;

    const { width, height, depth, rotation } = boxData;

    const origin = inertial.origin || { xyz: { x: 0, y: 0, z: 0 }, rpy: { r: 0, p: 0, y: 0 } };
    const pos = origin.xyz || { x: 0, y: 0, z: 0 };

    const originRot = new THREE.Euler(
      origin.rpy?.r || 0,
      origin.rpy?.p || 0,
      origin.rpy?.y || 0,
      'ZYX',
    );
    const originQuat = new THREE.Quaternion().setFromEuler(originRot);
    const finalQuat = originQuat.multiply(rotation);
    const finalEuler = new THREE.Euler().setFromQuaternion(finalQuat, 'ZYX');
    const isActive = hovered || selected;
    const fillColor = isActive ? 0x7dd3fc : 0x4a9eff;
    const edgeColor = isActive ? 0xe0f2fe : 0x93c5fd;
    const opacity = hovered ? 0.58 : selected ? 0.5 : 0.35;

    return (
      <group position={[pos.x, pos.y, pos.z]} rotation={finalEuler}>
        <mesh renderOrder={9999}>
          <boxGeometry args={[width, height, depth]} />
          <meshPhongMaterial
            color={fillColor}
            emissive={isActive ? 0x0f172a : 0x000000}
            transparent
            opacity={opacity}
            depthWrite={false}
            depthTest
            shininess={isActive ? 80 : 50}
          />
        </mesh>
        <lineSegments
          ref={outlineRef}
          renderOrder={10000}
          scale={hovered ? [1.01, 1.01, 1.01] : [1.005, 1.005, 1.005]}
        >
          <edgesGeometry args={[new THREE.BoxGeometry(width, height, depth)]} />
          <lineBasicMaterial
            color={edgeColor}
            transparent
            opacity={hovered ? 0.95 : 0.7}
            depthWrite={false}
            depthTest
          />
        </lineSegments>
      </group>
    );
  },
);
