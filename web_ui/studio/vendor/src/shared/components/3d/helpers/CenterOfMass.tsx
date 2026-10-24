/**
 * Center of Mass Indicator Component
 * Displays a checkerboard sphere at the center of mass position
 */

import React from 'react';
import { Html } from '@react-three/drei';
import type { UrdfLink } from '@/types';
import {
  GIZMO_BASE_RENDER_ORDER,
  COM_VISUAL_RENDER_ORDER,
} from '@/shared/components/3d/unified-transform-controls/gizmoCore';

interface CenterOfMassProps {
  link: UrdfLink;
  hovered?: boolean;
  selected?: boolean;
}

export const LinkCenterOfMass = React.memo(
  ({ link, hovered = false, selected = false }: CenterOfMassProps) => {
    const inertial = link.inertial;
    if (!inertial || inertial.mass <= 0) return null;

    const origin = inertial.origin || { xyz: { x: 0, y: 0, z: 0 } };
    const pos = origin.xyz || { x: 0, y: 0, z: 0 };
    const isActive = hovered || selected;
    const radius = hovered ? 0.017 : selected ? 0.016 : 0.015;
    const pickRadius = hovered ? radius * 3.6 : selected ? radius * 3.4 : radius * 3.2;
    const haloOpacity = hovered ? 0.24 : selected ? 0.18 : 0.1;

    return (
      <group position={[pos.x, pos.y, pos.z]} scale={isActive ? 1.08 : 1}>
        {/* Keep a wider invisible hit shell so CoM hover does not fall through to collision meshes behind it. */}
        <mesh renderOrder={GIZMO_BASE_RENDER_ORDER}>
          <sphereGeometry args={[pickRadius, 16, 16]} />
          <meshBasicMaterial colorWrite={false} depthWrite={false} depthTest={false} />
        </mesh>
        <mesh renderOrder={COM_VISUAL_RENDER_ORDER}>
          <sphereGeometry args={[radius * 1.9, 20, 20]} />
          <meshBasicMaterial
            color={hovered ? '#fb7185' : '#ef4444'}
            transparent
            opacity={haloOpacity}
            depthWrite={false}
            depthTest={false}
          />
        </mesh>
        {/* 8 octants forming a sphere */}
        <mesh rotation={[0, 0, 0]} renderOrder={COM_VISUAL_RENDER_ORDER + 1}>
          <sphereGeometry args={[radius, 16, 16, 0, Math.PI / 2, 0, Math.PI / 2]} />
          <meshBasicMaterial color="#000000" depthTest={false} />
        </mesh>
        <mesh rotation={[0, Math.PI / 2, 0]} renderOrder={COM_VISUAL_RENDER_ORDER + 1}>
          <sphereGeometry args={[radius, 16, 16, 0, Math.PI / 2, 0, Math.PI / 2]} />
          <meshBasicMaterial color="#ffffff" depthTest={false} />
        </mesh>
        <mesh rotation={[0, Math.PI, 0]} renderOrder={COM_VISUAL_RENDER_ORDER + 1}>
          <sphereGeometry args={[radius, 16, 16, 0, Math.PI / 2, 0, Math.PI / 2]} />
          <meshBasicMaterial color="#000000" depthTest={false} />
        </mesh>
        <mesh rotation={[0, -Math.PI / 2, 0]} renderOrder={COM_VISUAL_RENDER_ORDER + 1}>
          <sphereGeometry args={[radius, 16, 16, 0, Math.PI / 2, 0, Math.PI / 2]} />
          <meshBasicMaterial color="#ffffff" depthTest={false} />
        </mesh>

        <mesh rotation={[Math.PI, 0, 0]} renderOrder={COM_VISUAL_RENDER_ORDER + 1}>
          <sphereGeometry args={[radius, 16, 16, 0, Math.PI / 2, 0, Math.PI / 2]} />
          <meshBasicMaterial color="#000000" depthTest={false} />
        </mesh>
        <mesh rotation={[Math.PI, Math.PI / 2, 0]} renderOrder={COM_VISUAL_RENDER_ORDER + 1}>
          <sphereGeometry args={[radius, 16, 16, 0, Math.PI / 2, 0, Math.PI / 2]} />
          <meshBasicMaterial color="#ffffff" depthTest={false} />
        </mesh>
        <mesh rotation={[Math.PI, Math.PI, 0]} renderOrder={COM_VISUAL_RENDER_ORDER + 1}>
          <sphereGeometry args={[radius, 16, 16, 0, Math.PI / 2, 0, Math.PI / 2]} />
          <meshBasicMaterial color="#000000" depthTest={false} />
        </mesh>
        <mesh rotation={[Math.PI, -Math.PI / 2, 0]} renderOrder={COM_VISUAL_RENDER_ORDER + 1}>
          <sphereGeometry args={[radius, 16, 16, 0, Math.PI / 2, 0, Math.PI / 2]} />
          <meshBasicMaterial color="#ffffff" depthTest={false} />
        </mesh>

        <Html position={[0.03, 0.03, 0]} style={{ pointerEvents: 'none' }}>
          <div
            className={`text-[8px] px-1 rounded whitespace-nowrap shadow-sm transition-colors ${
              isActive
                ? 'text-red-600 dark:text-red-300 bg-white dark:bg-black/75 border border-red-300 dark:border-red-700'
                : 'text-red-500 dark:text-red-400 bg-white/90 dark:bg-black/60 border border-red-200 dark:border-red-900'
            }`}
          >
            CoM
          </div>
        </Html>
      </group>
    );
  },
);

// Alias for backwards compatibility
export const CenterOfMass = LinkCenterOfMass;
