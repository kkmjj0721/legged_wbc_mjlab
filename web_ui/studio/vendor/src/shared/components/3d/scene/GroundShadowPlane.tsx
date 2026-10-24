import * as THREE from 'three';
import type { Theme } from '@/types';
import {
  GROUND_SHADOW_RENDER_ORDER,
  GROUND_SHADOW_STYLE,
  GROUND_SHADOW_Z_OFFSET,
} from './constants';
import { resolveEffectiveTheme } from './themeUtils';

interface GroundShadowPlaneProps {
  theme: Theme;
  groundOffset?: number;
  centerX?: number;
  centerY?: number;
  size?: number;
  name?: string;
}

const ignoreRaycast = (_raycaster: THREE.Raycaster, _intersects: THREE.Intersection[]) => undefined;

export function GroundShadowPlane({
  theme,
  groundOffset = 0,
  centerX = 0,
  centerY = 0,
  size = 20,
  name = 'GroundShadowPlane',
}: GroundShadowPlaneProps) {
  const effectiveTheme = resolveEffectiveTheme(theme);
  const shadowStyle = GROUND_SHADOW_STYLE[effectiveTheme];

  return (
    <mesh
      name={name}
      userData={{ isHelper: true, excludeFromSceneBounds: true }}
      position={[centerX, centerY, groundOffset + GROUND_SHADOW_Z_OFFSET]}
      renderOrder={GROUND_SHADOW_RENDER_ORDER}
      frustumCulled={false}
      receiveShadow
      raycast={ignoreRaycast}
    >
      <planeGeometry args={[size, size]} />
      <shadowMaterial
        color={shadowStyle.color}
        transparent
        opacity={shadowStyle.opacity}
        depthWrite={false}
      />
    </mesh>
  );
}
