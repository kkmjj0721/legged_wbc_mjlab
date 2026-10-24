import { Suspense, useMemo } from 'react';
import { Environment, MeshReflectorMaterial } from '@react-three/drei';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { Theme } from '@/types';
import { computeVisibleMeshBounds } from '@/shared/utils/threeBounds';
import type { SnapshotCaptureOptions } from './snapshotConfig';
import {
  SNAPSHOT_ENVIRONMENT_PRESET_SETTINGS,
  SNAPSHOT_GROUND_STYLE_SETTINGS,
} from './snapshotSceneQuality';
import { GroundShadowPlane } from './GroundShadowPlane';
import { resolveGroundPlaneLayout } from './groundPlaneSizing';
import { NeutralStudioEnvironment } from './NeutralStudioEnvironment';
import { SnapshotContactShadows } from './SnapshotContactShadows';
import { resolveEffectiveTheme } from './themeUtils';

interface SnapshotExportLookProps {
  options: SnapshotCaptureOptions;
  theme: Theme;
  groundOffset?: number;
}

const ignoreRaycast = (_raycaster: THREE.Raycaster, _intersects: THREE.Intersection[]) => undefined;

export function SnapshotExportLook({ options, theme, groundOffset = 0 }: SnapshotExportLookProps) {
  const scene = useThree((state) => state.scene);
  const effectiveTheme = resolveEffectiveTheme(theme);
  const environmentSettings = SNAPSHOT_ENVIRONMENT_PRESET_SETTINGS[options.environmentPreset];
  const groundSettings = SNAPSHOT_GROUND_STYLE_SETTINGS[options.groundStyle];

  const layout = useMemo(() => {
    return resolveGroundPlaneLayout(computeVisibleMeshBounds(scene));
  }, [scene]);

  const environmentRotation = useMemo(
    () => new THREE.Euler(0, 0, environmentSettings.environmentRotationZ),
    [environmentSettings.environmentRotationZ],
  );
  const contactSize = Math.max(10, Math.min(160, layout.size * 0.78));
  const floorSize = Math.max(18, Math.min(280, layout.size * 1.12));
  const contactShadowColor = effectiveTheme === 'light' ? '#111827' : '#000000';
  const reflectorColor = effectiveTheme === 'light' ? '#edf3fa' : '#101720';
  const reflectorResolution =
    options.detailLevel === 'ultra' ? 1024 : options.detailLevel === 'high' ? 896 : 768;

  return (
    <>
      <Suspense fallback={null}>
        {environmentSettings.kind === 'studio' ? (
          <NeutralStudioEnvironment
            intensity={environmentSettings.environmentIntensity[effectiveTheme]}
          />
        ) : null}
        {environmentSettings.kind === 'hdri' ? (
          <Environment
            files="/potsdamer_platz_1k.hdr"
            background={false}
            environmentIntensity={environmentSettings.environmentIntensity[effectiveTheme]}
            environmentRotation={environmentRotation}
          />
        ) : null}
      </Suspense>

      {options.groundStyle !== 'shadow' ? (
        <SnapshotContactShadows
          name="SnapshotContactShadows"
          position={[layout.centerX, layout.centerY, groundOffset + 0.001]}
          width={contactSize}
          height={contactSize}
          blur={groundSettings.contactShadowBlur}
          opacity={groundSettings.contactShadowOpacity}
          resolution={groundSettings.contactShadowResolution}
          far={groundSettings.contactShadowFar}
          color={contactShadowColor}
          frames={1}
          smooth
          renderOrder={-104}
          userData={{ isHelper: true, excludeFromSceneBounds: true }}
        />
      ) : null}

      {options.groundStyle === 'shadow' ? (
        <GroundShadowPlane
          name="SnapshotGroundShadowPlane"
          theme={theme}
          groundOffset={groundOffset}
          centerX={layout.centerX}
          centerY={layout.centerY}
          size={floorSize}
        />
      ) : null}

      {options.groundStyle === 'reflective' ? (
        <mesh
          name="SnapshotReflectiveFloor"
          userData={{ isHelper: true, excludeFromSceneBounds: true }}
          position={[layout.centerX, layout.centerY, groundOffset - 0.004]}
          renderOrder={-106}
          frustumCulled={false}
          receiveShadow
          raycast={ignoreRaycast}
        >
          <planeGeometry args={[floorSize, floorSize]} />
          <MeshReflectorMaterial
            color={reflectorColor}
            roughness={groundSettings.reflectorRoughness}
            metalness={0.18}
            blur={groundSettings.reflectorBlur}
            resolution={reflectorResolution}
            mixBlur={groundSettings.reflectorStrength * 0.72}
            mixStrength={groundSettings.reflectorStrength}
            mixContrast={1.45}
            mirror={groundSettings.reflectorMirror}
            depthScale={0.26}
            minDepthThreshold={0.78}
            maxDepthThreshold={1.02}
            depthToBlurRatioBias={0.22}
            reflectorOffset={0.01}
          />
        </mesh>
      ) : null}
    </>
  );
}
