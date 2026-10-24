import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from 'react';
import { useFrame, useThree, type ThreeElements } from '@react-three/fiber';
import * as THREE from 'three';
import { HorizontalBlurShader, VerticalBlurShader } from 'three-stdlib';

interface SnapshotContactShadowsProps extends Omit<ThreeElements['group'], 'ref' | 'scale'> {
  opacity?: number;
  width?: number;
  height?: number;
  blur?: number;
  near?: number;
  far?: number;
  smooth?: boolean;
  resolution?: number;
  frames?: number;
  color?: THREE.ColorRepresentation;
  depthWrite?: boolean;
}

const SNAPSHOT_HELPER_NAMES = new Set(['ReferenceGrid', 'GroundShadowPlane']);

export const SnapshotContactShadows = forwardRef<THREE.Group, SnapshotContactShadowsProps>(function SnapshotContactShadows({
  opacity = 1,
  width = 10,
  height = 10,
  blur = 1.8,
  near = 0,
  far = 12,
  smooth = true,
  resolution = 768,
  frames = 1,
  color = '#000000',
  depthWrite = false,
  renderOrder,
  ...props
}, forwardedRef) {
  const groupRef = useRef<THREE.Group>(null);
  const shadowCameraRef = useRef<THREE.OrthographicCamera>(null);
  const scene = useThree((state) => state.scene);
  const gl = useThree((state) => state.gl);
  const frameCountRef = useRef(0);

  const {
    renderTarget,
    renderTargetBlur,
    planeGeometry,
    depthMaterial,
    blurPlane,
    horizontalBlurMaterial,
    verticalBlurMaterial,
  } = useMemo(() => {
    const nextRenderTarget = new THREE.WebGLRenderTarget(resolution, resolution);
    const nextRenderTargetBlur = new THREE.WebGLRenderTarget(resolution, resolution);
    nextRenderTarget.texture.generateMipmaps = false;
    nextRenderTargetBlur.texture.generateMipmaps = false;

    const nextPlaneGeometry = new THREE.PlaneGeometry(width, height);
    const nextBlurPlane = new THREE.Mesh(nextPlaneGeometry);
    const nextDepthMaterial = new THREE.MeshDepthMaterial();
    nextDepthMaterial.depthTest = false;
    nextDepthMaterial.depthWrite = false;
    nextDepthMaterial.onBeforeCompile = (shader) => {
      shader.uniforms = {
        ...shader.uniforms,
        ucolor: { value: new THREE.Color(color) },
      };
      shader.fragmentShader = shader.fragmentShader.replace(
        'void main() {',
        `
          uniform vec3 ucolor;
          void main() {
        `,
      );
      shader.fragmentShader = shader.fragmentShader.replace(
        'vec4( vec3( 1.0 - fragCoordZ ), opacity );',
        'vec4( ucolor * fragCoordZ * 2.0, ( 1.0 - fragCoordZ ) * 1.0 );',
      );
    };

    const nextHorizontalBlurMaterial = new THREE.ShaderMaterial(HorizontalBlurShader);
    const nextVerticalBlurMaterial = new THREE.ShaderMaterial(VerticalBlurShader);
    nextHorizontalBlurMaterial.depthTest = false;
    nextVerticalBlurMaterial.depthTest = false;

    return {
      renderTarget: nextRenderTarget,
      renderTargetBlur: nextRenderTargetBlur,
      planeGeometry: nextPlaneGeometry,
      depthMaterial: nextDepthMaterial,
      blurPlane: nextBlurPlane,
      horizontalBlurMaterial: nextHorizontalBlurMaterial,
      verticalBlurMaterial: nextVerticalBlurMaterial,
    };
  }, [color, height, resolution, width]);

  useImperativeHandle(forwardedRef, () => groupRef.current as THREE.Group, []);

  useEffect(() => {
    frameCountRef.current = 0;
  }, [blur, color, far, frames, height, near, opacity, resolution, smooth, width]);

  useEffect(() => () => {
    renderTarget.dispose();
    renderTargetBlur.dispose();
    planeGeometry.dispose();
    depthMaterial.dispose();
    horizontalBlurMaterial.dispose();
    verticalBlurMaterial.dispose();
  }, [depthMaterial, horizontalBlurMaterial, planeGeometry, renderTarget, renderTargetBlur, verticalBlurMaterial]);

  useFrame(() => {
    const group = groupRef.current;
    const shadowCamera = shadowCameraRef.current;
    if (!group || !shadowCamera || frameCountRef.current >= frames) {
      return;
    }

    frameCountRef.current += 1;

    const hiddenHelpers: Array<{ object: THREE.Object3D; visible: boolean }> = [];
    scene.traverse((object) => {
      if (object === group || group.children.includes(object)) {
        return;
      }

      if (
        object.userData?.isHelper === true
        || object.userData?.excludeFromSceneBounds === true
        || SNAPSHOT_HELPER_NAMES.has(object.name)
      ) {
        hiddenHelpers.push({ object, visible: object.visible });
        object.visible = false;
      }
    });

    const previousBackground = scene.background;
    const previousOverrideMaterial = scene.overrideMaterial;
    const previousAutoClear = gl.autoClear;

    shadowCamera.position.set(0, 0, Math.max(far * 0.5, 0.001));
    shadowCamera.lookAt(0, 0, 0);
    shadowCamera.updateProjectionMatrix();
    shadowCamera.updateMatrixWorld();

    const blurShadows = (blurAmount: number) => {
      blurPlane.visible = true;
      blurPlane.material = horizontalBlurMaterial;
      horizontalBlurMaterial.uniforms.tDiffuse.value = renderTarget.texture;
      horizontalBlurMaterial.uniforms.h.value = blurAmount / 256;
      gl.setRenderTarget(renderTargetBlur);
      gl.render(blurPlane, shadowCamera);

      blurPlane.material = verticalBlurMaterial;
      verticalBlurMaterial.uniforms.tDiffuse.value = renderTargetBlur.texture;
      verticalBlurMaterial.uniforms.v.value = blurAmount / 256;
      gl.setRenderTarget(renderTarget);
      gl.render(blurPlane, shadowCamera);
      blurPlane.visible = false;
    };

    try {
      group.visible = false;
      scene.background = null;
      scene.overrideMaterial = depthMaterial;
      gl.autoClear = true;
      gl.setRenderTarget(renderTarget);
      gl.clear();
      gl.render(scene, shadowCamera);
      blurShadows(blur);
      if (smooth) {
        blurShadows(blur * 0.45);
      }
      gl.setRenderTarget(null);
    } finally {
      group.visible = true;
      scene.overrideMaterial = previousOverrideMaterial;
      scene.background = previousBackground;
      gl.autoClear = previousAutoClear;
      gl.setRenderTarget(null);
      hiddenHelpers.forEach(({ object, visible }) => {
        object.visible = visible;
      });
    }
  });

  return (
    <group ref={groupRef} {...props}>
      <mesh
        renderOrder={renderOrder}
        geometry={planeGeometry}
        scale={[1, -1, 1]}
        frustumCulled={false}
      >
        <meshBasicMaterial
          transparent
          map={renderTarget.texture}
          opacity={opacity}
          depthWrite={depthWrite}
        />
      </mesh>
      <orthographicCamera
        ref={shadowCameraRef}
        args={[-width / 2, width / 2, height / 2, -height / 2, near, far]}
      />
    </group>
  );
});
