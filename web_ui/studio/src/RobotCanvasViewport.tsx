/** Local viewport adapter; the pinned RobotCanvas and model runtime stay intact. */
import { memo, useContext, useEffect, type ReactNode, type RefObject } from 'react';
import { useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { WorkspaceCanvas, STUDIO_ENVIRONMENT_INTENSITY, type SnapshotCaptureAction } from '@/shared/components/3d';
import { ViewportControls } from './viewportControls';

function CameraTools() {
  const {robot, action, onResolution} = useContext(ViewportControls);
  const camera = useThree(state => state.camera) as THREE.PerspectiveCamera;
  const controls = useThree(state => state.controls) as any;
  const invalidate = useThree(state => state.invalidate);
  const dpr = useThree(state => state.viewport.dpr);

  useEffect(() => { onResolution(dpr); }, [dpr, onResolution]);
  useEffect(() => {
    if (!robot || !controls) return;
    const debug = (window as any).__migrationStudio;
    if (debug) Object.assign(debug, {camera, controls, resolution:dpr});
  }, [camera, controls, dpr, robot]);
  useEffect(() => {
    if (!action || !robot || !controls?.target) return;
    robot.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(robot);
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const offset = camera.position.clone().sub(controls.target);
    if (action.type === 'zoomIn' || action.type === 'zoomOut') {
      const factor = action.type === 'zoomIn' ? 0.8 : 1.25;
      const distance = Math.max(0.02, offset.length() * factor);
      camera.position.copy(controls.target).add(offset.normalize().multiplyScalar(distance));
    } else {
      if (action.type === 'reset') offset.set(1.25, -1.65, 1.0);
      if (action.type === 'front') offset.set(1, 0, 0);
      if (action.type === 'side') offset.set(0, -1, 0);
      if (action.type === 'top') offset.set(0, 0, 1);
      if (offset.lengthSq() < 1e-12) offset.set(1.25, -1.65, 1.0);
      camera.up.set(0, action.type === 'top' ? 1 : 0, action.type === 'top' ? 0 : 1);
      // A bounding sphere fits each aligned view, including narrow portrait canvases.
      const radius = Math.max(size.length() / 2, 0.025);
      const verticalFov = THREE.MathUtils.degToRad(camera.fov);
      const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * Math.max(camera.aspect, 0.1));
      const distance = radius / Math.sin(Math.min(verticalFov, horizontalFov) / 2) * 1.12;
      controls.target.copy(center);
      camera.position.copy(center).add(offset.normalize().multiplyScalar(distance));
    }
    camera.lookAt(controls.target);
    camera.updateProjectionMatrix();
    controls.update();
    camera.updateMatrixWorld(true);
    invalidate();
  }, [action, camera, controls, invalidate, robot]);
  return null;
}

interface Props {
  children: ReactNode;
  cameraProjection?: 'perspective' | 'orthographic';
  groundOffset?: number;
  lang: 'en' | 'zh';
  onOrbitEnd?: () => void;
  onOrbitStart?: () => void;
  onPointerMissed?: () => void;
  orbitEnabled: boolean;
  resolvedTheme?: 'light' | 'dark';
  robotName?: string;
  snapshotAction?: RefObject<SnapshotCaptureAction | null>;
}

export const RobotCanvasViewport = memo(function RobotCanvasViewport({
  children, cameraProjection = 'perspective', groundOffset = 0, lang,
  onOrbitEnd, onOrbitStart, onPointerMissed, orbitEnabled,
  resolvedTheme = 'light', robotName = 'robot', snapshotAction,
}: Props) {
  const {mode, showWorldAxes} = useContext(ViewportControls);
  return <WorkspaceCanvas
    theme={resolvedTheme} cameraProjection={cameraProjection} lang={lang}
    minDpr={1.75} maxDpr={2.5} shadowMapSize={1536}
    className="relative h-full w-full" snapshotAction={snapshotAction}
    robotName={robotName} onPointerMissed={onPointerMissed}
    environment="studio" environmentIntensityByTheme={STUDIO_ENVIRONMENT_INTENSITY.viewer}
    groundOffset={groundOffset} toneMapping={THREE.NeutralToneMapping} toneMappingExposure={1}
    cameraFollowPrimary showUsageGuide={false} showWorldOriginAxes={showWorldAxes}
    orbitControlsProps={{
      enabled: orbitEnabled, onStart: onOrbitStart, onEnd: onOrbitEnd,
      mouseButtons: {LEFT: mode === 'pan' ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN},
      touches: {ONE: mode === 'pan' ? THREE.TOUCH.PAN : THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN},
    }}
  >{children}<CameraTools /></WorkspaceCanvas>;
});
