import { useEffect, useRef } from 'react';
import { useThree } from '@react-three/fiber';
import type { Camera, Object3D, Scene, WebGLRenderer } from 'three';

type CompileTargetScene = Scene | null | undefined;
const SOFTWARE_RENDERER_PATTERN = /(swiftshader|llvmpipe|softpipe|software|mesa offscreen|microsoft basic render)/i;

interface CompileCapableRenderer {
  compile?: (scene: Object3D, camera: Camera, targetScene?: CompileTargetScene) => unknown;
  compileAsync?: (scene: Object3D, camera: Camera, targetScene?: CompileTargetScene) => Promise<unknown>;
  getContext?: WebGLRenderer['getContext'];
}

export type SceneCompileWarmupMode = 'async' | 'sync' | 'unsupported';

interface SceneCompileWarmupProps {
  active: boolean;
  warmupKey: string;
  settleFrames?: number;
  idleTimeoutMs?: number;
}

export function isSceneCompileWarmupBlocked(renderer: Pick<CompileCapableRenderer, 'getContext'>): boolean {
  const context = renderer.getContext?.();
  if (context?.isContextLost?.()) {
    return true;
  }

  const debugRendererInfo = context?.getExtension?.('WEBGL_debug_renderer_info');
  const unmaskedRendererValue = debugRendererInfo
    ? context?.getParameter?.(debugRendererInfo.UNMASKED_RENDERER_WEBGL)
    : null;
  const maskedRendererValue = typeof (context as WebGLRenderingContext | null)?.RENDERER === 'number'
    ? context?.getParameter?.((context as WebGLRenderingContext).RENDERER)
    : null;
  const rendererDescription = [unmaskedRendererValue, maskedRendererValue]
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
    .join(' | ');

  return SOFTWARE_RENDERER_PATTERN.test(rendererDescription);
}

export async function warmupSceneCompile(
  renderer: CompileCapableRenderer,
  scene: Object3D,
  camera: Camera,
): Promise<SceneCompileWarmupMode> {
  // ponytail: prefer synchronous compile() over compileAsync(). compileAsync's
  // internal readiness poll inside THREE.WebGLRenderer cannot be
  // cancelled and is not guarded against materials being disposed mid-poll.
  // During rapid file switching the old robot's materials are torn down while
  // that poll is still running, so it throws
  // "Cannot read properties of undefined (reading 'isReady')" as an uncaught
  // async poll error (not a promise rejection, so callers can't catch it). Synchronous
  // compile() has no dangling poll and finishes within a single task, so the
  // scene-graph swap cannot interleave. compile() always exists on WebGLRenderer.
  if (typeof renderer.compile === 'function') {
    renderer.compile(scene, camera);
    return 'sync';
  }

  if (typeof renderer.compileAsync === 'function') {
    await renderer.compileAsync(scene, camera);
    return 'async';
  }

  return 'unsupported';
}

export const SceneCompileWarmup = ({
  active,
  warmupKey,
  settleFrames = 2,
  idleTimeoutMs = 300,
}: SceneCompileWarmupProps) => {
  const { gl, scene, camera, invalidate } = useThree();
  const warmedKeysRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!active || !warmupKey || warmedKeysRef.current.has(warmupKey)) {
      return;
    }

    if (isSceneCompileWarmupBlocked(gl)) {
      return;
    }

    const requestIdle = typeof window !== 'undefined'
      ? window.requestIdleCallback?.bind(window)
      : undefined;
    const cancelIdle = typeof window !== 'undefined'
      ? window.cancelIdleCallback?.bind(window)
      : undefined;

    let cancelled = false;
    let frameHandle: number | null = null;
    let idleHandle: number | null = null;
    let timeoutHandle: number | null = null;

    const cancelScheduledWork = () => {
      if (frameHandle !== null && typeof window !== 'undefined') {
        window.cancelAnimationFrame(frameHandle);
        frameHandle = null;
      }

      if (idleHandle !== null && cancelIdle) {
        cancelIdle(idleHandle);
        idleHandle = null;
      }

      if (timeoutHandle !== null && typeof window !== 'undefined') {
        window.clearTimeout(timeoutHandle);
        timeoutHandle = null;
      }
    };

    const runWarmup = async () => {
      if (cancelled || isSceneCompileWarmupBlocked(gl)) {
        return;
      }

      scene.updateMatrixWorld(true);
      camera.updateMatrixWorld(true);
      if ('updateProjectionMatrix' in camera && typeof camera.updateProjectionMatrix === 'function') {
        camera.updateProjectionMatrix();
      }

      invalidate();

      try {
        const mode = await warmupSceneCompile(gl, scene, camera);
        if (mode !== 'unsupported') {
          warmedKeysRef.current.add(warmupKey);
        }
      } catch (error) {
        if (import.meta.env.DEV) {
          console.warn('[SceneCompileWarmup] Failed to precompile scene', error);
        }
      } finally {
        if (!cancelled) {
          invalidate();
        }
      }
    };

    const scheduleWarmup = () => {
      if (cancelled) {
        return;
      }

      if (requestIdle) {
        idleHandle = requestIdle(() => {
          idleHandle = null;
          void runWarmup();
        }, { timeout: idleTimeoutMs });
        return;
      }

      timeoutHandle = window.setTimeout(() => {
        timeoutHandle = null;
        void runWarmup();
      }, idleTimeoutMs);
    };

    const waitForStableFrames = (remainingFrames: number) => {
      if (cancelled) {
        return;
      }

      if (typeof window === 'undefined' || typeof window.requestAnimationFrame !== 'function' || remainingFrames <= 0) {
        scheduleWarmup();
        return;
      }

      frameHandle = window.requestAnimationFrame(() => {
        frameHandle = null;
        waitForStableFrames(remainingFrames - 1);
      });
    };

    invalidate();
    waitForStableFrames(Math.max(0, settleFrames));

    return () => {
      cancelled = true;
      cancelScheduledWork();
    };
  }, [active, camera, gl, idleTimeoutMs, invalidate, scene, settleFrames, warmupKey]);

  return null;
};
