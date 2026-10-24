type RegressionDebugWindow = Window & {
  __URDF_STUDIO_DEBUG__?: unknown;
  __URDF_STUDIO_MESH_LOAD_PERFORMANCE__?: unknown;
  __URDF_STUDIO_MESH_LOAD_PERFORMANCE_HISTORY__?: unknown;
  __usdStageLoadDebug?: unknown;
  __usdStageLoadDebugHistory?: unknown;
  __visualizerCollisionLoadDebug?: unknown;
  __visualizerCollisionLoadDebugHistory?: unknown;
};

function getWindowSearch(targetWindow?: Pick<Window, 'location'> | null): string {
  try {
    if (targetWindow?.location) {
      return targetWindow.location.search ?? '';
    }
    if (typeof window !== 'undefined') {
      return window.location.search ?? '';
    }
  } catch {
    return '';
  }

  return '';
}

export function isRegressionDebugEnabled(targetWindow?: Pick<Window, 'location'> | null): boolean {
  return new URLSearchParams(getWindowSearch(targetWindow)).get('regressionDebug') === '1';
}

export function clearRegressionDebugGlobals(targetWindow: Window): void {
  const runtimeWindow = targetWindow as RegressionDebugWindow;
  delete runtimeWindow.__URDF_STUDIO_DEBUG__;
  delete runtimeWindow.__URDF_STUDIO_MESH_LOAD_PERFORMANCE__;
  delete runtimeWindow.__URDF_STUDIO_MESH_LOAD_PERFORMANCE_HISTORY__;
  delete runtimeWindow.__usdStageLoadDebug;
  delete runtimeWindow.__usdStageLoadDebugHistory;
  delete runtimeWindow.__visualizerCollisionLoadDebug;
  delete runtimeWindow.__visualizerCollisionLoadDebugHistory;
}
