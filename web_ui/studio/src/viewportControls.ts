import { createContext } from 'react';
import type { Object3D } from 'three';

export type CameraAction = 'fit' | 'reset' | 'front' | 'side' | 'top' | 'zoomIn' | 'zoomOut';

export const ViewportControls = createContext({
  mode: 'rotate' as 'rotate' | 'pan',
  showWorldAxes: true,
  robot: null as Object3D | null,
  action: null as {type: CameraAction; revision: number} | null,
  onResolution: (_dpr: number) => {},
});
