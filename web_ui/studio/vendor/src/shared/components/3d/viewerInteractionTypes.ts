import type { InteractionHelperKind, InteractionSelection } from '@/types';

export type ToolMode =
  | 'select'
  | 'translate'
  | 'rotate'
  | 'universal'
  | 'view'
  | 'face'
  | 'measure'
  | 'paint';
export type ViewerSceneMode = 'editor';
export type ViewerHelperKind = InteractionHelperKind;
export type ViewerInteractiveLayer =
  | 'ik-handle'
  | 'visual'
  | 'collision'
  | 'origin-axes'
  | 'joint-axis'
  | 'center-of-mass'
  | 'inertia';

export interface ViewerRuntimeStageBridge {
  onRobotResolved?: (robot: any | null) => void;
  onSelectionChange?: (
    type: Exclude<InteractionSelection['type'], null>,
    id: string,
    subType?: 'visual' | 'collision',
    helperKind?: ViewerHelperKind,
  ) => void;
  onActiveJointChange?: (jointName: string | null) => void;
  onJointAnglesChange?: (jointAngles: Record<string, number>) => void;
}
