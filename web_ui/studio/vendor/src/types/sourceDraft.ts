import type { RobotFile } from './robot';

export type ComponentSourceFormat = Extract<
  RobotFile['format'],
  'urdf' | 'mjcf' | 'usd' | 'xacro' | 'sdf'
>;

/** Editable source owned by exactly one canonical workspace component. */
export interface ComponentSourceDraft {
  componentId: string;
  format: ComponentSourceFormat;
  content: string;
  robotSnapshotHash: string;
}
