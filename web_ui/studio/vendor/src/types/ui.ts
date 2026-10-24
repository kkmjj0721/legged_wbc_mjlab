/**
 * UI related types
 */

export type AppMode = 'editor';
export type Theme = 'light' | 'dark' | 'system';
export type DetailLinkTab = 'visual' | 'collision' | 'physics';
export type LoadingProgressMode = 'count' | 'percent' | 'indeterminate';

export type InteractionHelperKind =
  | 'center-of-mass'
  | 'inertia'
  | 'ik-handle'
  | 'origin-axes'
  | 'joint-axis';

export interface InteractionSelection {
  type: 'link' | 'joint' | 'tendon' | null;
  id: string | null;
  subType?: 'visual' | 'collision';
  objectIndex?: number;
  helperKind?: InteractionHelperKind;
  highlightObjectId?: number;
}
