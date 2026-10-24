import type { LoadingProgressMode } from '@/types';

export type RobotLoadingPhase =
  | 'preparing-scene'
  | 'streaming-meshes'
  | 'finalizing-scene'
  | 'ready';

export type UsdLoadingPhase =
  | 'checking-path'
  | 'preloading-dependencies'
  | 'initializing-renderer'
  | 'streaming-meshes'
  | 'applying-stage-fixes'
  | 'resolving-metadata'
  | 'finalizing-scene'
  | 'ready';

export interface UsdLoadingProgress {
  phase: UsdLoadingPhase;
  message?: string | null;
  progressMode?: LoadingProgressMode | null;
  progressPercent?: number | null;
  loadedCount?: number | null;
  totalCount?: number | null;
}

export type ViewerLoadingPhase = RobotLoadingPhase | UsdLoadingPhase;

export interface ViewerDocumentLoadEvent {
  status: 'loading' | 'ready' | 'error';
  phase?: ViewerLoadingPhase | null;
  message?: string | null;
  progressMode?: LoadingProgressMode | null;
  progressPercent?: number | null;
  loadedCount?: number | null;
  totalCount?: number | null;
  error?: string | null;
}

export type UsdLoadingPhaseLabels = Record<Exclude<UsdLoadingPhase, 'ready'>, string>;
