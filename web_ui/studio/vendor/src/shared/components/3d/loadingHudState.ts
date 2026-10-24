import type { LoadingProgressMode } from '@/types';

export interface LoadingHudStateInput {
  phase?: string | null;
  progressMode?: LoadingProgressMode | null;
  loadedCount?: number | null;
  totalCount?: number | null;
  progressPercent?: number | null;
  fallbackDetail: string;
}

export interface LoadingHudState {
  detail: string;
  progress: number | null;
  progressMode: LoadingProgressMode;
  statusLabel: string | null;
}

export interface IndeterminateStreamingMeshProgressInput {
  phase?: string | null;
  loadedCount?: number | null;
  totalCount?: number | null;
}

export interface LoadingProgressLike {
  phase?: string | null;
  progressMode?: LoadingProgressMode | null;
  loadedCount?: number | null;
  totalCount?: number | null;
  progressPercent?: number | null;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function normalizePercent(value: number | null | undefined): number | null {
  if (!Number.isFinite(value)) {
    return null;
  }

  return clamp(Math.round(Number(value)), 0, 100);
}

function normalizeCount(value: number | null | undefined): number | null {
  if (!Number.isFinite(value)) {
    return null;
  }

  return Math.max(0, Math.round(Number(value)));
}

export function hasDeterminateLoadingCounts(
  loadedCount?: number | null,
  totalCount?: number | null,
): boolean {
  const safeLoadedCount = normalizeCount(loadedCount);
  const safeTotalCount = normalizeCount(totalCount);

  return safeLoadedCount !== null && safeTotalCount !== null && safeTotalCount > 0;
}

export function resolveLoadingProgressMode({
  phase,
  progressMode,
  loadedCount,
  totalCount,
  progressPercent,
}: LoadingProgressLike): LoadingProgressMode {
  if (progressMode === 'count' || progressMode === 'percent' || progressMode === 'indeterminate') {
    return progressMode;
  }

  if (phase === 'ready' && normalizePercent(progressPercent) !== null) {
    return 'percent';
  }

  if (hasDeterminateLoadingCounts(loadedCount, totalCount)) {
    return 'count';
  }

  return 'indeterminate';
}

export function normalizeLoadingProgress<T extends LoadingProgressLike>(
  progress: T,
): T & { progressMode: LoadingProgressMode } {
  const nextProgressMode = resolveLoadingProgressMode(progress);
  const safeTotalCount = normalizeCount(progress.totalCount);
  const safeLoadedCount = normalizeCount(progress.loadedCount);
  const clampedLoadedCount =
    safeLoadedCount !== null && safeTotalCount !== null
      ? clamp(safeLoadedCount, 0, safeTotalCount)
      : safeLoadedCount;

  return {
    ...progress,
    progressMode: nextProgressMode,
    progressPercent:
      nextProgressMode === 'percent' ? normalizePercent(progress.progressPercent) : null,
    loadedCount: nextProgressMode === 'count' ? clampedLoadedCount : null,
    totalCount: nextProgressMode === 'count' ? safeTotalCount : null,
  };
}

export function buildLoadingHudState({
  phase,
  progressMode,
  loadedCount,
  totalCount,
  progressPercent,
  fallbackDetail,
}: LoadingHudStateInput): LoadingHudState {
  const normalizedProgress = normalizeLoadingProgress({
    phase,
    progressMode,
    loadedCount,
    totalCount,
    progressPercent,
  });

  const safeTotalCount = normalizedProgress.totalCount ?? 0;
  const safeLoadedCount = normalizedProgress.loadedCount ?? 0;

  if (normalizedProgress.progressMode === 'count' && safeTotalCount > 0) {
    const normalizedLoadedCount = clamp(safeLoadedCount, 0, safeTotalCount);

    return {
      detail: `${normalizedLoadedCount} / ${safeTotalCount}`,
      progress: normalizedLoadedCount / safeTotalCount,
      progressMode: 'count',
      statusLabel: `${normalizedLoadedCount} / ${safeTotalCount}`,
    };
  }

  if (normalizedProgress.progressMode === 'percent') {
    const normalizedPercent = normalizedProgress.progressPercent ?? 0;

    return {
      detail: `${normalizedPercent}%`,
      progress: normalizedPercent / 100,
      progressMode: 'percent',
      statusLabel: `${normalizedPercent}%`,
    };
  }

  return {
    detail: fallbackDetail,
    progress: null,
    progressMode: 'indeterminate',
    statusLabel: null,
  };
}

export function shouldUseIndeterminateStreamingMeshProgress({
  phase,
  loadedCount,
  totalCount,
}: IndeterminateStreamingMeshProgressInput): boolean {
  if (phase !== 'streaming-meshes') {
    return false;
  }

  const safeTotalCount = Number.isFinite(totalCount)
    ? Math.max(0, Math.round(Number(totalCount)))
    : 0;
  const safeLoadedCount = Number.isFinite(loadedCount)
    ? Math.max(0, Math.round(Number(loadedCount)))
    : 0;

  return safeTotalCount > 0 && safeLoadedCount === 0;
}
