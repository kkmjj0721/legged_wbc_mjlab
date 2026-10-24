import { useEffect, useState } from 'react';
import type { LoadingProgressMode } from '@/types';

interface LoadingHudProps {
  title: string;
  detail: string;
  progress: number | null;
  progressMode?: LoadingProgressMode | null;
  statusLabel?: string | null;
  stageLabel?: string | null;
  delayMs?: number;
}

export function LoadingHud({
  title,
  detail,
  progress,
  progressMode = null,
  statusLabel = null,
  stageLabel = null,
  delayMs = 300,
}: LoadingHudProps) {
  const [isVisible, setIsVisible] = useState(delayMs <= 0);

  useEffect(() => {
    if (delayMs <= 0) {
      setIsVisible(true);
      return undefined;
    }

    const timer = window.setTimeout(() => {
      setIsVisible(true);
    }, delayMs);

    return () => {
      window.clearTimeout(timer);
    };
  }, [delayMs]);

  if (!isVisible) {
    return null;
  }

  const resolvedProgressMode = progressMode ?? (progress === null ? 'indeterminate' : 'percent');
  const progressWidth = `${Math.round(Math.min(1, Math.max(0, progress ?? 0)) * 100)}%`;
  const normalizedDetail = detail.trim();
  const normalizedStatusLabel = statusLabel?.trim() ?? '';
  const normalizedStageLabel = stageLabel?.trim() ?? '';
  const shouldRenderDetail =
    normalizedDetail.length > 0 &&
    normalizedDetail !== normalizedStatusLabel &&
    normalizedDetail !== normalizedStageLabel;

  return (
    <div
      role="status"
      aria-live="polite"
      className="pointer-events-none min-w-0 rounded-2xl border border-border-black bg-panel-bg/95 px-3.5 py-3 shadow-xl backdrop-blur-sm"
      style={{
        boxSizing: 'border-box',
        maxWidth: '100%',
        width: 'min(280px, 100%)',
      }}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className="h-2 w-2 shrink-0 rounded-full bg-slider-accent motion-safe:animate-pulse"
            />
            <span className="truncate text-xs font-medium text-text-primary">{title}</span>
          </div>
          {stageLabel ? (
            <div className="mt-2 inline-flex max-w-full items-center rounded-full border border-border-black/70 bg-element-bg px-2 py-0.5 text-[10px] font-medium text-text-secondary">
              <span className="truncate">{stageLabel}</span>
            </div>
          ) : null}
        </div>
        {statusLabel ? (
          <div className="shrink-0 text-[11px] font-medium tabular-nums text-text-secondary">
            {statusLabel}
          </div>
        ) : null}
      </div>
      {shouldRenderDetail ? (
        <div className="mt-2 truncate text-[11px] font-medium text-text-secondary">
          {normalizedDetail}
        </div>
      ) : null}
      <div className="mt-3 h-1 overflow-hidden rounded-full bg-border-black/50">
        {resolvedProgressMode === 'indeterminate' ? (
          <div
            aria-hidden="true"
            className="h-full w-full rounded-full bg-[linear-gradient(90deg,rgba(0,136,255,0.12)_0%,rgba(0,136,255,0.4)_45%,rgba(0,136,255,0.12)_100%)] motion-safe:animate-pulse"
          />
        ) : (
          <div
            aria-hidden="true"
            className="h-full rounded-full bg-slider-accent transition-[width] duration-200 ease-out motion-reduce:transition-none"
            style={{ width: progressWidth }}
          />
        )}
      </div>
    </div>
  );
}
