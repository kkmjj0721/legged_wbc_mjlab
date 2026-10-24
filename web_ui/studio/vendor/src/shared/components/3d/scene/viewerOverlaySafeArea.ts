import type { CSSProperties } from 'react';

export const WORKSPACE_OVERLAY_LEFT_INSET_VAR = '--workspace-overlay-left-inset';
export const WORKSPACE_OVERLAY_RIGHT_INSET_VAR = '--workspace-overlay-right-inset';
export const WORKSPACE_OVERLAY_EDGE_GAP_PX = 16;
export const WORKSPACE_OVERLAY_GIZMO_MARGIN_PX = 68;

export const WORKSPACE_OVERLAY_LEFT_EDGE_GAP = resolveWorkspaceOverlayInsetOffset(
  WORKSPACE_OVERLAY_LEFT_INSET_VAR,
  WORKSPACE_OVERLAY_EDGE_GAP_PX,
);
export const WORKSPACE_OVERLAY_RIGHT_EDGE_GAP = resolveWorkspaceOverlayInsetOffset(
  WORKSPACE_OVERLAY_RIGHT_INSET_VAR,
  WORKSPACE_OVERLAY_EDGE_GAP_PX,
);
export const DEFAULT_WORKSPACE_OVERLAY_GIZMO_MARGIN: WorkspaceOverlayGizmoMargin = [
  WORKSPACE_OVERLAY_GIZMO_MARGIN_PX,
  WORKSPACE_OVERLAY_GIZMO_MARGIN_PX,
];

export const VIEWER_CORNER_OVERLAY_CLASS_NAME =
  'pointer-events-none absolute inset-0 z-20 box-border flex min-w-0 items-end justify-end overflow-hidden py-4 pl-[calc(var(--workspace-overlay-left-inset,0px)+1rem)] pr-[calc(var(--workspace-overlay-right-inset,0px)+1rem)]';

export interface WorkspaceOverlaySafeAreaInput {
  leftCollapsed: boolean;
  propertyEditorWidth: number;
  rightCollapsed: boolean;
  treeSidebarWidth: number;
}

export type WorkspaceOverlaySafeAreaStyle = CSSProperties &
  Record<typeof WORKSPACE_OVERLAY_LEFT_INSET_VAR | typeof WORKSPACE_OVERLAY_RIGHT_INSET_VAR, string>;

export type WorkspaceOverlayGizmoMargin = [number, number];

export function resolveWorkspaceOverlayInsetOffset(
  insetVar: typeof WORKSPACE_OVERLAY_LEFT_INSET_VAR | typeof WORKSPACE_OVERLAY_RIGHT_INSET_VAR,
  offsetPx: number,
): string {
  const normalizedOffset = Number.isFinite(offsetPx) ? Math.round(offsetPx) : 0;
  if (normalizedOffset === 0) {
    return `var(${insetVar},0px)`;
  }

  const operator = normalizedOffset < 0 ? '-' : '+';
  return `calc(var(${insetVar},0px) ${operator} ${Math.abs(normalizedOffset)}px)`;
}

export function resolveWorkspaceOverlaySafeAreaStyle({
  leftCollapsed,
  propertyEditorWidth,
  rightCollapsed,
  treeSidebarWidth,
}: WorkspaceOverlaySafeAreaInput): WorkspaceOverlaySafeAreaStyle {
  return {
    [WORKSPACE_OVERLAY_LEFT_INSET_VAR]: `${leftCollapsed ? 0 : treeSidebarWidth}px`,
    [WORKSPACE_OVERLAY_RIGHT_INSET_VAR]: `${rightCollapsed ? 0 : propertyEditorWidth}px`,
  };
}

export function resolveWorkspaceOverlayGizmoMargin({
  propertyEditorWidth,
  rightCollapsed,
}: WorkspaceOverlaySafeAreaInput): WorkspaceOverlayGizmoMargin {
  return [
    (rightCollapsed ? 0 : propertyEditorWidth) + WORKSPACE_OVERLAY_GIZMO_MARGIN_PX,
    WORKSPACE_OVERLAY_GIZMO_MARGIN_PX,
  ];
}
