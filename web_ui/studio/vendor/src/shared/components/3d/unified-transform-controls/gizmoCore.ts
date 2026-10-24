import React from 'react';
import { TransformControls as DreiTransformControls } from '@react-three/drei';
import * as THREE from 'three';

type DreiTransformControlsProps = React.ComponentProps<typeof DreiTransformControls>;
export type TransformControlObjectTarget = NonNullable<DreiTransformControlsProps['object']>;

export type SharedControlRef = React.MutableRefObject<any | null> | React.RefObject<any | null>;

export type UnifiedTransformMode = 'translate' | 'rotate' | 'scale' | 'universal';
export type UnifiedTransformHoverStyle = 'stock' | 'single-axis';
export type UnifiedTransformDisplayStyle = 'stock' | 'thick-primary';

export const VISUALIZER_UNIFIED_GIZMO_SIZE = 0.78;
export const DEFAULT_DISPLAY_THICKNESS_SCALE = 1;

export interface UnifiedTransformControlsProps extends Omit<
  DreiTransformControlsProps,
  'mode' | 'object'
> {
  mode: UnifiedTransformMode;
  object?: DreiTransformControlsProps['object'];
  translateObject?: TransformControlObjectTarget;
  translateSpace?: DreiTransformControlsProps['space'];
  rotateRef?: SharedControlRef;
  rotateObject?: TransformControlObjectTarget;
  rotateSize?: number;
  rotateSpace?: DreiTransformControlsProps['space'];
  rotateEnabled?: boolean;
  onRotateChange?: DreiTransformControlsProps['onChange'];
  onDraggingChanged?: (event: { target?: unknown; value: boolean }) => void;
  enableUniversalPriority?: boolean;
  hoverStyle?: UnifiedTransformHoverStyle;
  displayStyle?: UnifiedTransformDisplayStyle;
  displayThicknessScale?: number;
  /**
   * Whether to render the free-rotate handles (screen-space E-ring + trackball)
   * in rotate mode. Defaults to true. Single-DOF joints pass false: a joint can
   * only rotate about its fixed axis, so these handles are meaningless and —
   * being the largest rings — visually swamp small robots.
   */
  showRotateFreeHandles?: boolean;
}

export type UniversalOwner = 'translate' | 'rotate' | null;
type ControlsWithEnabled = THREE.EventDispatcher & { enabled: boolean };

export type VisibleControlHit = {
  owner: Exclude<UniversalOwner, null>;
  axis: 'X' | 'Y' | 'Z';
  renderOrder: number;
  distance: number;
  score: number;
};

export const AXIS_NAMES = new Set(['X', 'Y', 'Z']);
export const THICK_TRANSLATE_SHAFT_RADIUS = 0.034;
export const THICK_TRANSLATE_TIP_RADIUS = 0.17;
export const TRANSLATE_ARROW_BASE_RADIUS = 0.038;
export const TRANSLATE_ARROW_LENGTH = 0.32;
export const THICK_ROTATE_ARC_RADIUS = 0.024;
export const THICK_TRANSLATE_PICKER_RADIUS = 0.075;
export const THICK_ROTATE_PICKER_ARC_RADIUS = 0.048;
export const TRANSLATE_CENTER_GAP = 0.12;
export const TRANSLATE_RING_INTERSECTION_RADIUS = 0.5;
export const TRANSLATE_RING_INTERSECTION_GAP = THICK_ROTATE_ARC_RADIUS * 3.6;
export const MIN_TRANSLATE_SHAFT_SEGMENT_LENGTH = 0.025;
export const TRANSLATE_ARROW_HANDLE_OFFSET = 0.94;
export const TRANSLATE_GAP_BRIDGE_DASH_COUNT = 3;
export const TRANSLATE_GAP_BRIDGE_RADIUS = 0.018;
export const TRANSLATE_GAP_BRIDGE_OPACITY = 0.9;
export const TRANSLATE_AUXILIARY_NAMES = new Set(['XY', 'YZ', 'XZ', 'XYZ']);
export const ROTATE_AUXILIARY_NAMES = new Set(['E', 'XYZE']);
export const HELPER_RENDER_ORDER = 999;
export const INERTIA_BOX_RENDER_ORDER = 9999;
export const GIZMO_BASE_RENDER_ORDER = 10000;
export const COM_VISUAL_RENDER_ORDER = 10001;
export const MJCF_SITE_FILL_RENDER_ORDER = 10002;
export const MJCF_SITE_WIREFRAME_RENDER_ORDER = 10003;
export const GIZMO_ARC_RENDER_ORDER = 10005;
export const IK_HANDLE_RENDER_ORDER = 10030;
export const DISPLAY_BEHAVIOR_PATCH_VERSION = 'thick-primary-v20';
const GIZMO_TAG_VERSION = 'interactive-only-v2';
export const AXIS_GEOMETRY_RADIAL_SEGMENTS = 16;
export const TRANSLATE_PICKER_START_OFFSET = 0.82;
export const TRANSLATE_PICKER_END_PADDING = 0.04;
export const ROTATE_ARC_GAP_SAMPLE_COUNT = 2;
export const ROTATE_ARC_BACK_OPACITY = 0.34;
export const ROTATE_ARC_FRONT_OPACITY = 0.94;
export const TRANSLATE_SCREEN_HOVER_SAMPLE_COUNT = 11;
export const TRANSLATE_SCREEN_HOVER_PADDING_PX = 1;
export const ROTATE_SCREEN_HOVER_PADDING_PX = 0.5;
export const ROTATE_SCREEN_HOVER_MIN_RADIUS_PX = 1.5;
export const ROTATE_BACK_HIT_THRESHOLD_SCALE = 0.55;
export const ROTATE_BACK_HIT_SCORE_PENALTY = 0.18;

export const hasEnabledFlag = (controls: unknown): controls is ControlsWithEnabled =>
  typeof controls === 'object' &&
  controls !== null &&
  'enabled' in controls &&
  typeof (controls as { enabled?: unknown }).enabled === 'boolean';

export const resolveTransformControlObject = (
  target: TransformControlObjectTarget | null | undefined,
): THREE.Object3D | null => {
  if (!target) return null;
  if (target instanceof THREE.Object3D) {
    return target;
  }

  const maybeRef = target as React.RefObject<THREE.Object3D | null>;
  return maybeRef.current instanceof THREE.Object3D ? maybeRef.current : null;
};

export const resolveAttachedTransformControlObject = (
  scene: THREE.Object3D | null | undefined,
  target: TransformControlObjectTarget | null | undefined,
) => {
  if (!scene) return null;

  const object = resolveTransformControlObject(target);
  if (!object) return null;

  let current: THREE.Object3D | null = object;
  while (current) {
    if (current === scene) {
      return object;
    }
    current = current.parent;
  }

  return null;
};

export const getGizmoRoot = (controls: any) => {
  if (!controls) return null;

  if (controls._gizmo?.isTransformControlsGizmo) {
    return controls._gizmo;
  }

  if (typeof controls.getHelper === 'function') {
    const helperRoot = controls.getHelper();
    const gizmoChild = helperRoot?.children?.find?.(
      (child: THREE.Object3D & { isTransformControlsGizmo?: boolean }) =>
        Boolean(child?.isTransformControlsGizmo),
    );
    if (gizmoChild) return gizmoChild;
  }

  if (controls._root?.children?.length) {
    const gizmoChild = controls._root.children.find(
      (child: THREE.Object3D & { isTransformControlsGizmo?: boolean }) =>
        Boolean(child?.isTransformControlsGizmo),
    );
    if (gizmoChild) return gizmoChild;
  }

  return controls?.children?.[0] ?? controls?.gizmo ?? null;
};

export const getHandleMaterials = (handle: any) => {
  const material = handle?.material;
  if (!material) return [] as THREE.Material[];
  return Array.isArray(material) ? material : [material];
};

export const getPositiveScale = (scale: THREE.Vector3) =>
  new THREE.Vector3(Math.abs(scale.x), Math.abs(scale.y), Math.abs(scale.z));

const tagGizmoBranch = (branch: unknown) => {
  if (!branch) return;

  if (
    branch instanceof THREE.Object3D ||
    typeof (branch as { traverse?: unknown }).traverse === 'function'
  ) {
    (branch as THREE.Object3D).traverse((node: THREE.Object3D) => {
      node.userData = {
        ...node.userData,
        isGizmo: true,
      };
    });
    return;
  }

  if (typeof branch === 'object') {
    Object.values(branch as Record<string, unknown>).forEach(tagGizmoBranch);
  }
};

export const markGizmoObjects = (controls: any) => {
  // The TransformControlsPlane is a direct child of the controls object (sibling
  // of the gizmo, not a descendant).  It carries a 1e5×1e5 unit plane geometry
  // used for raycasting but is never visually rendered (material.visible = false).
  // Tag it so it is excluded from scene bounds calculations regardless of whether
  // the gizmo branch tagging runs or is skipped by the version guard.
  if (controls?.plane) {
    controls.plane.userData = {
      ...controls.plane.userData,
      excludeFromSceneBounds: true,
    };
  }

  const gizmo = getGizmoRoot(controls);
  if (!gizmo || gizmo.userData?.urdfGizmoTagVersion === GIZMO_TAG_VERSION) return;

  gizmo.traverse((node: THREE.Object3D) => {
    if (!node.userData?.isGizmo) return;
    const nextUserData = { ...node.userData };
    delete nextUserData.isGizmo;
    node.userData = nextUserData;
  });

  tagGizmoBranch(gizmo.gizmo);
  tagGizmoBranch(gizmo.picker);
  gizmo.userData.urdfGizmoTagVersion = GIZMO_TAG_VERSION;
};
