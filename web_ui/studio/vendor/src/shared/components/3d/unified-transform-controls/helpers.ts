export {
  DEFAULT_DISPLAY_THICKNESS_SCALE,
  VISUALIZER_UNIFIED_GIZMO_SIZE,
  hasEnabledFlag,
  markGizmoObjects,
  resolveAttachedTransformControlObject,
} from './gizmoCore';

export { patchHoverBehavior } from './hoverBehavior';

export {
  forceReleaseTransformControl,
  hasHoveredHandle,
  resolvePreferredUniversalOwner,
  resolvePreferredVisibleOwner,
  resolveUniversalOwner,
} from './ownership';

export { patchDisplayBehavior } from './displayPatches';

export {
  patchVisibleHoverHitFallback,
  patchVisiblePointerDownFallback,
  resolveVisibleRotateHit,
  resolveVisibleTranslateHit,
} from './visibleHitTesting';

export type {
  SharedControlRef,
  UnifiedTransformControlsProps,
  UnifiedTransformDisplayStyle,
  UnifiedTransformHoverStyle,
  UnifiedTransformMode,
  UniversalOwner,
} from './gizmoCore';
