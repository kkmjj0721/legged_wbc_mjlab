import {
  DEFAULT_DISPLAY_THICKNESS_SCALE,
  ROTATE_AUXILIARY_NAMES,
  TRANSLATE_AUXILIARY_NAMES,
  getGizmoRoot,
  type UnifiedTransformDisplayStyle,
} from './gizmoCore';
import {
  enforceGizmoRenderPriority,
  getDisplayBehaviorPatchKey,
  removeHandlesByNames,
} from './displayPatchShared';
import { applyRotateDisplayPatches } from './rotateDisplayPatches';
import { applyTranslateDisplayPatches } from './translateDisplayPatches';

export const patchDisplayBehavior = (
  controls: any,
  displayStyle: UnifiedTransformDisplayStyle,
  thicknessScale: number,
  {
    leaveTranslateRingGap = false,
  }: {
    leaveTranslateRingGap?: boolean;
  } = {}
) => {
  const gizmo = getGizmoRoot(controls);
  if (!gizmo || displayStyle !== 'thick-primary') return;

  const normalizedThicknessScale = Number.isFinite(thicknessScale) && thicknessScale > 0
    ? thicknessScale
    : DEFAULT_DISPLAY_THICKNESS_SCALE;
  const patchKey = `${getDisplayBehaviorPatchKey(normalizedThicknessScale)}:${leaveTranslateRingGap ? 'gap' : 'solid'}`;
  if (gizmo.userData?.urdfDisplayBehaviorVersion === patchKey) return;

  removeHandlesByNames(gizmo.gizmo?.translate, TRANSLATE_AUXILIARY_NAMES);
  removeHandlesByNames(gizmo.picker?.translate, TRANSLATE_AUXILIARY_NAMES);
  removeHandlesByNames(gizmo.gizmo?.rotate, ROTATE_AUXILIARY_NAMES);
  removeHandlesByNames(gizmo.picker?.rotate, ROTATE_AUXILIARY_NAMES);

  applyTranslateDisplayPatches(gizmo, normalizedThicknessScale, {
    leaveRingGap: leaveTranslateRingGap,
  });
  applyRotateDisplayPatches(gizmo, normalizedThicknessScale);

  enforceGizmoRenderPriority(gizmo);
  gizmo.userData.urdfDisplayBehaviorVersion = patchKey;
};
