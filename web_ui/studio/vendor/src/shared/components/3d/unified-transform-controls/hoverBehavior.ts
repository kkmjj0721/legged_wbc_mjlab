import * as THREE from 'three';
import {
  AXIS_NAMES,
  TRANSLATE_GAP_BRIDGE_OPACITY,
  getGizmoRoot,
  getHandleMaterials,
  getPositiveScale,
  type UnifiedTransformHoverStyle,
} from './gizmoCore';

const FALLBACK_ACTIVE_AXIS_COLOR = new THREE.Color(0x0a84ff);
const RESOLVED_ACTIVE_AXIS_COLOR = new THREE.Color(0x0a84ff);
const TRANSLATE_AXIS_SCALE_EPSILON = 1e-8;

const resolveActiveAxisColor = () => {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return FALLBACK_ACTIVE_AXIS_COLOR;
  }

  const rootStyle = window.getComputedStyle(document.documentElement);
  const colorValue =
    rootStyle.getPropertyValue('--color-system-blue').trim()
    || rootStyle.getPropertyValue('--ui-accent').trim();

  if (!colorValue) {
    return FALLBACK_ACTIVE_AXIS_COLOR;
  }

  try {
    RESOLVED_ACTIVE_AXIS_COLOR.set(colorValue);
    return RESOLVED_ACTIVE_AXIS_COLOR;
  } catch {
    return FALLBACK_ACTIVE_AXIS_COLOR;
  }
};

const rememberBaseMaterialState = (material: THREE.Material & { color?: THREE.Color }) => {
  if (material.userData.urdfBaseOpacity === undefined) {
    material.userData.urdfBaseOpacity = material.opacity;
  }
  if (material.userData.urdfBaseTransparent === undefined) {
    material.userData.urdfBaseTransparent = material.transparent;
  }
  if (!material.userData.urdfBaseColor && material.color instanceof THREE.Color) {
    material.userData.urdfBaseColor = material.color.clone();
  }
};

const getBaseOpacity = (material: THREE.Material) =>
  typeof material.userData.urdfBaseOpacity === 'number'
    ? (material.userData.urdfBaseOpacity as number)
    : material.opacity;

const getBaseColor = (material: THREE.Material & { color?: THREE.Color }) => {
  if (material.userData.urdfBaseColor instanceof THREE.Color) {
    return material.userData.urdfBaseColor as THREE.Color;
  }
  return material.color instanceof THREE.Color ? material.color : null;
};

const getMaterialAxis = (handle: any, material: THREE.Material & { color?: THREE.Color }) => {
  if (AXIS_NAMES.has(handle?.userData?.urdfAxis)) {
    return handle.userData.urdfAxis as 'X' | 'Y' | 'Z';
  }

  if (AXIS_NAMES.has(handle?.name)) {
    return handle.name as 'X' | 'Y' | 'Z';
  }

  const color = getBaseColor(material);
  if (!color) return null;

  if (color.r > 0.5 && color.g < 0.4 && color.b < 0.4) return 'X';
  if (color.g > 0.5 && color.r < 0.4 && color.b < 0.4) return 'Y';
  if (color.b > 0.5 && color.r < 0.4 && color.g < 0.4) return 'Z';
  return null;
};

const applySingleAxisHoverAppearance = (controls: any) => {
  const gizmo = getGizmoRoot(controls);
  const mode = controls?.mode ?? gizmo?.mode;
  const activeAxisColor = resolveActiveAxisColor();
  const activeAxis =
    controls?.enabled && typeof controls.axis === 'string' && AXIS_NAMES.has(controls.axis)
      ? (controls.axis as 'X' | 'Y' | 'Z')
      : null;

  const handles = [
    ...(gizmo?.picker?.[mode]?.children ?? []).map((handle: THREE.Object3D) => ({
      handle,
      source: 'picker' as const,
    })),
    ...(gizmo?.gizmo?.[mode]?.children ?? []).map((handle: THREE.Object3D) => ({
      handle,
      source: 'gizmo' as const,
    })),
    ...(gizmo?.helper?.[mode]?.children ?? []).map((handle: THREE.Object3D) => ({
      handle,
      source: 'helper' as const,
    })),
  ];

  for (const { handle, source } of handles) {
    for (const material of getHandleMaterials(handle)) {
      if (!(material instanceof THREE.Material)) continue;

      rememberBaseMaterialState(material as THREE.Material & { color?: THREE.Color });

      const baseOpacity = getBaseOpacity(material);
      const baseColor = getBaseColor(material as THREE.Material & { color?: THREE.Color });

      material.opacity = baseOpacity;
      material.transparent = true;
      material.depthTest = false;
      material.depthWrite = false;

      if (baseColor && 'color' in material && material.color instanceof THREE.Color) {
        material.color.copy(baseColor);
      }

      if (source === 'picker') {
        material.opacity = 0;
        material.transparent = true;
        material.needsUpdate = true;
        continue;
      }

      const isGapBridge = Boolean(handle?.userData?.urdfTranslateGapBridge);
      if (isGapBridge) {
        const materialAxis = getMaterialAxis(handle, material as THREE.Material & { color?: THREE.Color });
        const isActiveBridge = Boolean(activeAxis && materialAxis === activeAxis);
        handle.visible = isActiveBridge;

        if (!isActiveBridge) {
          material.opacity = 0;
          material.transparent = true;
          material.needsUpdate = true;
          continue;
        }

        if ('color' in material && material.color instanceof THREE.Color) {
          material.color.copy(activeAxisColor);
        }
        material.opacity = TRANSLATE_GAP_BRIDGE_OPACITY;
        material.transparent = true;
        material.needsUpdate = true;
        continue;
      }

      if (!activeAxis) {
        material.needsUpdate = true;
        continue;
      }

      const materialAxis = getMaterialAxis(handle, material as THREE.Material & { color?: THREE.Color });
      if (materialAxis === activeAxis) {
        if ('color' in material && material.color instanceof THREE.Color) {
          material.color.copy(activeAxisColor);
        }
        material.opacity = 1;
        material.transparent = true;
        material.needsUpdate = true;
      } else {
        material.needsUpdate = true;
      }
    }
  }
};

const enforcePatchedVisibility = (controls: any) => {
  const gizmo = getGizmoRoot(controls);
  if (!gizmo) return;

  gizmo.traverse((node: THREE.Object3D) => {
    if (node.userData?.urdfHideStockAxisLine) {
      node.visible = false;
    }

    if (!node.userData?.urdfTranslateFixedVisible) {
      return;
    }

    const scaledNode = node as THREE.Object3D & { scale?: THREE.Vector3 };
    if (scaledNode.scale instanceof THREE.Vector3) {
      if (scaledNode.scale.lengthSq() > TRANSLATE_AXIS_SCALE_EPSILON) {
        const normalizedScale = getPositiveScale(scaledNode.scale);
        scaledNode.scale.copy(normalizedScale);
        node.userData.urdfTranslateStableScale = normalizedScale;
      } else if (node.userData.urdfTranslateStableScale instanceof THREE.Vector3) {
        const stableScale = node.userData.urdfTranslateStableScale as THREE.Vector3;
        scaledNode.scale.copy(getPositiveScale(stableScale));
      }
    }

    node.visible = true;
  });
};

const preRecordGizmoMaterialStates = (controls: any) => {
  // Pre-record the base material state of every visible handle before
  // TransformControls gets a chance to modify opacity/color on first hover.
  // Without this, `rememberBaseMaterialState` might capture a TC-modified color
  // (e.g. yellow hover highlight) as the "base", causing permanent visual glitches.
  const gizmo = getGizmoRoot(controls);
  if (!gizmo) return;

  const mode = controls?.mode;
  const groups = mode
    ? [gizmo?.gizmo?.[mode], gizmo?.picker?.[mode], gizmo?.helper?.[mode]]
    : [
        gizmo?.gizmo?.translate,
        gizmo?.gizmo?.rotate,
        gizmo?.gizmo?.scale,
        gizmo?.picker?.translate,
        gizmo?.picker?.rotate,
        gizmo?.picker?.scale,
      ];

  for (const group of groups) {
    if (!group) continue;
    (group as THREE.Object3D).traverse?.((node: THREE.Object3D) => {
      for (const material of getHandleMaterials(node as any)) {
        if (material instanceof THREE.Material) {
          rememberBaseMaterialState(material as THREE.Material & { color?: THREE.Color });
        }
      }
    });
  }
};

export const patchHoverBehavior = (controls: any, hoverStyle: UnifiedTransformHoverStyle) => {
  const gizmo = getGizmoRoot(controls);
  if (!gizmo || hoverStyle !== 'single-axis') return;
  if (gizmo.userData?.urdfHoverBehaviorPatched) return;

  // Record base states NOW, before TransformControls can mutate them.
  preRecordGizmoMaterialStates(controls);

  const originalUpdateMatrixWorld = gizmo.updateMatrixWorld.bind(gizmo);
  gizmo.updateMatrixWorld = (...args: any[]) => {
    const result = originalUpdateMatrixWorld(...args);
    enforcePatchedVisibility(controls);
    applySingleAxisHoverAppearance(controls);
    return result;
  };

  gizmo.userData.urdfHoverBehaviorPatched = true;
  enforcePatchedVisibility(controls);
  applySingleAxisHoverAppearance(controls);
};
