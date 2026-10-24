import type { UniversalOwner, VisibleControlHit } from './gizmoCore';

export const hasHoveredHandle = (controls: any): boolean =>
  typeof controls?.axis === 'string' && controls.axis.length > 0;

export const preferVisibleControlHit = (
  currentHit: VisibleControlHit | null,
  nextHit: VisibleControlHit,
  previousOwner?: UniversalOwner,
) => {
  if (!currentHit) return nextHit;

  const renderOrderDelta = nextHit.renderOrder - currentHit.renderOrder;
  if (Math.abs(renderOrderDelta) > 1e-6) {
    return renderOrderDelta > 0 ? nextHit : currentHit;
  }

  const distanceDelta = currentHit.distance - nextHit.distance;
  if (Math.abs(distanceDelta) > 1e-6) {
    return distanceDelta > 0 ? nextHit : currentHit;
  }

  const scoreDelta = currentHit.score - nextHit.score;
  if (Math.abs(scoreDelta) > 1e-6) {
    return scoreDelta > 0 ? nextHit : currentHit;
  }

  if (previousOwner) {
    if (nextHit.owner === previousOwner && currentHit.owner !== previousOwner) {
      return nextHit;
    }
    if (currentHit.owner === previousOwner && nextHit.owner !== previousOwner) {
      return currentHit;
    }
  }

  return currentHit;
};

export const resolvePreferredVisibleOwner = (
  translateHit: VisibleControlHit | null,
  rotateHit: VisibleControlHit | null,
  previousOwner: UniversalOwner,
): UniversalOwner => {
  if (!translateHit && !rotateHit) return null;
  if (!translateHit) return 'rotate';
  if (!rotateHit) return 'translate';

  return preferVisibleControlHit(translateHit, rotateHit, previousOwner).owner;
};

export const resolvePreferredUniversalOwner = ({
  translateHovered,
  rotateHovered,
  translateHit,
  rotateHit,
  previousOwner,
}: {
  translateHovered: boolean;
  rotateHovered: boolean;
  translateHit: VisibleControlHit | null;
  rotateHit: VisibleControlHit | null;
  previousOwner: UniversalOwner;
}): UniversalOwner => {
  const visibleOwner = resolvePreferredVisibleOwner(translateHit, rotateHit, previousOwner);
  if (visibleOwner) {
    return visibleOwner;
  }

  if (translateHovered !== rotateHovered) {
    return translateHovered ? 'translate' : 'rotate';
  }

  return null;
};

export const resolveUniversalOwner = (
  translateControls: any,
  rotateControls: any,
  pointerOwner: UniversalOwner,
): UniversalOwner => {
  if (translateControls.dragging) return 'translate';
  if (rotateControls.dragging) return 'rotate';

  if (pointerOwner === 'rotate') return 'rotate';
  if (pointerOwner === 'translate') return 'translate';

  return null;
};

export const forceReleaseTransformControl = (controls: any): boolean => {
  if (!controls) return false;

  const wasActive = Boolean(controls.dragging) || controls.axis !== null;
  if (!wasActive) return false;

  if (typeof controls.pointerUp === 'function') {
    controls.pointerUp({ button: 0 });
  } else {
    controls.dragging = false;
    controls.axis = null;
  }

  return true;
};
