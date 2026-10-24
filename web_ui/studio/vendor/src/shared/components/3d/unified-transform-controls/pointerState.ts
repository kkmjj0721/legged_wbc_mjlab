export const hasControlPointerIntent = (controls: any) => {
  if (!controls) return false;

  if (controls.dragging === true) {
    return true;
  }

  if (typeof controls.axis === 'string' && controls.axis.length > 0) {
    return true;
  }

  const cachedAxis = controls.userData?.urdfLastVisibleAxisHit?.axis;
  return typeof cachedAxis === 'string' && cachedAxis.length > 0;
};

export const clearControlPointerState = (controls: any) => {
  if (!controls) return;

  if (!controls.dragging && controls.axis !== null) {
    controls.axis = null;
  }

  if (
    controls.userData &&
    Object.prototype.hasOwnProperty.call(controls.userData, 'urdfLastVisibleAxisHit')
  ) {
    delete controls.userData.urdfLastVisibleAxisHit;
  }
};
