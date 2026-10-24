import * as THREE from 'three';

export const ignoreRaycast: THREE.Object3D['raycast'] = (
  _raycaster,
  _intersects,
) => undefined;
