import * as THREE from 'three';

const IDENTITY_MATRIX = new THREE.Matrix4();

function cloneAndBakeGeometry(object: THREE.Object3D, transform: THREE.Matrix4): void {
  if (!(object as THREE.Mesh).isMesh && !(object as THREE.Line).isLine && !(object as THREE.Points).isPoints) {
    return;
  }

  const renderable = object as THREE.Mesh | THREE.Line | THREE.Points;
  const geometry = renderable.geometry;
  if (!geometry) {
    return;
  }

  renderable.geometry = geometry.clone();
  renderable.geometry.applyMatrix4(transform);
  renderable.geometry.computeBoundingBox();
  renderable.geometry.computeBoundingSphere();
}

export function bakeColladaRootTransformInPlace(object: THREE.Object3D): void {
  object.updateMatrix();
  const rootTransform = object.matrix.clone();

  if (rootTransform.equals(IDENTITY_MATRIX)) {
    object.position.set(0, 0, 0);
    object.rotation.set(0, 0, 0);
    object.quaternion.identity();
    object.scale.set(1, 1, 1);
    object.updateMatrix();
    return;
  }

  cloneAndBakeGeometry(object, rootTransform);

  object.children.forEach((child) => {
    child.applyMatrix4(rootTransform);
  });

  object.position.set(0, 0, 0);
  object.rotation.set(0, 0, 0);
  object.quaternion.identity();
  object.scale.set(1, 1, 1);
  object.updateMatrix();
}
