import * as THREE from 'three';

import { mitigateCoplanarMaterialZFighting } from './coplanarMaterialOffset';

export const applyColladaCoplanarMaterialFixups = (root: THREE.Object3D) => {
    root.traverse((child) => {
        if (!(child as THREE.Mesh).isMesh) {
            return;
        }

        mitigateCoplanarMaterialZFighting(child as THREE.Mesh);
    });
};
