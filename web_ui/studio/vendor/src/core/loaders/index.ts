/**
 * Loaders Module
 * Provides mesh loading utilities for STL, DAE, OBJ, GLTF/GLB formats
 */

export {
    buildAssetIndex,
    findAssetByIndex,
    findAssetByPath,
    createLoadingManager,
    resolveManagedAssetUrl,
    postProcessColladaScene,
    createPlaceholderMesh,
    resetUnitDetection,
    createMeshLoader
} from './meshLoader';
export {
    markMaterialAsCoplanarOffset,
    mitigateCoplanarMaterialZFighting,
    isCoplanarOffsetMaterial,
    cloneMaterialWithCoplanarOffset,
} from './coplanarMaterialOffset';
export {
    applyColladaCoplanarMaterialFixups,
} from './colladaCoplanarMaterialFixups';
export {
    isLikelyThinSheetGeometry,
    prepareMeshSurfaceForSingleSidedRendering,
} from './meshSurfaceOrientation';

export type { AssetIndex } from './meshLoader';
export { bakeColladaRootTransformInPlace } from './colladaRootTransform';
export { cleanFilePath } from './pathNormalization';
export {
    buildColladaRootNormalizationHints,
    shouldNormalizeColladaRoot,
} from './colladaRootNormalization';
export type { ColladaRootNormalizationHints } from './colladaRootNormalization';
