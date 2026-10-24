import * as THREE from 'three';
// Source of truth for coplanar material stabilization.
// Shared by the app runtime and the browser regression viewer so both render
// Unitree USD assets with identical subset stacking behavior.
const POSITION_KEY_DECIMALS = 6;
const COPLANAR_OFFSET_FACTOR = -2;
const COPLANAR_OFFSET_UNITS = -2;
const COPLANAR_OFFSET_FLAG = '__urdfStudioCoplanarOffset';
const COPLANAR_OFFSET_STACK_INDEX_KEY = '__urdfStudioCoplanarOffsetStackIndex';
const NEAR_COPLANAR_NORMAL_DOT = 0.999;
const NEAR_COPLANAR_PLANE_DISTANCE = 1e-3;
const NEAR_COPLANAR_CENTROID_DISTANCE = 1e-2;
const NEAR_COPLANAR_CENTROID_DISTANCE_SQ = NEAR_COPLANAR_CENTROID_DISTANCE ** 2;
const NEAR_COPLANAR_MIN_AREA_RATIO = 0.2;
const TRIANGLE_AREA_EPSILON = 1e-9;
const DOMINANT_MATERIAL_TRIANGLE_COUNT_RATIO = 4;
const DOMINANT_MATERIAL_TRIANGLE_COUNT_SHARE = 0.75;
const EMPTY_ANALYSIS = Object.freeze({
    adjustedMaterialIndices: Object.freeze([]),
    groupStackAssignments: Object.freeze([]),
    duplicateTriangleCount: 0,
    nearCoplanarTriangleCount: 0,
});
const geometryAnalysisCache = new WeakMap();
const _geometryCenter = new THREE.Vector3();
const isBufferGeometryLike = (geometry) => Boolean(geometry
    && typeof geometry === 'object'
    && Array.isArray(geometry.groups)
    && typeof geometry.getAttribute === 'function'
    && typeof geometry.clearGroups === 'function');
const toFiniteGroupValue = (value) => {
    if (!Number.isFinite(value) || value < 0) {
        return 0;
    }
    return Math.floor(value);
};
const buildGroupsSignature = (groups) => groups
    .map((group) => `${toFiniteGroupValue(group.start)}:${toFiniteGroupValue(group.count)}:${toFiniteGroupValue(group.materialIndex ?? 0)}`)
    .join('|');
const getTriangleVertexIndex = (indexArray, start, offset) => {
    const pointer = start + offset;
    return indexArray ? Number(indexArray[pointer] ?? 0) : pointer;
};
const buildVertexKey = (positionArray, vertexIndex, itemSize) => {
    const start = vertexIndex * itemSize;
    const x = Number(positionArray[start] ?? 0).toFixed(POSITION_KEY_DECIMALS);
    const y = Number(positionArray[start + 1] ?? 0).toFixed(POSITION_KEY_DECIMALS);
    const z = Number(positionArray[start + 2] ?? 0).toFixed(POSITION_KEY_DECIMALS);
    return `${x},${y},${z}`;
};
const buildTriangleKey = (positionArray, itemSize, indexArray, start, offset) => {
    const vertices = [
        buildVertexKey(positionArray, getTriangleVertexIndex(indexArray, start, offset), itemSize),
        buildVertexKey(positionArray, getTriangleVertexIndex(indexArray, start, offset + 1), itemSize),
        buildVertexKey(positionArray, getTriangleVertexIndex(indexArray, start, offset + 2), itemSize),
    ];
    vertices.sort();
    return vertices.join('|');
};
const getPositionComponent = (positionArray, vertexIndex, itemSize, axisOffset) => Number(positionArray[(vertexIndex * itemSize) + axisOffset] ?? 0);
const getTriangleBucketCoordinate = (value) => Math.floor(value / NEAR_COPLANAR_CENTROID_DISTANCE);
const buildTriangleBucketKey = (bucketX, bucketY, bucketZ) => `${bucketX}:${bucketY}:${bucketZ}`;
const buildTriangleDescriptor = (positionArray, itemSize, indexArray, start, offset, groupIndex, materialIndex, triangleKey) => {
    const vertexIndexA = getTriangleVertexIndex(indexArray, start, offset);
    const vertexIndexB = getTriangleVertexIndex(indexArray, start, offset + 1);
    const vertexIndexC = getTriangleVertexIndex(indexArray, start, offset + 2);
    const ax = getPositionComponent(positionArray, vertexIndexA, itemSize, 0);
    const ay = getPositionComponent(positionArray, vertexIndexA, itemSize, 1);
    const az = getPositionComponent(positionArray, vertexIndexA, itemSize, 2);
    const bx = getPositionComponent(positionArray, vertexIndexB, itemSize, 0);
    const by = getPositionComponent(positionArray, vertexIndexB, itemSize, 1);
    const bz = getPositionComponent(positionArray, vertexIndexB, itemSize, 2);
    const cx = getPositionComponent(positionArray, vertexIndexC, itemSize, 0);
    const cy = getPositionComponent(positionArray, vertexIndexC, itemSize, 1);
    const cz = getPositionComponent(positionArray, vertexIndexC, itemSize, 2);
    const edgeABX = bx - ax;
    const edgeABY = by - ay;
    const edgeABZ = bz - az;
    const edgeACX = cx - ax;
    const edgeACY = cy - ay;
    const edgeACZ = cz - az;
    const normalX = (edgeABY * edgeACZ) - (edgeABZ * edgeACY);
    const normalY = (edgeABZ * edgeACX) - (edgeABX * edgeACZ);
    const normalZ = (edgeABX * edgeACY) - (edgeABY * edgeACX);
    const normalLength = Math.hypot(normalX, normalY, normalZ);
    if (normalLength <= TRIANGLE_AREA_EPSILON) {
        return null;
    }
    const inverseNormalLength = 1 / normalLength;
    const normalizedNormalX = normalX * inverseNormalLength;
    const normalizedNormalY = normalY * inverseNormalLength;
    const normalizedNormalZ = normalZ * inverseNormalLength;
    const centroidX = (ax + bx + cx) / 3;
    const centroidY = (ay + by + cy) / 3;
    const centroidZ = (az + bz + cz) / 3;
    return {
        area: normalLength * 0.5,
        centroidX,
        centroidY,
        centroidZ,
        groupIndex,
        materialIndex,
        normalX: normalizedNormalX,
        normalY: normalizedNormalY,
        normalZ: normalizedNormalZ,
        planeConstant: (normalizedNormalX * centroidX) + (normalizedNormalY * centroidY) + (normalizedNormalZ * centroidZ),
        triangleKey,
    };
};
const isNearCoplanarTrianglePair = (triangle, candidate) => {
    const normalDot = (triangle.normalX * candidate.normalX)
        + (triangle.normalY * candidate.normalY)
        + (triangle.normalZ * candidate.normalZ);
    if (Math.abs(normalDot) < NEAR_COPLANAR_NORMAL_DOT) {
        return false;
    }
    const alignedCandidatePlaneConstant = normalDot >= 0
        ? candidate.planeConstant
        : -candidate.planeConstant;
    if (Math.abs(triangle.planeConstant - alignedCandidatePlaneConstant) > NEAR_COPLANAR_PLANE_DISTANCE) {
        return false;
    }
    const centroidDeltaX = triangle.centroidX - candidate.centroidX;
    const centroidDeltaY = triangle.centroidY - candidate.centroidY;
    const centroidDeltaZ = triangle.centroidZ - candidate.centroidZ;
    const centroidDistanceSq = (centroidDeltaX ** 2) + (centroidDeltaY ** 2) + (centroidDeltaZ ** 2);
    if (centroidDistanceSq > NEAR_COPLANAR_CENTROID_DISTANCE_SQ) {
        return false;
    }
    const largerArea = Math.max(triangle.area, candidate.area);
    const smallerArea = Math.min(triangle.area, candidate.area);
    return largerArea > 0 && (smallerArea / largerArea) >= NEAR_COPLANAR_MIN_AREA_RATIO;
};
const analyzeCoplanarMaterialGroups = (geometry) => {
    const groups = geometry.groups;
    const position = geometry.getAttribute('position');
    if (!position || position.itemSize < 3 || groups.length < 2) {
        return EMPTY_ANALYSIS;
    }
    geometry.computeBoundingBox();
    const geometryBoundingBox = geometry.boundingBox;
    if (!geometryBoundingBox) {
        return EMPTY_ANALYSIS;
    }
    geometryBoundingBox.getCenter(_geometryCenter);
    const signature = buildGroupsSignature(groups);
    const cachedBySignature = geometryAnalysisCache.get(geometry);
    const cached = cachedBySignature?.get(signature);
    if (cached) {
        return cached;
    }
    const positionArray = position.array;
    const indexArray = geometry.index?.array ?? null;
    const triangleOwners = new Map();
    const trianglesPerGroup = new Map();
    const centroidDistanceSumPerGroup = new Map();
    const measuredTrianglesPerGroup = new Map();
    const overlappingGroups = new Map();
    const nearCoplanarTriangleBuckets = new Map();
    const groupStackAssignments = [];
    let duplicateTriangleCount = 0;
    let nearCoplanarTriangleCount = 0;
    const connectOverlappingGroups = (groupIndices) => {
        if (groupIndices.length < 2) {
            return;
        }
        for (const groupIndex of groupIndices) {
            if (!overlappingGroups.has(groupIndex)) {
                overlappingGroups.set(groupIndex, new Set());
            }
        }
        const firstGroupIndex = groupIndices[0];
        const firstNeighbors = overlappingGroups.get(firstGroupIndex);
        for (let index = 1; index < groupIndices.length; index += 1) {
            const groupIndex = groupIndices[index];
            firstNeighbors.add(groupIndex);
            overlappingGroups.get(groupIndex).add(firstGroupIndex);
        }
    };
    for (const [groupIndex, group] of groups.entries()) {
        const materialIndex = toFiniteGroupValue(group.materialIndex ?? 0);
        const start = toFiniteGroupValue(group.start);
        const count = toFiniteGroupValue(group.count);
        const triangleCount = Math.floor(count / 3);
        if (triangleCount <= 0) {
            continue;
        }
        trianglesPerGroup.set(groupIndex, triangleCount);
        for (let offset = 0; offset + 2 < count; offset += 3) {
            const triangleKey = buildTriangleKey(positionArray, position.itemSize, indexArray, start, offset);
            const owners = triangleOwners.get(triangleKey);
            if (owners) {
                owners.add(groupIndex);
            }
            else {
                triangleOwners.set(triangleKey, new Set([groupIndex]));
            }
            const triangle = buildTriangleDescriptor(positionArray, position.itemSize, indexArray, start, offset, groupIndex, materialIndex, triangleKey);
            if (!triangle) {
                continue;
            }
            const centroidDistance = Math.hypot(triangle.centroidX - _geometryCenter.x, triangle.centroidY - _geometryCenter.y, triangle.centroidZ - _geometryCenter.z);
            centroidDistanceSumPerGroup.set(groupIndex, (centroidDistanceSumPerGroup.get(groupIndex) ?? 0) + centroidDistance);
            measuredTrianglesPerGroup.set(groupIndex, (measuredTrianglesPerGroup.get(groupIndex) ?? 0) + 1);
            const bucketX = getTriangleBucketCoordinate(triangle.centroidX);
            const bucketY = getTriangleBucketCoordinate(triangle.centroidY);
            const bucketZ = getTriangleBucketCoordinate(triangle.centroidZ);
            for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
                for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
                    for (let deltaZ = -1; deltaZ <= 1; deltaZ += 1) {
                        const bucketKey = buildTriangleBucketKey(bucketX + deltaX, bucketY + deltaY, bucketZ + deltaZ);
                        const candidates = nearCoplanarTriangleBuckets.get(bucketKey);
                        if (!candidates) {
                            continue;
                        }
                        for (const candidate of candidates) {
                            if (candidate.groupIndex === groupIndex || candidate.triangleKey === triangleKey) {
                                continue;
                            }
                            if (!isNearCoplanarTrianglePair(triangle, candidate)) {
                                continue;
                            }
                            nearCoplanarTriangleCount += 1;
                            connectOverlappingGroups([candidate.groupIndex, groupIndex]);
                        }
                    }
                }
            }
            const ownBucketKey = buildTriangleBucketKey(bucketX, bucketY, bucketZ);
            const bucketEntries = nearCoplanarTriangleBuckets.get(ownBucketKey);
            if (bucketEntries) {
                bucketEntries.push(triangle);
            }
            else {
                nearCoplanarTriangleBuckets.set(ownBucketKey, [triangle]);
            }
        }
    }
    for (const owners of triangleOwners.values()) {
        if (owners.size < 2) {
            continue;
        }
        duplicateTriangleCount += 1;
        const groupIndices = [...owners].sort((left, right) => left - right);
        connectOverlappingGroups(groupIndices);
    }
    const getGroupMaterialIndex = (groupIndex) => toFiniteGroupValue(groups[groupIndex]?.materialIndex ?? 0);
    const visitedGroups = new Set();
    const getAverageCentroidDistance = (groupIndex) => {
        const triangleCount = measuredTrianglesPerGroup.get(groupIndex) ?? 0;
        if (triangleCount <= 0) {
            return Number.POSITIVE_INFINITY;
        }
        return (centroidDistanceSumPerGroup.get(groupIndex) ?? 0) / triangleCount;
    };
    const getGroupTriangleCount = (groupIndex) => trianglesPerGroup.get(groupIndex) ?? 0;
    const pickDominantSurfaceGroupIndex = (componentGroupIndices) => {
        const rankedGroups = componentGroupIndices
            .map((candidateGroupIndex) => ({
            groupIndex: candidateGroupIndex,
            triangleCount: getGroupTriangleCount(candidateGroupIndex),
        }))
            .sort((left, right) => {
            if (left.triangleCount !== right.triangleCount) {
                return right.triangleCount - left.triangleCount;
            }
            return left.groupIndex - right.groupIndex;
        });
        const dominantGroup = rankedGroups[0];
        if (!dominantGroup || dominantGroup.triangleCount <= 0) {
            return null;
        }
        const secondTriangleCount = rankedGroups[1]?.triangleCount ?? 0;
        const totalTriangleCount = rankedGroups.reduce((sum, entry) => sum + entry.triangleCount, 0);
        const dominantTriangleShare = totalTriangleCount > 0
            ? dominantGroup.triangleCount / totalTriangleCount
            : 0;
        const dominatesByRatio = dominantGroup.triangleCount >= Math.max(1, secondTriangleCount) * DOMINANT_MATERIAL_TRIANGLE_COUNT_RATIO;
        const dominatesByShare = dominantTriangleShare >= DOMINANT_MATERIAL_TRIANGLE_COUNT_SHARE;
        // Large shell materials should remain the depth anchor when tiny decals/logos
        // happen to sit closer to the mesh centroid.
        if (dominatesByRatio || dominatesByShare) {
            return dominantGroup.groupIndex;
        }
        return null;
    };
    const pickInteriorAnchorGroupIndex = (componentGroupIndices) => componentGroupIndices.reduce((bestGroupIndex, groupIndex) => {
        const bestAverageCentroidDistance = getAverageCentroidDistance(bestGroupIndex);
        const averageCentroidDistance = getAverageCentroidDistance(groupIndex);
        if (averageCentroidDistance < (bestAverageCentroidDistance - 1e-6)) {
            return groupIndex;
        }
        if (Math.abs(averageCentroidDistance - bestAverageCentroidDistance) > 1e-6) {
            return bestGroupIndex;
        }
        const bestTriangleCount = getGroupTriangleCount(bestGroupIndex);
        const triangleCount = getGroupTriangleCount(groupIndex);
        if (triangleCount > bestTriangleCount) {
            return groupIndex;
        }
        const bestMaterialIndex = getGroupMaterialIndex(bestGroupIndex);
        const materialIndex = getGroupMaterialIndex(groupIndex);
        if (triangleCount === bestTriangleCount && materialIndex < bestMaterialIndex) {
            return groupIndex;
        }
        if (triangleCount === bestTriangleCount && materialIndex === bestMaterialIndex && groupIndex < bestGroupIndex) {
            return groupIndex;
        }
        return bestGroupIndex;
    }, componentGroupIndices[0]);
    const compareExteriorGroupIndices = (leftGroupIndex, rightGroupIndex) => {
        const leftAverageCentroidDistance = getAverageCentroidDistance(leftGroupIndex);
        const rightAverageCentroidDistance = getAverageCentroidDistance(rightGroupIndex);
        if (Math.abs(leftAverageCentroidDistance - rightAverageCentroidDistance) > 1e-6) {
            return leftAverageCentroidDistance - rightAverageCentroidDistance;
        }
        const leftTriangleCount = getGroupTriangleCount(leftGroupIndex);
        const rightTriangleCount = getGroupTriangleCount(rightGroupIndex);
        if (leftTriangleCount !== rightTriangleCount) {
            return leftTriangleCount - rightTriangleCount;
        }
        const leftMaterialIndex = getGroupMaterialIndex(leftGroupIndex);
        const rightMaterialIndex = getGroupMaterialIndex(rightGroupIndex);
        if (leftMaterialIndex !== rightMaterialIndex) {
            return leftMaterialIndex - rightMaterialIndex;
        }
        return leftGroupIndex - rightGroupIndex;
    };
    const compareDominantShellExteriorGroupIndices = (leftGroupIndex, rightGroupIndex) => {
        const leftTriangleCount = getGroupTriangleCount(leftGroupIndex);
        const rightTriangleCount = getGroupTriangleCount(rightGroupIndex);
        if (leftTriangleCount !== rightTriangleCount) {
            return leftTriangleCount - rightTriangleCount;
        }
        const leftAverageCentroidDistance = getAverageCentroidDistance(leftGroupIndex);
        const rightAverageCentroidDistance = getAverageCentroidDistance(rightGroupIndex);
        if (Math.abs(leftAverageCentroidDistance - rightAverageCentroidDistance) > 1e-6) {
            return leftAverageCentroidDistance - rightAverageCentroidDistance;
        }
        const leftMaterialIndex = getGroupMaterialIndex(leftGroupIndex);
        const rightMaterialIndex = getGroupMaterialIndex(rightGroupIndex);
        if (leftMaterialIndex !== rightMaterialIndex) {
            return leftMaterialIndex - rightMaterialIndex;
        }
        return leftGroupIndex - rightGroupIndex;
    };
    for (const groupIndex of overlappingGroups.keys()) {
        if (visitedGroups.has(groupIndex)) {
            continue;
        }
        const componentGroupIndices = [];
        const stack = [groupIndex];
        visitedGroups.add(groupIndex);
        while (stack.length > 0) {
            const currentGroupIndex = stack.pop();
            componentGroupIndices.push(currentGroupIndex);
            for (const neighborGroupIndex of overlappingGroups.get(currentGroupIndex) ?? []) {
                if (visitedGroups.has(neighborGroupIndex)) {
                    continue;
                }
                visitedGroups.add(neighborGroupIndex);
                stack.push(neighborGroupIndex);
            }
        }
        if (componentGroupIndices.length < 2) {
            continue;
        }
        const dominantSurfaceGroupIndex = pickDominantSurfaceGroupIndex(componentGroupIndices);
        const anchorGroupIndex = dominantSurfaceGroupIndex ?? pickInteriorAnchorGroupIndex(componentGroupIndices);
        const exteriorGroupIndices = componentGroupIndices
            .filter((candidateGroupIndex) => candidateGroupIndex !== anchorGroupIndex)
            .sort(dominantSurfaceGroupIndex !== null
            ? compareDominantShellExteriorGroupIndices
            : compareExteriorGroupIndices);
        exteriorGroupIndices.forEach((candidateGroupIndex, index) => {
            groupStackAssignments.push({
                groupIndex: candidateGroupIndex,
                materialIndex: getGroupMaterialIndex(candidateGroupIndex),
                stackIndex: index + 1,
            });
        });
    }
    const analysis = {
        adjustedMaterialIndices: [...new Set(groupStackAssignments.map(({ materialIndex }) => materialIndex))]
            .sort((left, right) => left - right),
        groupStackAssignments: groupStackAssignments.slice().sort((left, right) => left.groupIndex - right.groupIndex),
        duplicateTriangleCount,
        nearCoplanarTriangleCount,
    };
    const nextCache = cachedBySignature ?? new Map();
    nextCache.set(signature, analysis);
    geometryAnalysisCache.set(geometry, nextCache);
    return analysis;
};
const getCoplanarOffsetFactorForStackIndex = (stackIndex) => Math.min(COPLANAR_OFFSET_FACTOR, -(stackIndex + 1));
const getCoplanarOffsetUnitsForStackIndex = (stackIndex) => Math.min(COPLANAR_OFFSET_UNITS, -(stackIndex * 2));
export const cloneMaterialWithCoplanarOffset = (material, stackIndex = 1) => {
    const normalizedStackIndex = Math.max(1, Math.floor(Number(stackIndex) || 1));
    const alreadyAdjusted = material.userData?.[COPLANAR_OFFSET_FLAG] === true;
    const existingStackIndex = Math.max(1, Math.floor(Number(material.userData?.[COPLANAR_OFFSET_STACK_INDEX_KEY]) || 1));
    const target = alreadyAdjusted && existingStackIndex === normalizedStackIndex
        ? material
        : material.clone();
    const nextUserData = { ...(target.userData ?? {}) };
    nextUserData[COPLANAR_OFFSET_FLAG] = true;
    nextUserData[COPLANAR_OFFSET_STACK_INDEX_KEY] = normalizedStackIndex;
    target.userData = nextUserData;
    target.polygonOffset = true;
    target.polygonOffsetFactor = Math.min(Number(target.polygonOffsetFactor) || 0, getCoplanarOffsetFactorForStackIndex(normalizedStackIndex));
    target.polygonOffsetUnits = Math.min(Number(target.polygonOffsetUnits) || 0, getCoplanarOffsetUnitsForStackIndex(normalizedStackIndex));
    target.needsUpdate = true;
    return target;
};
const ensureOffsetMaterial = (material, stackIndex = 1) => cloneMaterialWithCoplanarOffset(material, stackIndex);
export const mitigateCoplanarMaterialZFighting = (mesh) => {
    const geometry = mesh.geometry;
    if (!isBufferGeometryLike(geometry)) {
        return {
            ...EMPTY_ANALYSIS,
            adjustedMaterialCount: 0,
        };
    }
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    if (materials.length < 2 || geometry.groups.length < 2) {
        return {
            ...EMPTY_ANALYSIS,
            adjustedMaterialCount: 0,
        };
    }
    const analysis = analyzeCoplanarMaterialGroups(geometry);
    if (analysis.groupStackAssignments.length === 0) {
        return {
            ...analysis,
            adjustedMaterialCount: 0,
        };
    }
    const nextMaterials = Array.isArray(mesh.material) ? mesh.material.slice() : materials.slice();
    const groupAssignmentsByGroupIndex = new Map(analysis.groupStackAssignments.map((assignment) => [assignment.groupIndex, assignment]));
    const groupsByMaterialIndex = new Map();
    geometry.groups.forEach((group, groupIndex) => {
        const materialIndex = toFiniteGroupValue(group.materialIndex ?? 0);
        const groupIndices = groupsByMaterialIndex.get(materialIndex);
        if (groupIndices) {
            groupIndices.push(groupIndex);
        }
        else {
            groupsByMaterialIndex.set(materialIndex, [groupIndex]);
        }
    });
    const remappedMaterialIndexByGroupIndex = new Map();
    const sharedCloneMaterialIndexByKey = new Map();
    let adjustedMaterialCount = 0;
    for (const materialIndex of analysis.adjustedMaterialIndices) {
        const material = nextMaterials[materialIndex];
        if (!material) {
            continue;
        }
        const groupIndices = groupsByMaterialIndex.get(materialIndex) ?? [];
        const adjustedAssignments = groupIndices
            .map((groupIndex) => groupAssignmentsByGroupIndex.get(groupIndex))
            .filter((assignment) => Boolean(assignment));
        if (adjustedAssignments.length === 0) {
            continue;
        }
        const uniqueStackIndices = [...new Set(adjustedAssignments.map(({ stackIndex }) => stackIndex))].sort((left, right) => left - right);
        const unadjustedGroupCount = groupIndices.length - adjustedAssignments.length;
        const canAdjustMaterialInPlace = unadjustedGroupCount === 0 && uniqueStackIndices.length === 1;
        if (canAdjustMaterialInPlace) {
            const nextMaterial = ensureOffsetMaterial(material, uniqueStackIndices[0] ?? 1);
            if (nextMaterial !== material) {
                nextMaterials[materialIndex] = nextMaterial;
            }
            adjustedMaterialCount += 1;
            continue;
        }
        for (const stackIndex of uniqueStackIndices) {
            const cloneKey = `${materialIndex}:${stackIndex}`;
            let remappedMaterialIndex = sharedCloneMaterialIndexByKey.get(cloneKey);
            if (remappedMaterialIndex === undefined) {
                remappedMaterialIndex = nextMaterials.length;
                nextMaterials.push(ensureOffsetMaterial(material, stackIndex));
                sharedCloneMaterialIndexByKey.set(cloneKey, remappedMaterialIndex);
                adjustedMaterialCount += 1;
            }
            for (const assignment of adjustedAssignments) {
                if (assignment.stackIndex === stackIndex) {
                    remappedMaterialIndexByGroupIndex.set(assignment.groupIndex, remappedMaterialIndex);
                }
            }
        }
    }
    if (remappedMaterialIndexByGroupIndex.size > 0) {
        const nextGeometry = geometry.clone();
        nextGeometry.clearGroups();
        geometry.groups.forEach((group, groupIndex) => {
            nextGeometry.addGroup(toFiniteGroupValue(group.start), toFiniteGroupValue(group.count), remappedMaterialIndexByGroupIndex.get(groupIndex) ?? toFiniteGroupValue(group.materialIndex ?? 0));
        });
        mesh.geometry = nextGeometry;
    }
    if (adjustedMaterialCount > 0) {
        mesh.material = Array.isArray(mesh.material) ? nextMaterials : nextMaterials[0];
    }
    return {
        ...analysis,
        adjustedMaterialCount,
    };
};
export const isCoplanarOffsetMaterial = (material) => material?.userData?.[COPLANAR_OFFSET_FLAG] === true;
export const markMaterialAsCoplanarOffset = (material) => cloneMaterialWithCoplanarOffset(material, 1);
