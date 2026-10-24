import * as THREE from 'three';

const ORIENTATION_EPSILON = 1e-9;
const THIN_SHEET_RATIO = 0.02;
const BOUNDS_EPSILON = 1e-6;
const WELD_POSITION_PRECISION = 1e-8;
const SIGNED_VOLUME_EPSILON = 1e-12;

function swapAttributeElements(
  attribute: THREE.BufferAttribute | THREE.InterleavedBufferAttribute,
  leftIndex: number,
  rightIndex: number,
): void {
  const itemSize = attribute.itemSize;
  for (let component = 0; component < itemSize; component += 1) {
    const leftValue = attribute.getComponent(leftIndex, component);
    const rightValue = attribute.getComponent(rightIndex, component);
    attribute.setComponent(leftIndex, component, rightValue);
    attribute.setComponent(rightIndex, component, leftValue);
  }
}

function swapTriangleVerticesInNonIndexedGeometry(
  geometry: THREE.BufferGeometry,
  leftVertexIndex: number,
  rightVertexIndex: number,
): void {
  Object.values(geometry.attributes).forEach((attribute) => {
    if (!attribute) {
      return;
    }

    swapAttributeElements(attribute, leftVertexIndex, rightVertexIndex);
    attribute.needsUpdate = true;
  });
}

function swapTriangleVerticesInIndexedGeometry(
  geometry: THREE.BufferGeometry,
  triangleOffset: number,
): void {
  const index = geometry.getIndex();
  if (!index) {
    return;
  }

  const left = index.getX(triangleOffset + 1);
  const right = index.getX(triangleOffset + 2);
  index.setX(triangleOffset + 1, right);
  index.setX(triangleOffset + 2, left);
  index.needsUpdate = true;
}

interface TopologyFaceRecord {
  adjacentFaces: Array<{
    sameDirection: boolean;
    targetFaceIndex: number;
  }>;
  originalTriangleOffset: number;
  signedVolumeContribution: number;
  valid: boolean;
}

function buildWeldedVertexIndices(
  position: THREE.BufferAttribute | THREE.InterleavedBufferAttribute,
): { vertexIndices: number[]; weldedVertices: THREE.Vector3[] } {
  const weldedVertices: THREE.Vector3[] = [];
  const weldedIndexByKey = new Map<string, number>();
  const vertexIndices = new Array<number>(position.count);
  const vertex = new THREE.Vector3();

  for (let vertexIndex = 0; vertexIndex < position.count; vertexIndex += 1) {
    vertex.fromBufferAttribute(position, vertexIndex);
    const key = [
      Math.round(vertex.x / WELD_POSITION_PRECISION),
      Math.round(vertex.y / WELD_POSITION_PRECISION),
      Math.round(vertex.z / WELD_POSITION_PRECISION),
    ].join('|');

    let weldedIndex = weldedIndexByKey.get(key);
    if (weldedIndex == null) {
      weldedIndex = weldedVertices.length;
      weldedVertices.push(vertex.clone());
      weldedIndexByKey.set(key, weldedIndex);
    }

    vertexIndices[vertexIndex] = weldedIndex;
  }

  return {
    vertexIndices,
    weldedVertices,
  };
}

function computeSignedTriangleVolume(
  a: THREE.Vector3,
  b: THREE.Vector3,
  c: THREE.Vector3,
): number {
  return a.dot(new THREE.Vector3().crossVectors(b, c)) / 6;
}

function createFaceEdgeKey(left: number, right: number): string {
  return left < right ? `${left}:${right}` : `${right}:${left}`;
}

function buildTopologyFaceRecords(geometry: THREE.BufferGeometry): TopologyFaceRecord[] | null {
  const position = geometry.getAttribute('position');
  if (!(position instanceof THREE.BufferAttribute) && !(position instanceof THREE.InterleavedBufferAttribute)) {
    return null;
  }

  const index = geometry.getIndex();
  const triangleVertexCount = index ? index.count : position.count;
  if (!Number.isFinite(triangleVertexCount) || triangleVertexCount < 3) {
    return null;
  }

  const { vertexIndices, weldedVertices } = buildWeldedVertexIndices(position);
  const faceRecords: TopologyFaceRecord[] = [];
  const edgeUsages = new Map<string, Array<{ edge: [number, number]; faceIndex: number }>>();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const normal = new THREE.Vector3();

  for (let triangleOffset = 0; triangleOffset <= triangleVertexCount - 3; triangleOffset += 3) {
    const ia = index ? index.getX(triangleOffset) : triangleOffset;
    const ib = index ? index.getX(triangleOffset + 1) : triangleOffset + 1;
    const ic = index ? index.getX(triangleOffset + 2) : triangleOffset + 2;
    const weldedA = vertexIndices[ia]!;
    const weldedB = vertexIndices[ib]!;
    const weldedC = vertexIndices[ic]!;
    const a = weldedVertices[weldedA]!;
    const b = weldedVertices[weldedB]!;
    const c = weldedVertices[weldedC]!;

    let valid = weldedA !== weldedB && weldedB !== weldedC && weldedC !== weldedA;
    if (valid) {
      ab.subVectors(b, a);
      ac.subVectors(c, a);
      normal.crossVectors(ab, ac);
      valid = normal.lengthSq() > ORIENTATION_EPSILON;
    }

    const faceIndex = faceRecords.length;
    faceRecords.push({
      adjacentFaces: [],
      originalTriangleOffset: triangleOffset,
      signedVolumeContribution: valid ? computeSignedTriangleVolume(a, b, c) : 0,
      valid,
    });

    if (!valid) {
      continue;
    }

    const orientedEdges: Array<[number, number]> = [
      [weldedA, weldedB],
      [weldedB, weldedC],
      [weldedC, weldedA],
    ];

    orientedEdges.forEach((edge) => {
      const key = createFaceEdgeKey(edge[0], edge[1]);
      const existing = edgeUsages.get(key);
      if (existing) {
        existing.push({ edge, faceIndex });
      } else {
        edgeUsages.set(key, [{ edge, faceIndex }]);
      }
    });
  }

  edgeUsages.forEach((usages) => {
    if (usages.length !== 2) {
      usages.forEach(({ faceIndex }) => {
        faceRecords[faceIndex]!.valid = false;
      });
      return;
    }

    const [leftUsage, rightUsage] = usages;
    const sameDirection = leftUsage.edge[0] === rightUsage.edge[0]
      && leftUsage.edge[1] === rightUsage.edge[1];

    faceRecords[leftUsage.faceIndex]!.adjacentFaces.push({
      sameDirection,
      targetFaceIndex: rightUsage.faceIndex,
    });
    faceRecords[rightUsage.faceIndex]!.adjacentFaces.push({
      sameDirection,
      targetFaceIndex: leftUsage.faceIndex,
    });
  });

  return faceRecords;
}

function fixGeometryTriangleWinding(geometry: THREE.BufferGeometry): number {
  const faceRecords = buildTopologyFaceRecords(geometry);
  if (!faceRecords || faceRecords.length === 0) {
    return 0;
  }

  const triangleFlipStates = new Array<boolean | null>(faceRecords.length).fill(null);
  const visited = new Array<boolean>(faceRecords.length).fill(false);
  let flippedTriangleCount = 0;

  for (let startFaceIndex = 0; startFaceIndex < faceRecords.length; startFaceIndex += 1) {
    if (visited[startFaceIndex] || !faceRecords[startFaceIndex]!.valid) {
      continue;
    }

    const componentFaceIndices: number[] = [];
    const pending = [startFaceIndex];
    triangleFlipStates[startFaceIndex] = false;
    let componentSignedVolume = 0;
    let componentValid = true;

    while (pending.length > 0) {
      const faceIndex = pending.pop()!;
      if (visited[faceIndex]) {
        continue;
      }

      const faceRecord = faceRecords[faceIndex]!;
      if (!faceRecord.valid) {
        componentValid = false;
        continue;
      }

      visited[faceIndex] = true;
      componentFaceIndices.push(faceIndex);
      const currentFlipState = triangleFlipStates[faceIndex] ?? false;
      componentSignedVolume += currentFlipState
        ? -faceRecord.signedVolumeContribution
        : faceRecord.signedVolumeContribution;

      for (const adjacentFace of faceRecord.adjacentFaces) {
        const nextFlipState = adjacentFace.sameDirection
          ? !currentFlipState
          : currentFlipState;

        if (visited[adjacentFace.targetFaceIndex]) {
          if (triangleFlipStates[adjacentFace.targetFaceIndex] !== nextFlipState) {
            componentValid = false;
          }
          continue;
        }

        if (triangleFlipStates[adjacentFace.targetFaceIndex] != null) {
          if (triangleFlipStates[adjacentFace.targetFaceIndex] !== nextFlipState) {
            componentValid = false;
          }
        } else {
          triangleFlipStates[adjacentFace.targetFaceIndex] = nextFlipState;
          pending.push(adjacentFace.targetFaceIndex);
        }
      }
    }

    if (!componentValid) {
      componentFaceIndices.forEach((faceIndex) => {
        triangleFlipStates[faceIndex] = null;
      });
      continue;
    }

    const invertComponent = componentSignedVolume < -SIGNED_VOLUME_EPSILON;
    componentFaceIndices.forEach((faceIndex) => {
      const currentFlipState = triangleFlipStates[faceIndex] ?? false;
      const shouldFlip = invertComponent
        ? !currentFlipState
        : currentFlipState;
      triangleFlipStates[faceIndex] = shouldFlip;
      if (!shouldFlip) {
        return;
      }

      const triangleOffset = faceRecords[faceIndex]!.originalTriangleOffset;
      if (geometry.getIndex()) {
        swapTriangleVerticesInIndexedGeometry(geometry, triangleOffset);
      } else {
        swapTriangleVerticesInNonIndexedGeometry(geometry, triangleOffset + 1, triangleOffset + 2);
      }
      flippedTriangleCount += 1;
    });
  }

  if (flippedTriangleCount > 0) {
    geometry.deleteAttribute('normal');
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
  }

  return flippedTriangleCount;
}

export function isLikelyThinSheetGeometry(geometry: THREE.BufferGeometry): boolean {
  geometry.computeBoundingBox();
  const bounds = geometry.boundingBox;
  if (!bounds) {
    return false;
  }

  const size = bounds.getSize(new THREE.Vector3());
  const axes = [size.x, size.y, size.z].filter((value) => value > BOUNDS_EPSILON);
  if (axes.length < 2) {
    return false;
  }

  const minAxis = Math.min(...axes);
  const maxAxis = Math.max(...axes);
  if (maxAxis <= BOUNDS_EPSILON) {
    return false;
  }

  return (minAxis / maxAxis) <= THIN_SHEET_RATIO;
}

export interface MeshSurfacePreparationSummary {
  flippedTriangleCount: number;
  repairedMeshCount: number;
}

export function prepareMeshSurfaceForSingleSidedRendering(root: THREE.Object3D): MeshSurfacePreparationSummary {
  const summary: MeshSurfacePreparationSummary = {
    flippedTriangleCount: 0,
    repairedMeshCount: 0,
  };

  root.traverse((child: any) => {
    if (!child?.isMesh || !(child.geometry instanceof THREE.BufferGeometry)) {
      return;
    }

    const flippedTriangleCount = fixGeometryTriangleWinding(child.geometry);
    if (flippedTriangleCount > 0) {
      summary.flippedTriangleCount += flippedTriangleCount;
      summary.repairedMeshCount += 1;
    }

    child.userData = {
      ...(child.userData ?? {}),
      mjcfPreferDoubleSide: isLikelyThinSheetGeometry(child.geometry),
    };
  });

  return summary;
}
