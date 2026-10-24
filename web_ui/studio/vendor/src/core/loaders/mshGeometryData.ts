import { BufferAttribute, BufferGeometry, Float32BufferAttribute, Vector3 } from 'three';
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry.js';

export interface SerializedMshGeometryData {
  positions: ArrayBuffer;
  normals: ArrayBuffer | null;
  uvs: ArrayBuffer | null;
  indices: ArrayBuffer | null;
  maxDimension: number | null;
}

export function cloneSerializedMshGeometryData(
  data: SerializedMshGeometryData,
): SerializedMshGeometryData {
  return {
    positions: data.positions.slice(0),
    normals: data.normals ? data.normals.slice(0) : null,
    uvs: data.uvs ? data.uvs.slice(0) : null,
    indices: data.indices ? data.indices.slice(0) : null,
    maxDimension: data.maxDimension,
  };
}

export function collectSerializedMshTransferables(
  data: SerializedMshGeometryData,
): Transferable[] {
  return [data.positions, data.normals, data.uvs, data.indices].filter(
    (value): value is ArrayBuffer => value instanceof ArrayBuffer,
  );
}

function cloneBufferSlice(
  buffer: ArrayBuffer,
  byteOffset: number,
  byteLength: number,
): ArrayBuffer {
  return buffer.slice(byteOffset, byteOffset + byteLength);
}

function computeMaxDimension(positionArray: Float32Array): number | null {
  if (positionArray.length < 3) {
    return null;
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;

  for (let index = 0; index + 2 < positionArray.length; index += 3) {
    const x = positionArray[index] ?? 0;
    const y = positionArray[index + 1] ?? 0;
    const z = positionArray[index + 2] ?? 0;

    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }

  return Math.max(maxX - minX, maxY - minY, maxZ - minZ);
}

export function parseMshGeometryData(data: ArrayBuffer): SerializedMshGeometryData {
  if (data.byteLength < 16) {
    throw new Error('Legacy MSH file is too small to contain a valid header');
  }

  const view = new DataView(data);
  const nvertex = view.getInt32(0, true);
  const nnormal = view.getInt32(4, true);
  const ntexcoord = view.getInt32(8, true);
  const nface = view.getInt32(12, true);

  if (nvertex < 4) {
    throw new Error(`Legacy MSH file must contain at least 4 vertices (received ${nvertex})`);
  }

  if (!(nnormal === 0 || nnormal === nvertex)) {
    throw new Error(
      `Legacy MSH file must contain either 0 or ${nvertex} normals (received ${nnormal})`,
    );
  }

  if (!(ntexcoord === 0 || ntexcoord === nvertex)) {
    throw new Error(
      `Legacy MSH file must contain either 0 or ${nvertex} texture coordinates (received ${ntexcoord})`,
    );
  }

  if (nface < 0) {
    throw new Error(`Legacy MSH file face count must be non-negative (received ${nface})`);
  }

  const expectedByteLength = 16 + 12 * (nvertex + nnormal + nface) + 8 * ntexcoord;
  if (data.byteLength !== expectedByteLength) {
    throw new Error(
      `Legacy MSH file size mismatch: expected ${expectedByteLength} bytes, received ${data.byteLength}`,
    );
  }

  let byteOffset = 16;

  const positionByteLength = nvertex * 3 * Float32Array.BYTES_PER_ELEMENT;
  const positions = cloneBufferSlice(data, byteOffset, positionByteLength);
  byteOffset += positionByteLength;

  let normals: ArrayBuffer | null = null;
  if (nnormal > 0) {
    const normalByteLength = nnormal * 3 * Float32Array.BYTES_PER_ELEMENT;
    normals = cloneBufferSlice(data, byteOffset, normalByteLength);
    byteOffset += normalByteLength;
  }

  let uvs: ArrayBuffer | null = null;
  if (ntexcoord > 0) {
    const uvByteLength = ntexcoord * 2 * Float32Array.BYTES_PER_ELEMENT;
    uvs = cloneBufferSlice(data, byteOffset, uvByteLength);
    byteOffset += uvByteLength;
  }

  let indices: ArrayBuffer | null = null;
  if (nface > 0) {
    const indexByteLength = nface * 3 * Int32Array.BYTES_PER_ELEMENT;
    indices = cloneBufferSlice(data, byteOffset, indexByteLength);
    const indexArray = new Int32Array(indices);
    for (let index = 0; index < indexArray.length; index += 1) {
      const vertexIndex = indexArray[index] ?? 0;
      if (vertexIndex < 0 || vertexIndex >= nvertex) {
        throw new Error(
          `Legacy MSH file face index ${vertexIndex} is outside vertex range 0..${nvertex - 1}`,
        );
      }
    }
    byteOffset += indexByteLength;
  }

  return {
    positions,
    normals,
    uvs,
    indices,
    maxDimension: computeMaxDimension(new Float32Array(positions)),
  };
}

function createConvexHullGeometry(positions: Float32Array): BufferGeometry {
  const points: Vector3[] = [];
  for (let index = 0; index + 2 < positions.length; index += 3) {
    points.push(
      new Vector3(positions[index] ?? 0, positions[index + 1] ?? 0, positions[index + 2] ?? 0),
    );
  }

  return new ConvexGeometry(points);
}

export function createGeometryFromSerializedMshData(
  data: SerializedMshGeometryData,
): BufferGeometry {
  const positions = new Float32Array(data.positions);

  if (!data.indices) {
    return createConvexHullGeometry(positions);
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));

  if (data.normals) {
    geometry.setAttribute('normal', new Float32BufferAttribute(new Float32Array(data.normals), 3));
  }

  if (data.uvs) {
    geometry.setAttribute('uv', new Float32BufferAttribute(new Float32Array(data.uvs), 2));
  }

  geometry.setIndex(new BufferAttribute(new Uint32Array(new Int32Array(data.indices)), 1));
  if (!data.normals) {
    geometry.computeVertexNormals();
  }

  return geometry;
}
