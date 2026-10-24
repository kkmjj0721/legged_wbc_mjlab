import { BufferGeometry, Float32BufferAttribute, Vector3 } from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';

export interface SerializedStlGeometryData {
    positions: ArrayBuffer;
    normals: ArrayBuffer;
    maxDimension: number | null;
}

function extractTransferableBuffer(array: Float32Array): ArrayBuffer {
    if (array.byteOffset === 0 && array.byteLength === array.buffer.byteLength) {
        return array.buffer;
    }

    return array.slice().buffer;
}

export function parseStlGeometryData(data: ArrayBuffer): SerializedStlGeometryData {
    const loader = new STLLoader();
    const geometry = loader.parse(data);
    const positionAttribute = geometry.getAttribute('position');
    const normalAttribute = geometry.getAttribute('normal');

    if (!(positionAttribute?.array instanceof Float32Array) || !(normalAttribute?.array instanceof Float32Array)) {
        throw new Error('Failed to parse STL geometry attributes');
    }

    geometry.computeBoundingBox();
    let maxDimension: number | null = null;
    if (geometry.boundingBox) {
        const size = geometry.boundingBox.getSize(loaderScopeSizeVector);
        maxDimension = Math.max(size.x, size.y, size.z);
    }

    return {
        positions: extractTransferableBuffer(positionAttribute.array),
        normals: extractTransferableBuffer(normalAttribute.array),
        maxDimension,
    };
}

export function createGeometryFromSerializedStlData(data: SerializedStlGeometryData): BufferGeometry {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute(new Float32Array(data.positions), 3));
    geometry.setAttribute('normal', new Float32BufferAttribute(new Float32Array(data.normals), 3));
    return geometry;
}

const loaderScopeSizeVector = new Vector3();
