import {
    BufferGeometry,
    Float32BufferAttribute,
    Group,
    LineBasicMaterial,
    LineSegments,
    Mesh,
    MeshPhongMaterial,
    Points,
    PointsMaterial,
    type Material,
    type Object3D,
} from 'three';
import {
    durationMs,
    markMainThreadBuildPerformance,
    type MeshLoadPerformanceEntry,
    readHighResolutionEpochMs,
} from './meshLoadPerformance';

export const GENERATED_OBJ_MATERIAL_USER_DATA_KEY = '__urdfStudioGeneratedObjMaterial';

export interface SerializedObjAttributeData {
    array: ArrayBuffer;
    byteLength?: number;
    byteOffset?: number;
    itemSize: number;
    normalized?: boolean;
}

export interface SerializedObjGeometryGroup {
    count: number;
    materialIndex: number;
    start: number;
}

export interface SerializedObjGeometryData {
    color?: SerializedObjAttributeData;
    groups: SerializedObjGeometryGroup[];
    normal?: SerializedObjAttributeData;
    position: SerializedObjAttributeData;
    uv?: SerializedObjAttributeData;
}

export interface SerializedObjMaterialData {
    color: number;
    flatShading?: boolean;
    kind: 'mesh-phong' | 'line-basic' | 'points';
    name: string;
    size?: number;
    sizeAttenuation?: boolean;
    vertexColors: boolean;
}

export interface SerializedObjNodeData {
    geometry: SerializedObjGeometryData;
    kind: 'mesh' | 'line-segments' | 'points';
    materials: SerializedObjMaterialData[];
    name: string;
}

export interface SerializedObjModelData {
    children: SerializedObjNodeData[];
    loadPerformance?: MeshLoadPerformanceEntry;
    materialLibraries: string[];
}

export interface CreateObjectFromSerializedObjDataAsyncOptions {
    nodeYieldInterval?: number;
    yieldIfNeeded?: () => Promise<void>;
}

const DEFAULT_ASYNC_OBJ_NODE_YIELD_INTERVAL = 8;

function createFloat32Attribute(data: SerializedObjAttributeData): Float32BufferAttribute {
    const byteOffset = data.byteOffset ?? 0;
    const byteLength = data.byteLength ?? data.array.byteLength - byteOffset;
    return new Float32BufferAttribute(
        new Float32Array(data.array, byteOffset, byteLength / Float32Array.BYTES_PER_ELEMENT),
        data.itemSize,
        data.normalized,
    );
}

function createGeometryFromSerializedObjNode(node: SerializedObjNodeData): BufferGeometry {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', createFloat32Attribute(node.geometry.position));

    if (node.geometry.normal) {
        geometry.setAttribute('normal', createFloat32Attribute(node.geometry.normal));
    }

    if (node.geometry.color) {
        geometry.setAttribute('color', createFloat32Attribute(node.geometry.color));
    }

    if (node.geometry.uv) {
        geometry.setAttribute('uv', createFloat32Attribute(node.geometry.uv));
    }

    if (node.materials.length > 1) {
        node.geometry.groups.forEach((group) => {
            geometry.addGroup(group.start, group.count, group.materialIndex);
        });
    }

    return geometry;
}

function createMaterialFromSerializedObjMaterial(
    data: SerializedObjMaterialData,
    options: { forceVertexColors?: boolean } = {},
): Material {
    const vertexColors = data.vertexColors || options.forceVertexColors === true;
    const materialColor = vertexColors ? 0xffffff : data.color;

    if (data.kind === 'line-basic') {
        const material = new LineBasicMaterial({ color: materialColor });
        material.name = data.name;
        material.vertexColors = vertexColors;
        material.toneMapped = vertexColors ? false : material.toneMapped;
        material.userData = {
            ...(material.userData ?? {}),
            [GENERATED_OBJ_MATERIAL_USER_DATA_KEY]: true,
            ...(vertexColors ? { usesVertexColors: true } : {}),
        };
        return material;
    }

    if (data.kind === 'points') {
        const material = new PointsMaterial({
            color: materialColor,
            size: data.size ?? 1,
            sizeAttenuation: data.sizeAttenuation ?? false,
        });
        material.name = data.name;
        material.vertexColors = vertexColors;
        material.toneMapped = vertexColors ? false : material.toneMapped;
        material.userData = {
            ...(material.userData ?? {}),
            [GENERATED_OBJ_MATERIAL_USER_DATA_KEY]: true,
            ...(vertexColors ? { usesVertexColors: true } : {}),
        };
        return material;
    }

    const material = new MeshPhongMaterial({
        color: materialColor,
        flatShading: data.flatShading ?? false,
    });
    material.name = data.name;
    material.vertexColors = vertexColors;
    material.toneMapped = vertexColors ? false : material.toneMapped;
    material.userData = {
        ...(material.userData ?? {}),
        [GENERATED_OBJ_MATERIAL_USER_DATA_KEY]: true,
        ...(vertexColors ? { usesVertexColors: true } : {}),
    };
    return material;
}

function createObjectFromSerializedObjNode(node: SerializedObjNodeData): Object3D {
    const geometry = createGeometryFromSerializedObjNode(node);
    const usesVertexColors = Boolean(node.geometry.color);
    const materials = node.materials.map((material) => (
        createMaterialFromSerializedObjMaterial(material, {
            forceVertexColors: usesVertexColors,
        })
    ));
    const material = materials.length > 1 ? materials : materials[0];
    let object: Object3D;

    if (node.kind === 'line-segments') {
        object = new LineSegments(geometry, material);
    } else if (node.kind === 'points') {
        object = new Points(geometry, material);
    } else {
        object = new Mesh(geometry, material);
    }

    object.name = node.name;
    return object;
}

export function createObjectFromSerializedObjData(data: SerializedObjModelData): Group {
    const startedAt = readHighResolutionEpochMs();
    const container = new Group() as Group & { materialLibraries?: string[] };
    container.materialLibraries = [...data.materialLibraries];

    data.children.forEach((child) => {
        container.add(createObjectFromSerializedObjNode(child));
    });

    markMainThreadBuildPerformance(data.loadPerformance, durationMs(startedAt));
    if (data.loadPerformance) {
        container.userData = {
            ...(container.userData ?? {}),
            meshLoadPerformance: data.loadPerformance,
        };
    }
    return container;
}

export async function createObjectFromSerializedObjDataAsync(
    data: SerializedObjModelData,
    options: CreateObjectFromSerializedObjDataAsyncOptions = {},
): Promise<Group> {
    const startedAt = readHighResolutionEpochMs();
    const container = new Group() as Group & { materialLibraries?: string[] };
    const yieldIfNeeded = options.yieldIfNeeded;
    const nodeYieldInterval = Math.max(
        1,
        Math.floor(options.nodeYieldInterval ?? DEFAULT_ASYNC_OBJ_NODE_YIELD_INTERVAL),
    );

    container.materialLibraries = [...data.materialLibraries];

    for (let index = 0; index < data.children.length; index += 1) {
        container.add(createObjectFromSerializedObjNode(data.children[index]));

        if (yieldIfNeeded && (index + 1) % nodeYieldInterval === 0) {
            await yieldIfNeeded();
        }
    }

    await yieldIfNeeded?.();
    markMainThreadBuildPerformance(data.loadPerformance, durationMs(startedAt));
    if (data.loadPerformance) {
        container.userData = {
            ...(container.userData ?? {}),
            meshLoadPerformance: data.loadPerformance,
        };
    }
    return container;
}

export function collectSerializedObjTransferables(data: SerializedObjModelData): ArrayBuffer[] {
    const transferables = new Set<ArrayBuffer>();

    data.children.forEach((child) => {
        transferables.add(child.geometry.position.array);
        if (child.geometry.normal) {
            transferables.add(child.geometry.normal.array);
        }
        if (child.geometry.color) {
            transferables.add(child.geometry.color.array);
        }
        if (child.geometry.uv) {
            transferables.add(child.geometry.uv.array);
        }
    });

    return Array.from(transferables);
}
