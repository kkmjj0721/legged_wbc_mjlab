import * as THREE from 'three';
import {
  AXIS_GEOMETRY_RADIAL_SEGMENTS,
  AXIS_NAMES,
  DEFAULT_DISPLAY_THICKNESS_SCALE,
  GIZMO_ARC_RENDER_ORDER,
  MIN_TRANSLATE_SHAFT_SEGMENT_LENGTH,
  THICK_TRANSLATE_PICKER_RADIUS,
  THICK_TRANSLATE_SHAFT_RADIUS,
  THICK_TRANSLATE_TIP_RADIUS,
  TRANSLATE_ARROW_BASE_RADIUS,
  TRANSLATE_ARROW_HANDLE_OFFSET,
  TRANSLATE_ARROW_LENGTH,
  TRANSLATE_CENTER_GAP,
  TRANSLATE_GAP_BRIDGE_DASH_COUNT,
  TRANSLATE_GAP_BRIDGE_RADIUS,
  TRANSLATE_PICKER_END_PADDING,
  TRANSLATE_PICKER_START_OFFSET,
  TRANSLATE_RING_INTERSECTION_GAP,
  TRANSLATE_RING_INTERSECTION_RADIUS,
  getPositiveScale,
} from './gizmoCore';
import {
  cloneAxisColorMaterial,
  getAxisComponentKey,
  hideStockAxisLines,
  removeGeneratedHandles,
  replaceMeshGeometry,
} from './displayPatchShared';

const createAxisAlignedCylinderGeometry = (
  axis: 'X' | 'Y' | 'Z',
  startOffset: number,
  endOffset: number,
  radius: number,
) => {
  const segmentLength = endOffset - startOffset;
  const geometry = new THREE.CylinderGeometry(
    radius,
    radius,
    segmentLength,
    AXIS_GEOMETRY_RADIAL_SEGMENTS,
  );
  const segmentCenter = startOffset + segmentLength * 0.5;

  if (axis === 'X') {
    geometry.rotateZ(-Math.PI / 2);
    geometry.translate(segmentCenter, 0, 0);
  } else if (axis === 'Y') {
    geometry.translate(0, segmentCenter, 0);
  } else {
    geometry.rotateX(Math.PI / 2);
    geometry.translate(0, 0, segmentCenter);
  }

  return geometry;
};

interface AxisAlignedArrowGeometryOptions {
  axis: 'X' | 'Y' | 'Z';
  direction: 1 | -1;
  startOffset: number;
  length: number;
  radius: number;
}

const createAxisAlignedArrowGeometry = ({
  axis,
  direction,
  startOffset,
  length,
  radius,
}: AxisAlignedArrowGeometryOptions) => {
  const geometry = new THREE.CylinderGeometry(0, radius, length, AXIS_GEOMETRY_RADIAL_SEGMENTS);
  const segmentCenter = direction * (startOffset + length * 0.5);

  if (axis === 'X') {
    geometry.rotateZ(direction === 1 ? -Math.PI / 2 : Math.PI / 2);
    geometry.translate(segmentCenter, 0, 0);
  } else if (axis === 'Y') {
    if (direction === -1) {
      geometry.rotateX(Math.PI);
    }
    geometry.translate(0, segmentCenter, 0);
  } else {
    geometry.rotateX(direction === 1 ? Math.PI / 2 : -Math.PI / 2);
    geometry.translate(0, 0, segmentCenter);
  }

  return geometry;
};

const createTranslateDisplayName = (kind: 'shaft' | 'tip' | 'gap', axis: 'X' | 'Y' | 'Z') =>
  `translate-${kind}-${axis.toLowerCase()}`;

const addTranslateShaftMeshes = (
  group: THREE.Object3D | undefined,
  {
    leaveRingGap = false,
  }: {
    leaveRingGap?: boolean;
  } = {},
) => {
  if (!group) return;

  removeGeneratedHandles(group, (node) => Boolean(node.userData?.urdfTranslateShaft));

  const axisMaterials = new Map<string, THREE.Material>();
  group.traverse((node) => {
    const line = node as THREE.Line & { material?: THREE.Material | THREE.Material[] };
    if (!line.isLine || !AXIS_NAMES.has(line.name) || axisMaterials.has(line.name)) return;

    const material = Array.isArray(line.material) ? line.material[0] : line.material;
    if (material) {
      axisMaterials.set(line.name, material);
    }
  });

  const shaftStart = TRANSLATE_CENTER_GAP;
  const shaftEnd = TRANSLATE_ARROW_HANDLE_OFFSET;

  for (const axis of ['X', 'Y', 'Z'] as const) {
    const positiveSegments: Array<[number, number]> = [];

    if (leaveRingGap) {
      const gapStart = Math.max(
        shaftStart,
        TRANSLATE_RING_INTERSECTION_RADIUS - TRANSLATE_RING_INTERSECTION_GAP * 0.5,
      );
      const gapEnd = Math.min(
        shaftEnd,
        TRANSLATE_RING_INTERSECTION_RADIUS + TRANSLATE_RING_INTERSECTION_GAP * 0.5,
      );

      if (gapStart - shaftStart > MIN_TRANSLATE_SHAFT_SEGMENT_LENGTH) {
        positiveSegments.push([shaftStart, gapStart]);
      }
      if (shaftEnd - gapEnd > MIN_TRANSLATE_SHAFT_SEGMENT_LENGTH) {
        positiveSegments.push([gapEnd, shaftEnd]);
      }
    }

    if (positiveSegments.length === 0) {
      positiveSegments.push([shaftStart, shaftEnd]);
    }

    for (const direction of [1, -1] as const) {
      const segments =
        direction === 1
          ? positiveSegments
          : positiveSegments.map(
              ([startOffset, endOffset]) => [-endOffset, -startOffset] as [number, number],
            );

      for (const [startOffset, endOffset] of segments) {
        const shaft = new THREE.Mesh(
          createAxisAlignedCylinderGeometry(
            axis,
            startOffset,
            endOffset,
            THICK_TRANSLATE_SHAFT_RADIUS,
          ),
          cloneAxisColorMaterial(axisMaterials.get(axis) ?? null),
        );
        shaft.name = createTranslateDisplayName('shaft', axis);
        shaft.renderOrder = GIZMO_ARC_RENDER_ORDER;
        shaft.userData = {
          ...shaft.userData,
          isGizmo: true,
          urdfAxis: axis,
          urdfTranslateDirection: direction,
          urdfTranslateShaft: true,
        };
        group.add(shaft);
      }
    }
  }
};

const addTranslateGapBridgeMeshes = (group: THREE.Object3D | undefined) => {
  if (!group) return;

  removeGeneratedHandles(group, (node) => Boolean(node.userData?.urdfTranslateGapBridge));

  const axisMaterials = new Map<string, THREE.Material>();
  group.traverse((node) => {
    const line = node as THREE.Line & { material?: THREE.Material | THREE.Material[] };
    if (!line.isLine || !AXIS_NAMES.has(line.name) || axisMaterials.has(line.name)) return;

    const material = Array.isArray(line.material) ? line.material[0] : line.material;
    if (material) {
      axisMaterials.set(line.name, material);
    }
  });

  const gapStart = TRANSLATE_RING_INTERSECTION_RADIUS - TRANSLATE_RING_INTERSECTION_GAP * 0.5;
  const gapEnd = TRANSLATE_RING_INTERSECTION_RADIUS + TRANSLATE_RING_INTERSECTION_GAP * 0.5;
  const bridgeGapSize = 0.006;
  const dashLength =
    (gapEnd - gapStart - bridgeGapSize * (TRANSLATE_GAP_BRIDGE_DASH_COUNT - 1)) /
    TRANSLATE_GAP_BRIDGE_DASH_COUNT;

  if (dashLength <= 0) return;

  for (const axis of ['X', 'Y', 'Z'] as const) {
    for (const direction of [1, -1] as const) {
      for (let index = 0; index < TRANSLATE_GAP_BRIDGE_DASH_COUNT; index += 1) {
        const positiveStartOffset = gapStart + index * (dashLength + bridgeGapSize);
        const positiveEndOffset = positiveStartOffset + dashLength;
        const startOffset = direction === 1 ? positiveStartOffset : -positiveEndOffset;
        const endOffset = direction === 1 ? positiveEndOffset : -positiveStartOffset;
        const bridge = new THREE.Mesh(
          createAxisAlignedCylinderGeometry(
            axis,
            startOffset,
            endOffset,
            TRANSLATE_GAP_BRIDGE_RADIUS,
          ),
          cloneAxisColorMaterial(axisMaterials.get(axis) ?? null),
        );

        bridge.name = createTranslateDisplayName('gap', axis);
        bridge.visible = false;
        bridge.renderOrder = GIZMO_ARC_RENDER_ORDER + 1;
        bridge.userData = {
          ...bridge.userData,
          isGizmo: true,
          urdfAxis: axis,
          urdfTranslateDirection: direction,
          urdfTranslateGapBridge: true,
        };
        group.add(bridge);
      }
    }
  }
};

const removeStockTranslateVisibleMeshes = (group: THREE.Object3D | undefined) => {
  if (!group) return;

  const nodesToRemove: THREE.Object3D[] = [];
  group.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh || !AXIS_NAMES.has(mesh.name)) return;

    const geoType = (mesh.geometry as THREE.BufferGeometry & { type?: string }).type;
    if (geoType !== 'CylinderGeometry') return;
    nodesToRemove.push(mesh);
  });

  for (const node of nodesToRemove) {
    node.parent?.remove(node);
    (node as THREE.Object3D).traverse((child) => {
      const geometry = (child as THREE.Mesh).geometry as THREE.BufferGeometry | undefined;
      const material = (child as THREE.Mesh & { material?: THREE.Material | THREE.Material[] })
        .material;
      geometry?.dispose?.();
      if (Array.isArray(material)) {
        material.forEach((entry) => entry?.dispose?.());
      } else {
        material?.dispose?.();
      }
    });
  }
};

const addTranslateArrowMeshes = (group: THREE.Object3D | undefined) => {
  if (!group) return;

  removeGeneratedHandles(group, (node) => Boolean(node.userData?.urdfTranslateTip));

  const axisMaterials = new Map<string, THREE.Material>();
  group.traverse((node) => {
    const line = node as THREE.Line & { material?: THREE.Material | THREE.Material[] };
    if (!line.isLine || !AXIS_NAMES.has(line.name) || axisMaterials.has(line.name)) return;

    const material = Array.isArray(line.material) ? line.material[0] : line.material;
    if (material) {
      axisMaterials.set(line.name, material);
    }
  });

  for (const axis of ['X', 'Y', 'Z'] as const) {
    for (const direction of [1, -1] as const) {
      const tip = new THREE.Mesh(
        createAxisAlignedArrowGeometry({
          axis,
          direction,
          startOffset: TRANSLATE_ARROW_HANDLE_OFFSET,
          length: TRANSLATE_ARROW_LENGTH,
          radius: TRANSLATE_ARROW_BASE_RADIUS,
        }),
        cloneAxisColorMaterial(axisMaterials.get(axis) ?? null),
      );
      tip.name = createTranslateDisplayName('tip', axis);
      tip.renderOrder = GIZMO_ARC_RENDER_ORDER + 2;
      tip.userData = {
        ...tip.userData,
        isGizmo: true,
        urdfAxis: axis,
        urdfTranslateDirection: direction,
        urdfTranslateTip: true,
      };
      group.add(tip);
    }
  }
};

const patchTranslatePickerGeometry = (group: THREE.Object3D | undefined) => {
  if (!group) return;

  const pickerEnd =
    TRANSLATE_ARROW_HANDLE_OFFSET + TRANSLATE_ARROW_LENGTH + TRANSLATE_PICKER_END_PADDING;
  const geometryKey = [
    TRANSLATE_PICKER_START_OFFSET.toFixed(3),
    pickerEnd.toFixed(3),
    THICK_TRANSLATE_PICKER_RADIUS.toFixed(3),
  ].join(':');

  group.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh || !AXIS_NAMES.has(mesh.name)) return;

    const geoType = (mesh.geometry as THREE.BufferGeometry & { type?: string }).type;
    if (geoType !== 'CylinderGeometry') return;
    if (mesh.userData?.urdfTranslatePickerGeometryKey === geometryKey) return;

    replaceMeshGeometry(
      mesh,
      createAxisAlignedCylinderGeometry(
        mesh.name as 'X' | 'Y' | 'Z',
        TRANSLATE_PICKER_START_OFFSET,
        pickerEnd,
        THICK_TRANSLATE_PICKER_RADIUS,
      ),
    );
    delete mesh.userData.urdfTranslateThicknessKey;
    mesh.userData.urdfTranslatePickerGeometryKey = geometryKey;
  });
};

const removeNegativeTranslateHandles = (group: THREE.Object3D | undefined) => {
  if (!group) return;

  const nodesToRemove: THREE.Object3D[] = [];
  group.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh || !AXIS_NAMES.has(mesh.name)) return;

    const geoType = (mesh.geometry as THREE.BufferGeometry & { type?: string }).type;
    if (geoType !== 'CylinderGeometry') return;

    const axisName = mesh.name as 'X' | 'Y' | 'Z';
    const axisKey = getAxisComponentKey(axisName);
    if (node.position[axisKey] < -1e-6) {
      nodesToRemove.push(node);
    }
  });

  for (const node of nodesToRemove) {
    node.parent?.remove(node);
    (node as THREE.Object3D).traverse((child) => {
      const geometry = (child as THREE.Mesh).geometry as THREE.BufferGeometry | undefined;
      const material = (child as THREE.Mesh & { material?: THREE.Material | THREE.Material[] })
        .material;
      geometry?.dispose?.();
      if (Array.isArray(material)) {
        material.forEach((entry) => entry?.dispose?.());
      } else {
        material?.dispose?.();
      }
    });
  }
};

const markFixedTranslateHandles = (group: THREE.Object3D | undefined) => {
  if (!group) return;

  group.traverse((node) => {
    if (node.userData?.urdfTranslateShaft || node.userData?.urdfTranslateGapBridge) {
      node.userData = {
        ...node.userData,
        urdfTranslateFixedVisible: true,
        urdfTranslateStableScale: getPositiveScale(
          (node as THREE.Object3D & { scale?: THREE.Vector3 }).scale ?? new THREE.Vector3(1, 1, 1),
        ),
      };
      return;
    }

    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh || !AXIS_NAMES.has(mesh.name)) return;

    const geoType = (mesh.geometry as THREE.BufferGeometry & { type?: string }).type;
    if (geoType !== 'CylinderGeometry') return;

    node.userData = {
      ...node.userData,
      urdfTranslateFixedVisible: true,
      urdfTranslateStableScale: getPositiveScale(mesh.scale),
    };
  });
};

const AXIS_THICKNESS_SCALE: Record<string, { a: 'x' | 'y' | 'z'; b: 'x' | 'y' | 'z' }> = {
  X: { a: 'y', b: 'z' },
  Y: { a: 'x', b: 'z' },
  Z: { a: 'x', b: 'y' },
};

const patchTranslateThickness = (
  group: THREE.Object3D | undefined,
  {
    isPicker = false,
    thicknessScale = DEFAULT_DISPLAY_THICKNESS_SCALE,
  }: { isPicker?: boolean; thicknessScale?: number } = {},
) => {
  if (!group) return;

  const zeroTolerance = 1e-6;
  const scaleKey = `${isPicker ? 'picker' : 'visible'}:${thicknessScale.toFixed(3)}`;

  group.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh) return;

    const axisName = mesh.name as string;
    const scaleAxes = AXIS_THICKNESS_SCALE[axisName];
    if (!scaleAxes) return;

    const geoType = (mesh.geometry as THREE.BufferGeometry & { type?: string }).type;
    if (geoType !== 'CylinderGeometry') return;

    const params =
      (
        mesh.geometry as THREE.BufferGeometry & {
          parameters?: { radiusTop?: number; radiusBottom?: number };
        }
      ).parameters ?? {};

    if (
      mesh.userData?.urdfTranslateThicknessKey === scaleKey &&
      mesh.userData?.urdfTranslateBaseScale instanceof THREE.Vector3
    ) {
      return;
    }

    const baseScale =
      mesh.userData?.urdfTranslateBaseScale instanceof THREE.Vector3
        ? (mesh.userData.urdfTranslateBaseScale as THREE.Vector3)
        : mesh.scale.clone();
    mesh.userData.urdfTranslateBaseScale = baseScale.clone();
    mesh.scale.copy(baseScale);

    const radiusTop = params.radiusTop ?? 0;
    const radiusBottom = params.radiusBottom ?? 0;
    const isShaft = Math.abs(radiusTop - radiusBottom) < zeroTolerance;

    let scaleFactor: number;
    if (isShaft) {
      const targetRadius =
        (isPicker ? THICK_TRANSLATE_PICKER_RADIUS : THICK_TRANSLATE_SHAFT_RADIUS) * thicknessScale;
      const srcRadius = Math.max(radiusTop, zeroTolerance);
      scaleFactor = targetRadius / srcRadius;
    } else {
      const targetRadius =
        (isPicker ? THICK_TRANSLATE_PICKER_RADIUS : THICK_TRANSLATE_TIP_RADIUS) * thicknessScale;
      const srcRadius = Math.max(Math.max(radiusTop, radiusBottom), zeroTolerance);
      scaleFactor = targetRadius / srcRadius;
    }

    mesh.scale[scaleAxes.a] *= scaleFactor;
    mesh.scale[scaleAxes.b] *= scaleFactor;
    mesh.userData.urdfTranslateThicknessKey = scaleKey;
  });
};

export const applyTranslateDisplayPatches = (
  gizmo: any,
  thicknessScale: number,
  {
    leaveRingGap = false,
  }: {
    leaveRingGap?: boolean;
  } = {},
) => {
  removeNegativeTranslateHandles(gizmo.picker?.translate);
  removeStockTranslateVisibleMeshes(gizmo.gizmo?.translate);
  patchTranslatePickerGeometry(gizmo.picker?.translate);
  patchTranslateThickness(gizmo.gizmo?.translate, { thicknessScale });
  patchTranslateThickness(gizmo.picker?.translate, {
    isPicker: true,
    thicknessScale,
  });
  hideStockAxisLines(gizmo.gizmo?.translate);
  addTranslateShaftMeshes(gizmo.gizmo?.translate, {
    leaveRingGap,
  });
  addTranslateArrowMeshes(gizmo.gizmo?.translate);
  if (leaveRingGap) {
    addTranslateGapBridgeMeshes(gizmo.gizmo?.translate);
  } else {
    removeGeneratedHandles(gizmo.gizmo?.translate, (node) =>
      Boolean(node.userData?.urdfTranslateGapBridge),
    );
  }
  markFixedTranslateHandles(gizmo.gizmo?.translate);
  markFixedTranslateHandles(gizmo.picker?.translate);
};
