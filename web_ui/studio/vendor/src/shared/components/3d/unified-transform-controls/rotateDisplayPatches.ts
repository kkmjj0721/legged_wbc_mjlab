import * as THREE from 'three';
import {
  AXIS_NAMES,
  DEFAULT_DISPLAY_THICKNESS_SCALE,
  GIZMO_ARC_RENDER_ORDER,
  ROTATE_ARC_BACK_OPACITY,
  ROTATE_ARC_FRONT_OPACITY,
  ROTATE_ARC_GAP_SAMPLE_COUNT,
  THICK_ROTATE_ARC_RADIUS,
  THICK_ROTATE_PICKER_ARC_RADIUS,
} from './gizmoCore';
import {
  cloneAxisColorMaterial,
  hideStockAxisLines,
  removeGeneratedHandles,
  replaceMeshGeometry,
} from './displayPatchShared';

const createRotateArcMaterial = (
  sourceMaterial: THREE.Material | null,
  opacity: number
) => {
  const material = cloneAxisColorMaterial(sourceMaterial);
  material.transparent = true;
  material.opacity = opacity;
  material.userData = {
    ...material.userData,
    urdfBaseOpacity: opacity,
    urdfBaseTransparent: true,
  };
  return material;
};

const createTubeArcGeometry = (points: THREE.Vector3[]) => {
  const curve = new THREE.CatmullRomCurve3(points, false);
  return new THREE.TubeGeometry(
    curve,
    Math.max(points.length * 3, 64),
    THICK_ROTATE_ARC_RADIUS,
    12,
    false
  );
};

const splitArcIntoOpenSegments = (points: THREE.Vector3[]) => {
  const endTrim = Math.min(
    ROTATE_ARC_GAP_SAMPLE_COUNT,
    Math.max(0, Math.floor((points.length - 4) / 2))
  );
  const midTrim = Math.min(
    ROTATE_ARC_GAP_SAMPLE_COUNT,
    Math.max(0, Math.floor((points.length - 6) / 4))
  );
  const middleIndex = Math.floor(points.length / 2);

  const segments = [
    points.slice(endTrim, Math.max(endTrim + 2, middleIndex - midTrim + 1)),
    points.slice(Math.min(points.length - 2, middleIndex + midTrim), points.length - endTrim),
  ];

  return segments.filter((segment) => segment.length >= 3);
};

const addRotateArcMeshes = (group: THREE.Object3D | undefined) => {
  if (!group) return;

  removeGeneratedHandles(group, (node) => Boolean(node.userData?.urdfRotateArcMesh));

  group.traverse((node) => {
    const line = node as THREE.Line & { material?: THREE.Material | THREE.Material[] };
    if (!line.isLine || !AXIS_NAMES.has(line.name)) return;

    const position = line.geometry.getAttribute('position');
    if (!position || position.count < 3) return;

    const points: THREE.Vector3[] = [];
    for (let index = 0; index < position.count; index += 1) {
      points.push(new THREE.Vector3(position.getX(index), position.getY(index), position.getZ(index)));
    }

    if (points.length > 2 && points[0].distanceToSquared(points[points.length - 1]) < 1e-8) {
      points.pop();
    }

    const frontSegments = splitArcIntoOpenSegments(points);
    if (frontSegments.length === 0) return;

    const material = Array.isArray(line.material) ? line.material[0] : line.material;

    frontSegments.forEach((segment, index) => {
      const frontArcMesh = new THREE.Mesh(
        createTubeArcGeometry(segment),
        createRotateArcMaterial(material ?? null, ROTATE_ARC_FRONT_OPACITY)
      );
      frontArcMesh.name = line.name;
      frontArcMesh.renderOrder = GIZMO_ARC_RENDER_ORDER + 1;
      frontArcMesh.userData = {
        ...frontArcMesh.userData,
        isGizmo: true,
        urdfRotateArcMesh: true,
        urdfRotateArcLayer: 'front',
        urdfRotateArcSegmentIndex: index,
        urdfRotateCenterlinePoints: segment.map((point) => point.clone()),
      };
      group.add(frontArcMesh);

      const backArcMesh = new THREE.Mesh(
        createTubeArcGeometry(
          segment
            .map((point) => point.clone().multiplyScalar(-1))
            .reverse()
        ),
        createRotateArcMaterial(material ?? null, ROTATE_ARC_BACK_OPACITY)
      );
      backArcMesh.name = line.name;
      backArcMesh.renderOrder = GIZMO_ARC_RENDER_ORDER;
      backArcMesh.userData = {
        ...backArcMesh.userData,
        isGizmo: true,
        urdfRotateArcMesh: true,
        urdfRotateArcLayer: 'back',
        urdfRotateArcSegmentIndex: index,
        urdfRotateCenterlinePoints: segment
          .map((point) => point.clone().multiplyScalar(-1))
          .reverse(),
      };
      group.add(backArcMesh);
    });
  });
};

const patchRotateHandleScale = (
  group: THREE.Object3D | undefined,
  thicknessScale: number
) => {
  if (!group) return;

  const scaleKey = thicknessScale.toFixed(3);
  const handleScale = Math.max(1, 0.9 + thicknessScale * 0.7);

  group.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh || !AXIS_NAMES.has(mesh.name)) return;

    const geoType = (mesh.geometry as THREE.BufferGeometry & { type?: string }).type;
    if (geoType !== 'OctahedronGeometry') return;
    if (mesh.userData?.urdfRotateHandleScaleKey === scaleKey) return;

    const baseScale =
      mesh.userData?.urdfRotateBaseScale instanceof THREE.Vector3
        ? (mesh.userData.urdfRotateBaseScale as THREE.Vector3)
        : mesh.scale.clone();

    mesh.userData.urdfRotateBaseScale = baseScale.clone();
    mesh.scale.copy(baseScale).multiplyScalar(handleScale);
    mesh.userData.urdfRotateHandleScaleKey = scaleKey;
  });
};

const ROTATE_GIZMO_SETUP_ROTATION: Record<'X' | 'Y' | 'Z', THREE.Euler> = {
  X: new THREE.Euler(0, 0, 0),
  Y: new THREE.Euler(0, 0, -Math.PI / 2),
  Z: new THREE.Euler(0, Math.PI / 2, 0),
};

const ROTATE_PICKER_SETUP_ROTATION: Record<'X' | 'Y' | 'Z', THREE.Euler> = {
  X: new THREE.Euler(0, -Math.PI / 2, -Math.PI / 2),
  Y: new THREE.Euler(Math.PI / 2, 0, 0),
  Z: new THREE.Euler(0, 0, -Math.PI / 2),
};

const patchRotateThickness = (
  group: THREE.Object3D | undefined,
  {
    isPicker = false,
    thicknessScale = DEFAULT_DISPLAY_THICKNESS_SCALE,
  }: { isPicker?: boolean; thicknessScale?: number } = {}
) => {
  if (!group) return;

  const scaleKey = `${isPicker ? 'picker' : 'visible'}:${thicknessScale.toFixed(3)}`;

  group.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh || !AXIS_NAMES.has(mesh.name)) return;

    const geoType = (mesh.geometry as THREE.BufferGeometry & { type?: string }).type;
    if (geoType !== 'TorusGeometry') return;

    const parameters = (mesh.geometry as THREE.BufferGeometry & {
      parameters?: { radius?: number; tube?: number; radialSegments?: number; tubularSegments?: number; arc?: number };
    }).parameters ?? {};

    if (mesh.userData?.urdfRotateThicknessKey === scaleKey) return;

    const originalTube = parameters?.tube ?? 0.0075;
    const targetTube = (isPicker ? THICK_ROTATE_PICKER_ARC_RADIUS : THICK_ROTATE_ARC_RADIUS) * thicknessScale;
    if (targetTube <= originalTube) return;

    const axisName = mesh.name as 'X' | 'Y' | 'Z';
    const arc = !isPicker
      ? Math.PI * 2
      : (parameters.arc ?? Math.PI * 2);

    const newGeo = new THREE.TorusGeometry(
      parameters.radius ?? 0.5,
      targetTube,
      Math.max(parameters.radialSegments ?? 8, 8),
      Math.max(parameters.tubularSegments ?? 64, 64),
      arc
    );

    if (!isPicker) {
      newGeo.rotateY(Math.PI / 2);
      newGeo.rotateX(Math.PI / 2);
      const setupRot = ROTATE_GIZMO_SETUP_ROTATION[axisName];
      if (setupRot) {
        newGeo.applyMatrix4(new THREE.Matrix4().makeRotationFromEuler(setupRot));
      }
    } else {
      const setupRot = ROTATE_PICKER_SETUP_ROTATION[axisName];
      if (setupRot) {
        newGeo.applyMatrix4(new THREE.Matrix4().makeRotationFromEuler(setupRot));
      }
    }

    replaceMeshGeometry(mesh, newGeo);
    mesh.userData.urdfRotateThicknessKey = scaleKey;
  });
};

export const applyRotateDisplayPatches = (
  gizmo: any,
  thicknessScale: number
) => {
  hideStockAxisLines(gizmo.gizmo?.rotate);
  addRotateArcMeshes(gizmo.gizmo?.rotate);
  patchRotateHandleScale(gizmo.gizmo?.rotate, thicknessScale);
  patchRotateThickness(gizmo.gizmo?.rotate, { thicknessScale });
  patchRotateThickness(gizmo.picker?.rotate, {
    isPicker: true,
    thicknessScale,
  });
};
