import * as THREE from 'three';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';

import { createObjectFromSerializedObjData } from './objModelData';
import {
  addMainThreadMaterialPerformance,
  durationMs,
  readHighResolutionEpochMs,
} from './meshLoadPerformance';
import { parseObjModelDataFromBytes } from './objWasmParser';
import { getSourceFileDirectory, resolveImportedAssetPath } from '@/core/parsers/meshPathUtils';
import { isRegressionDebugEnabled } from '@/core/utils/runtimeDiagnostics';
import type { UrdfVisualMaterial } from '@/types';

export interface ObjMaterialMetadata {
  color?: string;
  texture?: string;
}

export type TextAssetContentLookup = ReadonlyMap<string, string>;
const TEXTURE_DIRECTIVE_PATTERN = /^([ \t]*(?:map_[^\s]+|bump|disp|decal|refl)\b[ \t]*)(.*)$/i;
const MATERIAL_TEXTURE_PROPERTIES = [
  'map',
  'lightMap',
  'bumpMap',
  'normalMap',
  'specularMap',
  'envMap',
  'alphaMap',
  'aoMap',
  'displacementMap',
  'emissiveMap',
  'gradientMap',
  'metalnessMap',
  'roughnessMap',
  'clearcoatMap',
  'clearcoatNormalMap',
  'clearcoatRoughnessMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'transmissionMap',
  'thicknessMap',
  'anisotropyMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'specularColorMap',
  'specularIntensityMap',
] as const;

function normalizeLookupPath(filePath: string): string {
  return filePath.replace(/\\/g, '/').trim().replace(/^\/+/, '').split('?')[0];
}

function rgbaUnitChannelToHex(value: number): string {
  const byte = Math.max(0, Math.min(255, Math.round(value * 255)));
  return byte.toString(16).padStart(2, '0');
}

function parseMtlColorHex(mtlText: string): string | undefined {
  const match = mtlText.match(
    /^[ \t]*K[da][ \t]+([0-9eE.+-]+)[ \t]+([0-9eE.+-]+)[ \t]+([0-9eE.+-]+)/m,
  );
  if (!match) {
    return undefined;
  }

  const channels = match.slice(1, 4).map((value) => Number.parseFloat(value));
  if (!channels.every((value) => Number.isFinite(value))) {
    return undefined;
  }

  return `#${channels.map(rgbaUnitChannelToHex).join('')}`;
}

function parseMtlTexturePath(line: string): string | null {
  const tokens = line.trim().split(/\s+/).slice(1);
  if (tokens.length === 0) {
    return null;
  }

  for (let index = tokens.length - 1; index >= 0; index -= 1) {
    const token = tokens[index]?.trim();
    if (!token || token.startsWith('-')) {
      continue;
    }
    return token;
  }

  return null;
}

function findBestLookupPath(
  assetPath: string,
  textAssetContentLookup: TextAssetContentLookup,
): string | null {
  const normalizedAssetPath = normalizeLookupPath(assetPath);
  if (!normalizedAssetPath) {
    return null;
  }

  if (textAssetContentLookup.has(normalizedAssetPath)) {
    return normalizedAssetPath;
  }

  const basename = normalizedAssetPath.split('/').pop()?.toLowerCase() || '';
  if (!basename) {
    return null;
  }

  const packagePrefix = normalizedAssetPath.split('/')[0]?.toLowerCase() || '';
  const candidates = Array.from(textAssetContentLookup.keys()).filter((candidatePath) => {
    const candidateBasename = candidatePath.split('/').pop()?.toLowerCase() || '';
    return candidateBasename === basename;
  });

  if (candidates.length === 0) {
    return null;
  }

  const samePackageCandidate = candidates.find((candidatePath) => {
    const candidatePrefix = candidatePath.split('/')[0]?.toLowerCase() || '';
    return packagePrefix && candidatePrefix === packagePrefix;
  });

  return samePackageCandidate ?? candidates[0] ?? null;
}

function resolveMtlTextureRewritePath(texturePath: string, materialFilePath: string): string {
  // Resolve the texture path relative to its MTL so the LoadingManager can later map it to a
  // blob URL. We deliberately do NOT call manager.resolveURL() here: for imported assets that
  // returns an extensionless blob: URL, which hides the file extension from
  // manager.getHandler() and makes .tga/.hdr textures skip their registered decoders (they
  // then fall back to TextureLoader and render white). Keeping the resolved *path* with its
  // extension lets getHandler pick the right loader, which resolves the URL via the manager
  // itself at load time.
  return resolveImportedAssetPath(texturePath, materialFilePath) || texturePath;
}

export function rewriteMtlTextureReferencesForManager(
  materialText: string,
  materialFilePath: string,
): string {
  return materialText
    .split(/\r?\n/)
    .map((line) => {
      const directiveMatch = line.match(TEXTURE_DIRECTIVE_PATTERN);
      if (!directiveMatch) {
        return line;
      }

      const texturePath = parseMtlTexturePath(line);
      if (!texturePath) {
        return line;
      }

      const value = directiveMatch[2] ?? '';
      const textureStart = value.lastIndexOf(texturePath);
      if (textureStart < 0) {
        return line;
      }

      const resolvedTexturePath = resolveMtlTextureRewritePath(texturePath, materialFilePath);

      return `${directiveMatch[1]}${value.slice(0, textureStart)}${resolvedTexturePath}${value.slice(
        textureStart + texturePath.length,
      )}`;
    })
    .join('\n');
}

function parseMtlTextureReference(mtlText: string): string | undefined {
  const lines = mtlText.split(/\r?\n/);
  for (const line of lines) {
    if (!/^[ \t]*(?:map_|bump|disp)[A-Za-z0-9_ \t-]*/.test(line)) {
      continue;
    }

    const texturePath = parseMtlTexturePath(line);
    if (texturePath) {
      return texturePath;
    }
  }

  return undefined;
}

export function parseObjMaterialLibraries(objText: string): string[] {
  const materialLibraries: string[] = [];
  const seenLibraries = new Set<string>();
  const matches = objText.matchAll(/^[ \t]*mtllib[ \t]+(.+)$/gim);
  for (const match of matches) {
    const rawValue = String(match[1] || '').trim();
    if (!rawValue) {
      continue;
    }

    for (const materialLibrary of rawValue.split(/\s+/)) {
      const normalizedMaterialLibrary = materialLibrary.trim();
      if (!normalizedMaterialLibrary || seenLibraries.has(normalizedMaterialLibrary)) {
        continue;
      }

      seenLibraries.add(normalizedMaterialLibrary);
      materialLibraries.push(normalizedMaterialLibrary);
    }
  }

  return materialLibraries;
}

export function createTextAssetContentLookup(
  allFileContents: Record<string, string>,
): TextAssetContentLookup {
  const lookup = new Map<string, string>();

  for (const [filePath, content] of Object.entries(allFileContents)) {
    if (typeof content !== 'string' || content.length === 0) {
      continue;
    }

    const normalizedPath = normalizeLookupPath(filePath);
    if (!normalizedPath) {
      continue;
    }

    lookup.set(normalizedPath, content);
  }

  return lookup;
}

function findTextAssetContent(
  assetPath: string,
  textAssetContentLookup: TextAssetContentLookup,
): string | null {
  const resolvedAssetPath = findBestLookupPath(assetPath, textAssetContentLookup);
  if (!resolvedAssetPath) {
    return null;
  }

  return textAssetContentLookup.get(resolvedAssetPath) ?? null;
}

function findTextAssetEntry(
  assetPath: string,
  textAssetContentLookup: TextAssetContentLookup,
): { path: string; content: string } | null {
  const resolvedAssetPath = findBestLookupPath(assetPath, textAssetContentLookup);
  if (!resolvedAssetPath) {
    return null;
  }

  const content = textAssetContentLookup.get(resolvedAssetPath);
  if (!content) {
    return null;
  }

  return {
    path: resolvedAssetPath,
    content,
  };
}

function parseMtlAuthoredMaterials(
  mtlText: string,
  materialFilePath: string,
  textAssetContentLookup: TextAssetContentLookup,
): UrdfVisualMaterial[] {
  const materials: UrdfVisualMaterial[] = [];
  let currentMaterial: UrdfVisualMaterial | null = null;

  const flushMaterial = () => {
    if (!currentMaterial) {
      return;
    }

    if (currentMaterial.name || currentMaterial.color || currentMaterial.texture) {
      materials.push(currentMaterial);
    }
  };

  mtlText.split(/\r?\n/).forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) {
      return;
    }

    const newMaterialMatch = trimmed.match(/^newmtl\s+(.+)$/i);
    if (newMaterialMatch) {
      flushMaterial();
      currentMaterial = {
        ...(newMaterialMatch[1]?.trim() ? { name: newMaterialMatch[1].trim() } : {}),
      };
      return;
    }

    if (!currentMaterial) {
      return;
    }

    const diffuseMatch = trimmed.match(/^Kd\s+(.+)$/i);
    if (diffuseMatch) {
      const color = parseMtlColorHex(`Kd ${diffuseMatch[1]}`);
      if (color) {
        currentMaterial.color = color;
      }
      return;
    }

    const ambientMatch = trimmed.match(/^Ka\s+(.+)$/i);
    if (ambientMatch && !currentMaterial.color) {
      const color = parseMtlColorHex(`Ka ${ambientMatch[1]}`);
      if (color) {
        currentMaterial.color = color;
      }
      return;
    }

    if (!/^[ \t]*(?:map_|bump|disp)[A-Za-z0-9_ \t-]*/.test(line) || currentMaterial.texture) {
      return;
    }

    const rawTexturePath = parseMtlTexturePath(line);
    if (!rawTexturePath) {
      return;
    }

    const resolvedTextureCandidate = resolveImportedAssetPath(rawTexturePath, materialFilePath);
    const resolvedTexturePath =
      findBestLookupPath(resolvedTextureCandidate || rawTexturePath, textAssetContentLookup) ||
      findBestLookupPath(rawTexturePath, textAssetContentLookup) ||
      normalizeLookupPath(resolvedTextureCandidate || rawTexturePath);
    if (resolvedTexturePath) {
      currentMaterial.texture = resolvedTexturePath;
    }
  });

  flushMaterial();
  return materials;
}

export function deriveObjAuthoredMaterialsFromLookup(
  meshPath: string,
  textAssetContentLookup: TextAssetContentLookup,
): UrdfVisualMaterial[] {
  const normalizedMeshPath = normalizeLookupPath(meshPath);
  if (!normalizedMeshPath) {
    return [];
  }

  const meshEntry = findTextAssetEntry(normalizedMeshPath, textAssetContentLookup);
  if (!meshEntry) {
    return [];
  }

  const authoredMaterials: UrdfVisualMaterial[] = [];
  for (const materialLibrary of parseObjMaterialLibraries(meshEntry.content)) {
    const resolvedMaterialCandidate = resolveImportedAssetPath(materialLibrary, meshEntry.path);
    const materialEntry =
      findTextAssetEntry(resolvedMaterialCandidate || materialLibrary, textAssetContentLookup) ||
      findTextAssetEntry(materialLibrary, textAssetContentLookup);
    if (!materialEntry) {
      continue;
    }

    authoredMaterials.push(
      ...parseMtlAuthoredMaterials(
        materialEntry.content,
        materialEntry.path,
        textAssetContentLookup,
      ),
    );
  }

  return authoredMaterials;
}

export function deriveObjMaterialMetadataFromLookup(
  meshPath: string,
  textAssetContentLookup: TextAssetContentLookup,
): ObjMaterialMetadata | null {
  const [primaryMaterial] = deriveObjAuthoredMaterialsFromLookup(meshPath, textAssetContentLookup);
  if (primaryMaterial) {
    return {
      ...(primaryMaterial.color ? { color: primaryMaterial.color } : {}),
      ...(primaryMaterial.texture ? { texture: primaryMaterial.texture } : {}),
    };
  }

  return null;
}

async function fetchText(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch text asset: ${response.status} ${response.statusText}`);
  }

  return await response.text();
}

async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch binary asset: ${response.status} ${response.statusText}`);
  }

  return await response.arrayBuffer();
}

async function loadObjMaterialCreator(
  materialLibraries: readonly string[],
  manager: THREE.LoadingManager,
  sourcePath?: string | null,
): Promise<ReturnType<MTLLoader['parse']> | null> {
  const rewrittenMaterialTexts: string[] = [];

  for (const materialLibrary of materialLibraries) {
    const resolvedMaterialPath = sourcePath
      ? resolveImportedAssetPath(materialLibrary, sourcePath)
      : normalizeLookupPath(materialLibrary);
    try {
      const materialRequestUrl = manager.resolveURL(resolvedMaterialPath || materialLibrary);
      const materialText = await fetchText(materialRequestUrl);
      const rewrittenMaterialText = rewriteMtlTextureReferencesForManager(
        materialText,
        resolvedMaterialPath || materialLibrary,
      );
      rewrittenMaterialTexts.push(rewrittenMaterialText);
    } catch (error) {
      // Treat referenced material libraries as optional. Missing MTLs (or missing
      // texture sidecars inside them) should not prevent bare OBJ geometry from loading.
      // Surface per-library failures only under regression debugging to avoid noise
      // from OBJs that legitimately reference optional MTLs.
      if (isRegressionDebugEnabled()) {
        console.warn('[ObjMaterial] Failed to load MTL library', {
          materialLibrary,
          resolvedMaterialPath: resolvedMaterialPath || materialLibrary,
          error,
        });
      }
    }
  }

  if (rewrittenMaterialTexts.length === 0) {
    // High-signal diagnostic for the "imported model is all white" symptom: the OBJ
    // declared material libraries but none of them could be loaded, so it falls back
    // to bare/vertex-color materials.
    if (materialLibraries.length > 0) {
      console.warn(
        `[ObjMaterial] OBJ declared ${materialLibraries.length} material ` +
          `${materialLibraries.length === 1 ? 'library' : 'libraries'} but none could be loaded; ` +
          'falling back to bare/vertex-color material (mesh may render white). ' +
          `Libraries: ${materialLibraries.join(', ')}`,
      );
    }
    return null;
  }

  const materials = new MTLLoader(manager).parse(rewrittenMaterialTexts.join('\n\n'), '');
  materials.preload();
  return materials;
}

function normalizeObjVertexColorMaterials(root: THREE.Object3D): void {
  root.traverse((child) => {
    const renderable = child as THREE.Mesh | THREE.LineSegments | THREE.Points;
    if (!renderable.geometry || !('material' in renderable)) {
      return;
    }

    const hasVertexColorAttribute = Boolean(renderable.geometry.getAttribute('color'));
    if (!hasVertexColorAttribute) {
      return;
    }

    const materials = Array.isArray(renderable.material)
      ? renderable.material
      : [renderable.material];

    materials.forEach((material) => {
      if (!material) {
        return;
      }

      (material as { vertexColors?: boolean }).vertexColors = true;
      const materialColor = (material as { color?: THREE.Color }).color;
      if (materialColor?.isColor) {
        materialColor.set(0xffffff);
      }
      material.toneMapped = false;
      material.userData = {
        ...(material.userData ?? {}),
        usesVertexColors: true,
      };
      material.needsUpdate = true;
    });
  });
}

function cloneMtlMaterialForRenderable(
  sourceMaterial: THREE.Material,
  renderable: THREE.Mesh | THREE.LineSegments | THREE.Points,
): THREE.Material {
  const sourceColor = (sourceMaterial as THREE.Material & { color?: THREE.Color }).color;
  const color = sourceColor instanceof THREE.Color ? sourceColor : new THREE.Color(0xffffff);

  if ((renderable as THREE.LineSegments).isLineSegments) {
    const material = new THREE.LineBasicMaterial({ color });
    material.name = sourceMaterial.name;
    material.userData = { ...(sourceMaterial.userData ?? {}) };
    return material;
  }

  if ((renderable as THREE.Points).isPoints) {
    const material = new THREE.PointsMaterial({
      color,
      size: 10,
      sizeAttenuation: false,
    });
    material.name = sourceMaterial.name;
    material.userData = { ...(sourceMaterial.userData ?? {}) };
    const sourceMap = (sourceMaterial as THREE.MeshPhongMaterial).map;
    if (sourceMap) {
      material.map = sourceMap;
    }
    return material;
  }

  return sourceMaterial.clone();
}

interface MtlMaterialNameLookup {
  exactNames: Set<string>;
  normalizedNameToOriginal: Map<string, string>;
  soleMaterialName: string | null;
}

function buildMtlMaterialNameLookup(
  materialCreator: ReturnType<MTLLoader['parse']>,
): MtlMaterialNameLookup {
  const materialNames = Object.keys(materialCreator.materialsInfo ?? {});
  const normalizedNameToOriginal = new Map<string, string>();
  for (const materialName of materialNames) {
    const normalizedName = materialName.trim().toLowerCase();
    if (normalizedName && !normalizedNameToOriginal.has(normalizedName)) {
      normalizedNameToOriginal.set(normalizedName, materialName);
    }
  }

  return {
    exactNames: new Set(materialNames),
    normalizedNameToOriginal,
    soleMaterialName: materialNames.length === 1 ? (materialNames[0] ?? null) : null,
  };
}

function renderableHasVertexColors(
  renderable: THREE.Mesh | THREE.LineSegments | THREE.Points,
): boolean {
  return Boolean(renderable.geometry?.getAttribute('color'));
}

// Match an OBJ `usemtl` group name against the MTL `newmtl` definitions, tolerating
// the common real-world mismatches that otherwise leave models rendering white:
// exact name, then trimmed/case-insensitive, then a single-material fallback for OBJs
// exported without a usable `usemtl` (or with a name the MTL does not declare).
function resolveMtlMaterialInfoName(
  generatedName: string | undefined,
  lookup: MtlMaterialNameLookup,
  options: { allowSoleMaterialFallback: boolean },
): string | null {
  const trimmedName = generatedName?.trim() ?? '';
  if (trimmedName && lookup.exactNames.has(trimmedName)) {
    return trimmedName;
  }

  if (trimmedName) {
    const normalizedMatch = lookup.normalizedNameToOriginal.get(trimmedName.toLowerCase());
    if (normalizedMatch) {
      return normalizedMatch;
    }
  }

  if (options.allowSoleMaterialFallback && lookup.soleMaterialName) {
    return lookup.soleMaterialName;
  }

  return null;
}

function createMtlMaterialForGeneratedMaterial(
  materialCreator: ReturnType<MTLLoader['parse']>,
  generatedMaterial: THREE.Material,
  renderable: THREE.Mesh | THREE.LineSegments | THREE.Points,
  lookup: MtlMaterialNameLookup,
): THREE.Material | null {
  const resolvedMaterialName = resolveMtlMaterialInfoName(generatedMaterial.name, lookup, {
    // Vertex-colored meshes intentionally fall through to normalizeObjVertexColorMaterials,
    // so never force a single-material MTL onto them.
    allowSoleMaterialFallback: !renderableHasVertexColors(renderable),
  });
  if (!resolvedMaterialName) {
    return null;
  }

  const sourceMaterial = materialCreator.create(resolvedMaterialName);
  if (!sourceMaterial) {
    return null;
  }

  const renderableMaterial = cloneMtlMaterialForRenderable(sourceMaterial, renderable);
  // Preserve the OBJ-assigned name so later name-based lookups stay stable even when we
  // matched via the case-insensitive or single-material fallback path.
  if (generatedMaterial.name) {
    renderableMaterial.name = generatedMaterial.name;
  }
  return renderableMaterial;
}

function applyMtlMaterialsToRenderable(
  renderable: THREE.Mesh | THREE.LineSegments | THREE.Points,
  materialCreator: ReturnType<MTLLoader['parse']>,
  lookup: MtlMaterialNameLookup,
): void {
  const originalMaterials = Array.isArray(renderable.material)
    ? renderable.material
    : [renderable.material];
  let didReplace = false;

  const nextMaterials = originalMaterials.map((material) => {
    const nextMaterial = createMtlMaterialForGeneratedMaterial(
      materialCreator,
      material,
      renderable,
      lookup,
    );
    if (!nextMaterial) {
      return material;
    }

    didReplace = true;
    return nextMaterial;
  });

  if (!didReplace) {
    return;
  }

  renderable.material = Array.isArray(renderable.material) ? nextMaterials : nextMaterials[0];
}

export async function applyObjMaterialLibrariesToObject(
  root: THREE.Object3D,
  materialLibraries: readonly string[],
  manager: THREE.LoadingManager,
  sourcePath?: string | null,
  options: { yieldIfNeeded?: () => Promise<void> } = {},
): Promise<void> {
  if (materialLibraries.length === 0) {
    await options.yieldIfNeeded?.();
    normalizeObjVertexColorMaterials(root);
    return;
  }

  const materialCreator = await loadObjMaterialCreator(materialLibraries, manager, sourcePath);
  if (!materialCreator) {
    await options.yieldIfNeeded?.();
    normalizeObjVertexColorMaterials(root);
    return;
  }

  const materialNameLookup = buildMtlMaterialNameLookup(materialCreator);
  await options.yieldIfNeeded?.();
  root.traverse((child) => {
    const renderable = child as THREE.Mesh | THREE.LineSegments | THREE.Points;
    if (!renderable.geometry || !('material' in renderable)) {
      return;
    }

    applyMtlMaterialsToRenderable(renderable, materialCreator, materialNameLookup);
  });

  normalizeObjVertexColorMaterials(root);
  await options.yieldIfNeeded?.();
}

export async function loadObjScene(
  assetUrl: string,
  manager: THREE.LoadingManager,
  sourcePath?: string | null,
): Promise<THREE.Group> {
  const requestUrl = manager.resolveURL(assetUrl);
  const serializedObject = await parseObjModelDataFromBytes(await fetchArrayBuffer(requestUrl));
  const object = createObjectFromSerializedObjData(serializedObject);
  const materialStartedAt = readHighResolutionEpochMs();
  await applyObjMaterialLibrariesToObject(
    object,
    serializedObject.materialLibraries,
    manager,
    sourcePath,
  );
  addMainThreadMaterialPerformance(serializedObject.loadPerformance, durationMs(materialStartedAt));
  if (serializedObject.loadPerformance) {
    object.userData = {
      ...(object.userData ?? {}),
      meshLoadPerformance: serializedObject.loadPerformance,
    };
  }
  return object;
}

function cloneTextureWithOwnedInstance<TValue>(value: TValue): TValue {
  if (!(value instanceof THREE.Texture)) {
    return value;
  }

  const clonedTexture = value.clone();
  clonedTexture.needsUpdate = true;
  return clonedTexture as TValue;
}

function cloneMaterialWithOwnedTextures<TMaterial extends THREE.Material>(
  material: TMaterial,
): TMaterial {
  const clonedMaterial = material.clone() as TMaterial;
  clonedMaterial.userData = {
    ...(material.userData ?? {}),
    ...(clonedMaterial.userData ?? {}),
  };

  MATERIAL_TEXTURE_PROPERTIES.forEach((property) => {
    const texture = (material as Record<string, unknown>)[property];
    if (texture instanceof THREE.Texture) {
      (clonedMaterial as Record<string, unknown>)[property] =
        cloneTextureWithOwnedInstance(texture);
    }
  });

  return clonedMaterial;
}

function cloneGeometryWithSharedAttributeStorage(
  geometry: THREE.BufferGeometry,
): THREE.BufferGeometry {
  const clonedGeometry = new THREE.BufferGeometry();
  const index = geometry.getIndex();

  clonedGeometry.name = geometry.name;
  clonedGeometry.userData = { ...(geometry.userData ?? {}) };
  clonedGeometry.drawRange = { ...geometry.drawRange };
  clonedGeometry.morphTargetsRelative = geometry.morphTargetsRelative;
  clonedGeometry.morphAttributes = { ...geometry.morphAttributes };

  if (index) {
    clonedGeometry.setIndex(index);
  }

  Object.entries(geometry.attributes).forEach(([name, attribute]) => {
    clonedGeometry.setAttribute(name, attribute);
  });

  geometry.groups.forEach((group) => {
    clonedGeometry.addGroup(group.start, group.count, group.materialIndex ?? 0);
  });

  if (geometry.boundingBox) {
    clonedGeometry.boundingBox = geometry.boundingBox.clone();
  }
  if (geometry.boundingSphere) {
    clonedGeometry.boundingSphere = geometry.boundingSphere.clone();
  }

  return clonedGeometry;
}

export function cloneObjSceneWithOwnedResources<TObject extends THREE.Object3D>(
  source: TObject,
): TObject {
  const clonedRoot = source.clone(true) as TObject;

  clonedRoot.traverse((child) => {
    const renderable = child as THREE.Mesh | THREE.LineSegments | THREE.Points;
    if (!renderable.geometry || !('material' in renderable)) {
      return;
    }

    renderable.geometry = cloneGeometryWithSharedAttributeStorage(renderable.geometry);

    if (Array.isArray(renderable.material)) {
      renderable.material = renderable.material.map((material) =>
        cloneMaterialWithOwnedTextures(material),
      );
      return;
    }

    if (renderable.material) {
      renderable.material = cloneMaterialWithOwnedTextures(renderable.material);
    }
  });

  normalizeObjVertexColorMaterials(clonedRoot);
  return clonedRoot;
}
