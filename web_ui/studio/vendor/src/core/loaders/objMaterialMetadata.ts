import { resolveImportedAssetPath } from '@/core/parsers/meshPathUtils';
import type { UrdfVisualMaterial } from '@/types';

function normalizeLookupPath(value: string | null | undefined): string {
  return (
    String(value || '')
      .trim()
      .replace(/\\/g, '/')
      .replace(/^\/+/, '')
      .split('?')[0] || ''
  );
}

function parseNumberTuple(text: string | null | undefined): number[] {
  return (text ?? '')
    .trim()
    .split(/\s+/)
    .map((value) => Number.parseFloat(value))
    .filter((value) => Number.isFinite(value));
}

function rgbaTextToHex(text: string | null | undefined): string | undefined {
  const [r, g, b] = parseNumberTuple(text);
  if (![r, g, b].every((value) => Number.isFinite(value))) {
    return undefined;
  }

  const toByte = (value: number) => Math.max(0, Math.min(255, Math.round(value * 255)));
  return `#${[toByte(r), toByte(g), toByte(b)]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('')}`;
}

function parseTextureReference(value: string): string | undefined {
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }

  const tokens = trimmed.split(/\s+/);
  for (let index = tokens.length - 1; index >= 0; index -= 1) {
    const candidate = tokens.slice(index).join(' ').trim();
    if (candidate && !candidate.startsWith('-')) {
      return candidate;
    }
  }

  return undefined;
}

function buildAvailablePathSet(availableAssetPaths: readonly string[]): Set<string> {
  return new Set(availableAssetPaths.map((entry) => normalizeLookupPath(entry)).filter(Boolean));
}

function resolveAvailableAssetPath(
  requestedPath: string,
  sourceFilePath: string,
  availableAssetPaths: Set<string>,
): string | undefined {
  const resolvedPath = normalizeLookupPath(resolveImportedAssetPath(requestedPath, sourceFilePath));
  if (resolvedPath && availableAssetPaths.has(resolvedPath)) {
    return resolvedPath;
  }

  const normalizedRequestedPath = normalizeLookupPath(requestedPath);
  if (normalizedRequestedPath && availableAssetPaths.has(normalizedRequestedPath)) {
    return normalizedRequestedPath;
  }

  const requestedBasename = normalizedRequestedPath.split('/').pop();
  if (!requestedBasename) {
    return resolvedPath || normalizedRequestedPath || undefined;
  }

  const basenameMatches = Array.from(availableAssetPaths).filter(
    (entry) => entry.split('/').pop() === requestedBasename,
  );
  if (basenameMatches.length === 1) {
    return basenameMatches[0];
  }

  return resolvedPath || normalizedRequestedPath || undefined;
}

export function parseObjMaterialLibraries(content: string): string[] {
  const materialLibraries: string[] = [];
  const matches = content.matchAll(/^[ \t]*mtllib[ \t]+(.+)$/gim);
  for (const match of matches) {
    const rawValue = String(match[1] || '').trim();
    if (!rawValue) {
      continue;
    }

    materialLibraries.push(rawValue);
  }

  return materialLibraries;
}

function parseObjMaterialUsageOrder(content: string): string[] {
  const orderedMaterials: string[] = [];
  const seen = new Set<string>();
  const matches = content.matchAll(/^[ \t]*usemtl[ \t]+(.+)$/gim);
  for (const match of matches) {
    const materialName = String(match[1] || '').trim();
    if (!materialName || seen.has(materialName)) {
      continue;
    }

    seen.add(materialName);
    orderedMaterials.push(materialName);
  }

  return orderedMaterials;
}

function parseMtlMaterialLibrary(
  content: string,
  materialFilePath: string,
  availableAssetPaths: Set<string>,
): Map<string, UrdfVisualMaterial> {
  const materialEntries = new Map<string, UrdfVisualMaterial>();
  let currentName = '';
  let currentEntry: UrdfVisualMaterial | null = null;

  const flushCurrentEntry = () => {
    if (!currentName || !currentEntry) {
      currentName = '';
      currentEntry = null;
      return;
    }

    const normalizedEntry: UrdfVisualMaterial = {
      ...(currentEntry.name ? { name: currentEntry.name } : {}),
      ...(currentEntry.color ? { color: currentEntry.color } : {}),
      ...(currentEntry.texture ? { texture: currentEntry.texture } : {}),
    };
    if (normalizedEntry.name || normalizedEntry.color || normalizedEntry.texture) {
      materialEntries.set(currentName, normalizedEntry);
    }

    currentName = '';
    currentEntry = null;
  };

  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) {
      continue;
    }

    const newMaterialMatch = trimmed.match(/^newmtl\s+(.+)$/i);
    if (newMaterialMatch) {
      flushCurrentEntry();
      currentName = String(newMaterialMatch[1] || '').trim();
      currentEntry = currentName ? { name: currentName } : null;
      continue;
    }

    if (!currentEntry) {
      continue;
    }

    const diffuseTextureMatch = trimmed.match(/^map_Kd\s+(.+)$/i);
    if (diffuseTextureMatch) {
      const textureReference = parseTextureReference(diffuseTextureMatch[1] || '');
      if (textureReference) {
        currentEntry.texture =
          resolveAvailableAssetPath(textureReference, materialFilePath, availableAssetPaths) ||
          textureReference;
      }
      continue;
    }

    const ambientTextureMatch = trimmed.match(/^map_Ka\s+(.+)$/i);
    if (ambientTextureMatch && !currentEntry.texture) {
      const textureReference = parseTextureReference(ambientTextureMatch[1] || '');
      if (textureReference) {
        currentEntry.texture =
          resolveAvailableAssetPath(textureReference, materialFilePath, availableAssetPaths) ||
          textureReference;
      }
      continue;
    }

    const diffuseColorMatch = trimmed.match(/^Kd\s+(.+)$/i);
    if (diffuseColorMatch) {
      currentEntry.color = rgbaTextToHex(diffuseColorMatch[1] || '') || currentEntry.color;
      continue;
    }

    const ambientColorMatch = trimmed.match(/^Ka\s+(.+)$/i);
    if (ambientColorMatch && !currentEntry.color) {
      currentEntry.color = rgbaTextToHex(ambientColorMatch[1] || '') || currentEntry.color;
    }
  }

  flushCurrentEntry();
  return materialEntries;
}

export function resolveObjAuthoredMaterialsFromAssets(
  meshPath: string,
  allFileContents: Readonly<Record<string, string>>,
  availableAssetPaths: readonly string[],
): UrdfVisualMaterial[] {
  const availablePathSet = buildAvailablePathSet(availableAssetPaths);
  const resolvedMeshPath = resolveAvailableAssetPath(meshPath, meshPath, availablePathSet);
  const meshContent = resolvedMeshPath ? allFileContents[resolvedMeshPath] : undefined;
  if (!meshContent) {
    return [];
  }

  const usageOrder = parseObjMaterialUsageOrder(meshContent);
  const materialLibraries = parseObjMaterialLibraries(meshContent);
  if (materialLibraries.length === 0) {
    return [];
  }

  const materialsByName = new Map<string, UrdfVisualMaterial>();
  for (const materialLibrary of materialLibraries) {
    const materialPath = resolveAvailableAssetPath(
      materialLibrary,
      resolvedMeshPath || meshPath,
      availablePathSet,
    );
    if (!materialPath) {
      continue;
    }

    const materialContent = allFileContents[materialPath];
    if (!materialContent) {
      continue;
    }

    parseMtlMaterialLibrary(materialContent, materialPath, availablePathSet).forEach(
      (entry, name) => {
        materialsByName.set(name, entry);
      },
    );
  }

  if (materialsByName.size === 0) {
    return [];
  }

  const orderedNames = [
    ...usageOrder.filter((name) => materialsByName.has(name)),
    ...Array.from(materialsByName.keys()).filter((name) => !usageOrder.includes(name)),
  ];

  return orderedNames
    .map((name) => materialsByName.get(name))
    .filter((entry): entry is UrdfVisualMaterial => Boolean(entry));
}
