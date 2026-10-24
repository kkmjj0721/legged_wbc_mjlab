import type { GazeboMaterialPass, UrdfVisualMaterial } from '@/types';
import { resolveImportedAssetPath } from '@/core/parsers/meshPathUtils';
import { normalizeRelativePath } from '@/core/utils/pathNormalization';

export interface GazeboScriptMaterialDefinition extends UrdfVisualMaterial {}

interface ResolveGazeboScriptMaterialOptions {
  allFileContents?: Record<string, string>;
  scriptName?: string;
  scriptUris?: string[];
  sourcePath?: string;
}

const TEXTURE_DIRECTORY_PATTERN = /(?:^|\/)(?:textures?|materials\/textures)(?:\/|$)/i;
const materialScriptCache = new Map<string, Record<string, GazeboScriptMaterialDefinition>>();

function uniquePaths(paths: Array<string | null | undefined>): string[] {
  return Array.from(
    new Set(paths.map((path) => normalizeRelativePath(String(path || '').trim())).filter(Boolean)),
  );
}

function resolveScriptUriPath(uri: string, sourcePath?: string): string | null {
  const trimmed = uri.trim();
  if (!trimmed) {
    return null;
  }

  if (/^https?:\/\//i.test(trimmed) || trimmed.startsWith('data:')) {
    return trimmed;
  }

  if (trimmed.startsWith('file://')) {
    return normalizeRelativePath(trimmed.slice('file://'.length));
  }

  return normalizeRelativePath(resolveImportedAssetPath(trimmed, sourcePath));
}

function parseMaterialColor(
  value: string,
): { color: string; colorRgba?: [number, number, number, number] } | undefined {
  const parts = value.trim().split(/\s+/);
  const numericParts = parts
    .map((part) => Number.parseFloat(part))
    .filter((part) => Number.isFinite(part));

  if (numericParts.length < 3) {
    return undefined;
  }

  const isUnitRange = numericParts.slice(0, 3).every((part) => part >= 0 && part <= 1);
  const normalizeChannel = (part: number) => {
    const normalized = isUnitRange ? part : part / 255;
    return Math.max(0, Math.min(1, normalized));
  };
  const toByte = (part: number) => {
    const normalized = isUnitRange ? part * 255 : part;
    return Math.max(0, Math.min(255, Math.round(normalized)));
  };
  const normalizeAlpha = (part: number) => {
    const normalized = part >= 0 && part <= 1 ? part : part / 255;
    return Math.max(0, Math.min(1, normalized));
  };

  const color = `#${numericParts
    .slice(0, 3)
    .map((part) => toByte(part).toString(16).padStart(2, '0'))
    .join('')}`;
  const alpha = numericParts[3];
  const colorRgba =
    numericParts.length >= 4 && Number.isFinite(alpha)
      ? ([
          normalizeChannel(numericParts[0]!),
          normalizeChannel(numericParts[1]!),
          normalizeChannel(numericParts[2]!),
          normalizeAlpha(alpha!),
        ] as [number, number, number, number])
      : undefined;

  return {
    color,
    ...(colorRgba ? { colorRgba } : {}),
  };
}

function readMaterialBlock(
  text: string,
  openingBraceIndex: number,
): { block: string; endIndex: number } | null {
  let depth = 1;

  for (let index = openingBraceIndex + 1; index < text.length; index += 1) {
    const char = text[index];
    if (char === '{') {
      depth += 1;
    } else if (char === '}') {
      depth -= 1;
      if (depth === 0) {
        return {
          block: text.slice(openingBraceIndex + 1, index),
          endIndex: index + 1,
        };
      }
    }
  }

  return null;
}

function parsePassesFromBlock(block: string): GazeboMaterialPass[] {
  const passes: GazeboMaterialPass[] = [];
  const passPattern = /\bpass\b/gi;
  let passMatch = passPattern.exec(block);

  while (passMatch) {
    const passOpeningBrace = block.indexOf('{', passMatch.index);
    if (passOpeningBrace < 0) {
      break;
    }

    const passResult = readMaterialBlock(block, passOpeningBrace);
    if (!passResult) {
      break;
    }

    const passBlock = passResult.block;
    const textureMatch = passBlock.match(/\btexture\s+([^\s{}]+)/i);
    const blendMatch = passBlock.match(/\bscene_blend\s+(\w+)/i);
    const depthWriteMatch = passBlock.match(/\bdepth_write\s+(\w+)/i);
    const lightingMatch = passBlock.match(/\blighting\s+(\w+)/i);

    const pass: GazeboMaterialPass = {};
    if (textureMatch) {
      pass.texture = textureMatch[1]?.trim();
    }
    if (blendMatch) {
      const mode = blendMatch[1]?.trim();
      if (mode === 'alpha_blend' || mode === 'add' || mode === 'modulate') {
        pass.sceneBlend = mode;
      }
    }
    if (depthWriteMatch) {
      pass.depthWrite = depthWriteMatch[1]?.trim() !== 'off';
    }
    if (lightingMatch) {
      pass.lighting = lightingMatch[1]?.trim() !== 'off';
    }

    passes.push(pass);
    passPattern.lastIndex = passResult.endIndex;
    passMatch = passPattern.exec(block);
  }

  return passes;
}

function parseMaterialScript(text: string): Record<string, GazeboScriptMaterialDefinition> {
  const cached = materialScriptCache.get(text);
  if (cached) {
    return cached;
  }

  const definitions: Record<string, GazeboScriptMaterialDefinition> = {};
  const materialHeaderPattern = /\bmaterial\s+([^\s{]+)\s*\{/g;
  let match = materialHeaderPattern.exec(text);

  while (match) {
    const materialName = match[1]?.trim();
    const openingBraceIndex = text.indexOf('{', match.index);
    if (!materialName || openingBraceIndex < 0) {
      match = materialHeaderPattern.exec(text);
      continue;
    }

    const blockResult = readMaterialBlock(text, openingBraceIndex);
    if (!blockResult) {
      break;
    }

    const block = blockResult.block;
    const texture = block.match(/\btexture\s+([^\s{}]+)/i)?.[1]?.trim();
    const diffuse = parseMaterialColor(block.match(/\bdiffuse\s+([^\r\n{}]+)/i)?.[1] || '');
    const ambient = parseMaterialColor(block.match(/\bambient\s+([^\r\n{}]+)/i)?.[1] || '');
    const colorDefinition = diffuse || ambient;

    const alphaRejectionMatch = block.match(
      /\balpha_rejection\s+(?:greater|greater_equal|less|less_equal|not_equal)\s+(\d+)/i,
    );
    const alphaTest = alphaRejectionMatch
      ? Math.min(1, Math.max(0, Number.parseFloat(alphaRejectionMatch[1]) / 255))
      : undefined;

    const passes = parsePassesFromBlock(block);

    definitions[materialName] = {
      name: materialName,
      ...(colorDefinition?.color ? { color: colorDefinition.color } : {}),
      ...(colorDefinition?.colorRgba ? { colorRgba: colorDefinition.colorRgba } : {}),
      ...(texture ? { texture } : {}),
      ...(alphaTest !== undefined ? { alphaTest } : {}),
      ...(passes.length > 1 ? { passes } : {}),
    };

    materialHeaderPattern.lastIndex = blockResult.endIndex;
    match = materialHeaderPattern.exec(text);
  }

  materialScriptCache.set(text, definitions);
  return definitions;
}

function collectMaterialScriptFiles(
  allFileContents: Record<string, string>,
  searchRoots: string[],
): Array<{ path: string; content: string }> {
  const entries = Object.entries(allFileContents)
    .map(([path, content]) => ({ path: normalizeRelativePath(path), content }))
    .filter(({ path }) => path.toLowerCase().endsWith('.material'));

  if (searchRoots.length === 0) {
    return entries;
  }

  const filteredEntries = entries.filter(({ path }) =>
    searchRoots.some((root) =>
      root.toLowerCase().endsWith('.material')
        ? path === root
        : path === root || path.startsWith(`${root}/`),
    ),
  );

  return filteredEntries.length > 0 ? filteredEntries : entries;
}

function resolveTexturePath(
  rawTexturePath: string | undefined,
  searchRoots: string[],
  sourcePath?: string,
): string | undefined {
  const trimmed = String(rawTexturePath || '').trim();
  if (!trimmed) {
    return undefined;
  }

  if (/^https?:\/\//i.test(trimmed) || trimmed.startsWith('data:')) {
    return trimmed;
  }

  if (trimmed.startsWith('file://')) {
    return normalizeRelativePath(trimmed.slice('file://'.length));
  }

  if (trimmed.startsWith('model://')) {
    return normalizeRelativePath(resolveImportedAssetPath(trimmed, sourcePath));
  }

  // Expand search roots with derived texture directories.
  // Gazebo models typically place textures in materials/textures/ while
  // the SDF <uri> points to materials/scripts/. Ogre3D's resource system
  // searches all registered paths, so we must mimic this by adding sibling
  // texture directories as candidates.
  const expandedRoots = [...searchRoots];

  for (const root of searchRoots) {
    const normalizedRoot = normalizeRelativePath(root);
    if (!normalizedRoot) {
      continue;
    }

    const lastSlash = normalizedRoot.lastIndexOf('/');
    if (lastSlash > 0) {
      const parentDir = normalizedRoot.slice(0, lastSlash);
      expandedRoots.push(`${parentDir}/textures`);
      expandedRoots.push(`${parentDir}/texture`);
    }

    const firstSlash = normalizedRoot.indexOf('/');
    if (firstSlash > 0) {
      const modelRoot = normalizedRoot.slice(0, firstSlash);
      expandedRoots.push(`${modelRoot}/materials/textures`);
      expandedRoots.push(`${modelRoot}/textures`);
    }
  }

  const sortedRoots = [...new Set(expandedRoots)].sort((left, right) => {
    const leftScore = TEXTURE_DIRECTORY_PATTERN.test(left) ? 0 : 1;
    const rightScore = TEXTURE_DIRECTORY_PATTERN.test(right) ? 0 : 1;
    return leftScore - rightScore;
  });

  for (const root of sortedRoots) {
    const normalizedRoot = normalizeRelativePath(root);
    const baseDirectory = normalizedRoot.toLowerCase().endsWith('.material')
      ? normalizedRoot.slice(0, Math.max(0, normalizedRoot.lastIndexOf('/')))
      : normalizedRoot;
    const resolved = normalizeRelativePath(
      `${baseDirectory.replace(/\/+$/, '')}/${trimmed.replace(/^\/+/, '')}`,
    );
    if (resolved && resolved !== trimmed) {
      return resolved;
    }
  }

  return normalizeRelativePath(resolveImportedAssetPath(trimmed, sourcePath));
}

export function resolveGazeboScriptMaterial({
  allFileContents = {},
  scriptName,
  scriptUris = [],
  sourcePath,
}: ResolveGazeboScriptMaterialOptions): GazeboScriptMaterialDefinition | null {
  const normalizedScriptName = String(scriptName || '').trim();
  if (!normalizedScriptName) {
    return null;
  }

  const searchRoots = uniquePaths(scriptUris.map((uri) => resolveScriptUriPath(uri, sourcePath)));
  const scriptFiles = collectMaterialScriptFiles(allFileContents, searchRoots);

  for (const { content } of scriptFiles) {
    const parsedDefinitions = parseMaterialScript(content);
    const matchedDefinition = parsedDefinitions[normalizedScriptName];
    if (!matchedDefinition) {
      continue;
    }

    return {
      ...matchedDefinition,
      ...(matchedDefinition.texture
        ? {
            texture:
              resolveTexturePath(matchedDefinition.texture, searchRoots, sourcePath) ||
              matchedDefinition.texture,
          }
        : {}),
      ...(matchedDefinition.passes
        ? {
            passes: matchedDefinition.passes.map((pass) => ({
              ...pass,
              ...(pass.texture
                ? {
                    texture:
                      resolveTexturePath(pass.texture, searchRoots, sourcePath) || pass.texture,
                  }
                : {}),
            })),
          }
        : {}),
    };
  }

  return null;
}
