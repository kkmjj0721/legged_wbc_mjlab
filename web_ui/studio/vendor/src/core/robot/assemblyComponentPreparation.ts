import type { RobotData, RobotFile } from '@/types';
import { rewriteRobotMeshPathsForSource } from '@/core/parsers/meshPathUtils';
import { syncRobotVisualColorsFromMaterials } from './materials';

const GENERIC_ASSEMBLY_COMPONENT_FILE_STEMS = new Set(['model', 'robot', 'scene']);

function sanitizeAssemblyComponentBaseName(value: string | null | undefined): string | null {
  const sanitized = String(value || '')
    .trim()
    .replace(/[^a-zA-Z0-9_]/g, '_');
  return sanitized || null;
}

function getPathSegments(fileName: string): string[] {
  return String(fileName || '')
    .replace(/\\/g, '/')
    .split('/')
    .filter(Boolean);
}

function getFileStem(fileName: string): string {
  const lastSegment = getPathSegments(fileName).pop() ?? '';
  return lastSegment.replace(/\.[^/.]+$/, '');
}

function resolveAssemblyComponentPathBaseName(fileName: string): string {
  const segments = getPathSegments(fileName);
  const fileStem = getFileStem(fileName);
  const parentSegment = segments.length > 1 ? segments[segments.length - 2] : '';
  const shouldPreferParent =
    Boolean(parentSegment) &&
    GENERIC_ASSEMBLY_COMPONENT_FILE_STEMS.has(fileStem.trim().toLowerCase());

  return (
    sanitizeAssemblyComponentBaseName(shouldPreferParent ? parentSegment : fileStem) ??
    sanitizeAssemblyComponentBaseName(parentSegment) ??
    'robot'
  );
}

function extractTagAttribute(
  source: string,
  tagPattern: string,
  attributeName: string,
): string | null {
  const tagMatch = source.match(new RegExp(`<\\s*${tagPattern}\\b[^>]*>`, 'i'));
  const tag = tagMatch?.[0];
  if (!tag) {
    return null;
  }

  const attributeMatch = tag.match(new RegExp(`\\b${attributeName}\\s*=\\s*(["'])(.*?)\\1`, 'i'));
  return attributeMatch?.[2]?.trim() || null;
}

function extractAssemblyComponentSourceName(
  content: string | null | undefined,
  format?: RobotFile['format'] | null,
): string | null {
  const source = String(content || '');
  if (!source.trim()) {
    return null;
  }

  switch (format) {
    case 'sdf':
      return (
        extractTagAttribute(source, 'model', 'name') ?? extractTagAttribute(source, 'world', 'name')
      );
    case 'mjcf':
      return extractTagAttribute(source, 'mujoco', 'model');
    case 'urdf':
    case 'xacro':
      return extractTagAttribute(source, '(?:[\\w.-]+:)?robot', 'name');
    default:
      return (
        extractTagAttribute(source, 'model', 'name') ??
        extractTagAttribute(source, '(?:[\\w.-]+:)?robot', 'name') ??
        extractTagAttribute(source, 'mujoco', 'model')
      );
  }
}

export function sanitizeAssemblyComponentId(filename: string): string {
  return sanitizeAssemblyComponentBaseName(getFileStem(filename)) ?? 'robot';
}

export function resolveAssemblyComponentBaseName(
  file: Pick<RobotFile, 'name' | 'content' | 'format'>,
  fallbackName?: string | null,
): string {
  return (
    sanitizeAssemblyComponentBaseName(
      extractAssemblyComponentSourceName(file.content, file.format),
    ) ??
    sanitizeAssemblyComponentBaseName(fallbackName) ??
    resolveAssemblyComponentPathBaseName(file.name)
  );
}

export function createUniqueAssemblyComponentName(
  baseName: string,
  existingNames: Set<string>,
): string {
  if (!existingNames.has(baseName)) {
    return baseName;
  }

  let suffix = 1;
  let candidate = `${baseName}_${suffix}`;
  while (existingNames.has(candidate)) {
    suffix += 1;
    candidate = `${baseName}_${suffix}`;
  }
  return candidate;
}

export function buildAssemblyComponentIdentity({
  fileName,
  baseName,
  existingComponentIds,
  existingComponentNames,
}: {
  fileName: string;
  baseName?: string | null;
  existingComponentIds: Iterable<string>;
  existingComponentNames: Iterable<string>;
}): {
  componentId: string;
  displayName: string;
} {
  const baseId =
    sanitizeAssemblyComponentBaseName(baseName) ?? resolveAssemblyComponentPathBaseName(fileName);
  const existingNameSet = new Set(existingComponentNames);
  const displayName = createUniqueAssemblyComponentName(baseId, existingNameSet);
  const existingIdSet = new Set(existingComponentIds);

  let componentId = `comp_${displayName}`;
  let suffix = 1;
  while (existingIdSet.has(componentId)) {
    componentId = `comp_${displayName}_${suffix++}`;
  }

  return {
    componentId,
    displayName,
  };
}

export function normalizeComponentRobot(
  data: RobotData,
  options: {
    sourceFilePath?: string | null;
    sourceFormat?: RobotFile['format'] | null;
  } = {},
): RobotData {
  const sourceRobotData = structuredClone(data);
  const pathNormalizedRobot = options.sourceFormat === 'usd'
    ? rewriteRobotMeshPathsForSource(sourceRobotData, options.sourceFilePath)
    : sourceRobotData;
  return syncRobotVisualColorsFromMaterials(pathNormalizedRobot);
}

export function prepareAssemblyRobotData(
  data: RobotData,
  options: {
    sourceFilePath?: string | null;
    sourceFormat?: RobotFile['format'] | null;
  } = {},
): RobotData {
  return normalizeComponentRobot(data, options);
}
