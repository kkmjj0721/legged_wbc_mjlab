import { GeometryType, type UrdfLink, type UrdfVisual } from '@/types';
import { buildMeshLookupCandidates } from '@/core/parsers/meshPathUtils';
import { getVisualGeometryEntries } from '@/core/robot';

import { cleanFilePath } from './pathNormalization';

const ORIGIN_EPSILON = 1e-6;

export interface ColladaRootNormalizationHints {
  exactMeshPaths: Set<string>;
  packageRelativeMeshPaths: Set<string>;
}

function normalizeMeshPath(path: string): string {
  return cleanFilePath(
    path
      .replace(/^package:\/\//i, '')
      .replace(/^\/+/, '')
  );
}

function extractPackageName(path: string): string | null {
  const slashIndex = path.indexOf('/');
  if (slashIndex <= 0) {
    return null;
  }

  return path.slice(0, slashIndex);
}

function getPackageRelativePath(path: string): string {
  const slashIndex = path.indexOf('/');
  if (slashIndex === -1) {
    return path;
  }

  return path.slice(slashIndex + 1);
}

function hasNonZeroRotationOrigin(origin: UrdfVisual['origin'] | undefined): boolean {
  if (!origin) {
    return false;
  }

  return (
    Math.abs(origin.rpy.r) > ORIGIN_EPSILON ||
    Math.abs(origin.rpy.p) > ORIGIN_EPSILON ||
    Math.abs(origin.rpy.y) > ORIGIN_EPSILON
  );
}

export function shouldNormalizeColladaGeometry(
  meshPath: string | undefined,
  origin: UrdfVisual['origin'] | undefined,
  hints?: ColladaRootNormalizationHints | null,
): boolean {
  if (!meshPath?.toLowerCase().endsWith('.dae')) {
    return false;
  }

  if (hints && shouldNormalizeColladaRoot(meshPath, hints)) {
    return true;
  }

  return hasNonZeroRotationOrigin(origin);
}

function* iterateMeshGeometries(links: Record<string, UrdfLink>): Generator<UrdfVisual> {
  for (const link of Object.values(links)) {
    for (const entry of getVisualGeometryEntries(link)) {
      yield entry.geometry;
    }
    yield link.collision;

    for (const collisionBody of link.collisionBodies ?? []) {
      yield collisionBody;
    }
  }
}

export function buildColladaRootNormalizationHints(
  links?: Record<string, UrdfLink> | null,
): ColladaRootNormalizationHints | null {
  if (!links) {
    return null;
  }

  const packageNamesToNormalize = new Set<string>();
  const relativePathsToNormalize = new Set<string>();
  const daeMeshes: Array<{
    normalizedPath: string;
    packageName: string | null;
    packageRelativePath: string;
    hasRotationTransform: boolean;
  }> = [];

  for (const geometry of iterateMeshGeometries(links)) {
    if (geometry.type !== GeometryType.MESH || !geometry.meshPath?.toLowerCase().endsWith('.dae')) {
      continue;
    }

    const normalizedPath = normalizeMeshPath(geometry.meshPath);
    const packageName = extractPackageName(normalizedPath);
    const packageRelativePath = getPackageRelativePath(normalizedPath);
    const hasRotationTransform = shouldNormalizeColladaGeometry(geometry.meshPath, geometry.origin);

    daeMeshes.push({
      normalizedPath,
      packageName,
      packageRelativePath,
      hasRotationTransform,
    });

    if (hasRotationTransform) {
      if (packageName) {
        packageNamesToNormalize.add(packageName);
      } else {
        relativePathsToNormalize.add(packageRelativePath);
      }
    }
  }

  if (daeMeshes.length === 0) {
    return null;
  }

  const exactMeshPaths = new Set<string>();
  const packageRelativeMeshPaths = new Set<string>();

  for (const mesh of daeMeshes) {
    if (
      (mesh.packageName && packageNamesToNormalize.has(mesh.packageName)) ||
      relativePathsToNormalize.has(mesh.packageRelativePath)
    ) {
      exactMeshPaths.add(mesh.normalizedPath);
      packageRelativeMeshPaths.add(mesh.packageRelativePath);
    }
  }

  if (exactMeshPaths.size === 0 && packageRelativeMeshPaths.size === 0) {
    return null;
  }

  return {
    exactMeshPaths,
    packageRelativeMeshPaths,
  };
}

export function shouldNormalizeColladaRoot(
  path: string,
  hints?: ColladaRootNormalizationHints | null,
): boolean {
  if (!hints) {
    return false;
  }

  const candidates = new Set<string>([path, ...buildMeshLookupCandidates(path)]);

  for (const candidate of candidates) {
    const normalizedCandidate = normalizeMeshPath(candidate);
    if (hints.exactMeshPaths.has(normalizedCandidate)) {
      return true;
    }

    const packageRelativePath = getPackageRelativePath(normalizedCandidate);
    if (hints.packageRelativeMeshPaths.has(packageRelativePath)) {
      return true;
    }
  }

  return false;
}
