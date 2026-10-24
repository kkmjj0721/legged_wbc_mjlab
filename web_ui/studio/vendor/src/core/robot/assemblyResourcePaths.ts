import { resolveImportedAssetPath } from '@/core/parsers/meshPathUtils';
import type {
  AssemblyState,
  RobotData,
  RobotMaterialState,
  UrdfVisual,
  UrdfVisualMaterial,
} from '@/types';

export type AssemblySceneRenderStrategy = 'direct-component' | 'assembled-scene';

const WORKSPACE_COMPONENT_RESOURCE_ROOT = '__workspace__/components';
const EXTERNAL_RESOURCE_PATTERN = /^(?:blob:|https?:\/\/|data:)/i;

export function resolveAssemblySceneRenderStrategy(
  assembly: AssemblyState,
): AssemblySceneRenderStrategy {
  const visibleComponentCount = Object.values(assembly.components).filter(
    (component) => component.visible !== false,
  ).length;
  return visibleComponentCount === 1 && Object.keys(assembly.bridges).length === 0
    ? 'direct-component'
    : 'assembled-scene';
}

function getComponentResourceRoot(componentId: string): string {
  return `${WORKSPACE_COMPONENT_RESOURCE_ROOT}/${encodeURIComponent(componentId)}`;
}

/**
 * Resolve a renderer/export resource without changing the canonical component.
 * Assembled scenes use a component-owned virtual root so equal source-local
 * basenames can never resolve through the loader's ambiguous basename index.
 */
export function resolveAssemblyComponentResourcePath({
  componentId,
  sourceFile,
  resourcePath,
  renderStrategy,
}: {
  componentId: string;
  sourceFile: string | null;
  resourcePath: string;
  renderStrategy: AssemblySceneRenderStrategy;
}): string {
  const rawPath = resourcePath.trim();
  if (!rawPath || renderStrategy === 'direct-component' || EXTERNAL_RESOURCE_PATTERN.test(rawPath)) {
    return resourcePath;
  }
  const sourceResolvedPath = resolveImportedAssetPath(rawPath, sourceFile) || rawPath;
  return `${getComponentResourceRoot(componentId)}/${sourceResolvedPath.replace(/^\/+/, '')}`;
}

function projectMaterial(
  material: UrdfVisualMaterial,
  projectPath: (path: string) => string,
): UrdfVisualMaterial {
  return {
    ...material,
    ...(material.texture ? { texture: projectPath(material.texture) } : {}),
    ...(material.passes
      ? {
          passes: material.passes.map((pass) => ({
            ...pass,
            ...(pass.texture ? { texture: projectPath(pass.texture) } : {}),
          })),
        }
      : {}),
  };
}

function projectGeometry(
  geometry: UrdfVisual,
  projectPath: (path: string) => string,
): UrdfVisual {
  return {
    ...geometry,
    ...(geometry.meshPath ? { meshPath: projectPath(geometry.meshPath) } : {}),
    ...(geometry.authoredMaterials
      ? {
          authoredMaterials: geometry.authoredMaterials.map((material) =>
            projectMaterial(material, projectPath)
          ),
        }
      : {}),
    ...(geometry.mjcfMesh?.file
      ? { mjcfMesh: { ...geometry.mjcfMesh, file: projectPath(geometry.mjcfMesh.file) } }
      : {}),
    ...(geometry.mjcfHfield?.file
      ? { mjcfHfield: { ...geometry.mjcfHfield, file: projectPath(geometry.mjcfHfield.file) } }
      : {}),
    ...(geometry.sdfHeightmap
      ? {
          sdfHeightmap: {
            ...geometry.sdfHeightmap,
            uri: projectPath(geometry.sdfHeightmap.uri),
            textures: geometry.sdfHeightmap.textures.map((texture) => ({
              ...texture,
              ...(texture.diffuse ? { diffuse: projectPath(texture.diffuse) } : {}),
              ...(texture.normal ? { normal: projectPath(texture.normal) } : {}),
            })),
          },
        }
      : {}),
  };
}

function projectRobotMaterial(
  material: RobotMaterialState,
  projectPath: (path: string) => string,
): RobotMaterialState {
  const usdMaterial = material.usdMaterial
    ? Object.fromEntries(
        Object.entries(material.usdMaterial).map(([key, value]) => [
          key,
          key.endsWith('MapPath') && typeof value === 'string' && value
            ? projectPath(value)
            : value,
        ]),
      )
    : material.usdMaterial;
  return {
    ...material,
    ...(material.texture ? { texture: projectPath(material.texture) } : {}),
    ...(usdMaterial ? { usdMaterial } : {}),
  };
}

function collectGeometryResourcePaths(geometry: UrdfVisual, target: Set<string>): void {
  if (geometry.meshPath) target.add(geometry.meshPath);
  geometry.authoredMaterials?.forEach((material) => {
    if (material.texture) target.add(material.texture);
    material.passes?.forEach((pass) => {
      if (pass.texture) target.add(pass.texture);
    });
  });
  if (geometry.mjcfMesh?.file) target.add(geometry.mjcfMesh.file);
  if (geometry.mjcfHfield?.file) target.add(geometry.mjcfHfield.file);
  if (geometry.sdfHeightmap) {
    target.add(geometry.sdfHeightmap.uri);
    geometry.sdfHeightmap.textures.forEach((texture) => {
      if (texture.diffuse) target.add(texture.diffuse);
      if (texture.normal) target.add(texture.normal);
    });
  }
}

function collectRobotResourcePaths(robot: RobotData): Set<string> {
  const paths = new Set<string>();
  Object.values(robot.links).forEach((link) => {
    [
      link.visual,
      ...(link.visualBodies ?? []),
      link.collision,
      ...(link.collisionBodies ?? []),
    ].forEach((geometry) => collectGeometryResourcePaths(geometry, paths));
  });
  Object.values(robot.materials ?? {}).forEach((material) => {
    if (material.texture) paths.add(material.texture);
    Object.entries(material.usdMaterial ?? {}).forEach(([key, value]) => {
      if (key.endsWith('MapPath') && typeof value === 'string' && value) {
        paths.add(value);
      }
    });
  });
  return paths;
}

/** Project only resource references; source-local entity identities are untouched. */
export function projectAssemblyComponentRobotResources({
  componentId,
  sourceFile,
  robot,
  renderStrategy,
}: {
  componentId: string;
  sourceFile: string | null;
  robot: RobotData;
  renderStrategy: AssemblySceneRenderStrategy;
}): RobotData {
  if (renderStrategy === 'direct-component') {
    return robot;
  }
  const projectPath = (resourcePath: string) =>
    resolveAssemblyComponentResourcePath({
      componentId,
      sourceFile,
      resourcePath,
      renderStrategy,
    });

  return {
    ...robot,
    links: Object.fromEntries(
      Object.entries(robot.links).map(([linkId, link]) => [
        linkId,
        {
          ...link,
          visual: projectGeometry(link.visual, projectPath),
          ...(link.visualBodies
            ? { visualBodies: link.visualBodies.map((body) => projectGeometry(body, projectPath)) }
            : {}),
          collision: projectGeometry(link.collision, projectPath),
          ...(link.collisionBodies
            ? {
                collisionBodies: link.collisionBodies.map((body) =>
                  projectGeometry(body, projectPath)
                ),
              }
            : {}),
        },
      ]),
    ),
    ...(robot.materials
      ? {
          materials: Object.fromEntries(
            Object.entries(robot.materials).map(([materialId, material]) => [
              materialId,
              projectRobotMaterial(material, projectPath),
            ]),
          ),
        }
      : {}),
  };
}

/** Build virtual-path aliases shared by viewer and every canonical export path. */
export function buildAssemblyProjectedAssetAliases({
  assembly,
  assets,
}: {
  assembly: AssemblyState;
  assets: Record<string, string>;
}): Record<string, string> {
  const renderStrategy = resolveAssemblySceneRenderStrategy(assembly);
  if (renderStrategy === 'direct-component') return {};
  const assetPaths = Object.keys(assets);
  const aliases: Record<string, string> = {};

  Object.values(assembly.components).forEach((component) => {
    if (component.visible === false) return;
    collectRobotResourcePaths(component.robot).forEach((sourcePath) => {
      const projectedPath = resolveAssemblyComponentResourcePath({
        componentId: component.id,
        sourceFile: component.sourceFile,
        resourcePath: sourcePath,
        renderStrategy,
      });
      const resolvedSourcePath = resolveImportedAssetPath(sourcePath, component.sourceFile, {
        candidateAssetPaths: assetPaths,
      });
      const assetUrl = assets[resolvedSourcePath] ?? assets[sourcePath];
      if (assetUrl) {
        aliases[projectedPath] = assetUrl;
      }
    });
  });
  return aliases;
}
