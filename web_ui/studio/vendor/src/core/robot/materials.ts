import { GeometryType, type RobotData, type UrdfLink } from '@/types';
import { getVisualGeometryEntries } from './visualBodies';
import {
  getEffectiveGeometryAuthoredMaterials,
  hasMultipleAuthoredMaterials,
} from './visualMaterials';

type RobotMaterials = RobotData['materials'];
type RobotMaterialEntry = NonNullable<RobotMaterials>[string];

function normalizeMaterialValue(value?: string | null): string | undefined {
  const trimmed = String(value || '').trim();
  return trimmed ? trimmed : undefined;
}

function materialValuesEqual(left?: string | null, right?: string | null): boolean {
  return (
    normalizeMaterialValue(left)?.toLowerCase() === normalizeMaterialValue(right)?.toLowerCase()
  );
}

function materialEntriesEqual(
  left: RobotMaterialEntry | undefined,
  right: RobotMaterialEntry | undefined,
): boolean {
  return (
    left?.color === right?.color &&
    JSON.stringify(left?.colorRgba ?? null) === JSON.stringify(right?.colorRgba ?? null) &&
    left?.texture === right?.texture &&
    JSON.stringify(left?.usdMaterial ?? null) === JSON.stringify(right?.usdMaterial ?? null)
  );
}

function resolveTrackedVisualMaterial(
  link: UrdfLink,
): Pick<RobotMaterialEntry, 'color' | 'colorRgba' | 'texture'> | null {
  if (link.visual.type === GeometryType.NONE) {
    return null;
  }

  const authoredMaterials = getEffectiveGeometryAuthoredMaterials(link.visual);
  if (authoredMaterials.length > 1) {
    return null;
  }

  const authoredMaterial = authoredMaterials[0];
  const color = normalizeMaterialValue(authoredMaterial?.color ?? link.visual.color);
  const colorRgba =
    Array.isArray(authoredMaterial?.colorRgba) &&
    authoredMaterial.colorRgba.length === 4 &&
    authoredMaterial.colorRgba.every((value) => Number.isFinite(value))
      ? ([
          Number(authoredMaterial.colorRgba[0]),
          Number(authoredMaterial.colorRgba[1]),
          Number(authoredMaterial.colorRgba[2]),
          Number(authoredMaterial.colorRgba[3]),
        ] as [number, number, number, number])
      : undefined;
  const texture = normalizeMaterialValue(authoredMaterial?.texture);

  if (!color && !colorRgba && !texture) {
    return null;
  }

  return {
    ...(color ? { color } : {}),
    ...(colorRgba ? { colorRgba } : {}),
    ...(texture ? { texture } : {}),
  };
}

function resolveExistingMaterialEntry(
  materials: RobotMaterials | undefined,
  nextLink: UrdfLink,
  previousLink?: UrdfLink,
): RobotMaterialEntry | undefined {
  return (
    materials?.[nextLink.id] ||
    materials?.[nextLink.name] ||
    (previousLink?.name ? materials?.[previousLink.name] : undefined)
  );
}

export function syncRobotMaterialsForLinkUpdate(
  materials: RobotMaterials | undefined,
  nextLink: UrdfLink,
  previousLink?: UrdfLink,
): RobotMaterials | undefined {
  const existingEntry = resolveExistingMaterialEntry(materials, nextLink, previousLink);
  const resolvedVisualMaterial = resolveTrackedVisualMaterial(nextLink);
  const shouldTrackEntry =
    getVisualGeometryEntries(nextLink).length > 0 ||
    Boolean(resolvedVisualMaterial?.color || resolvedVisualMaterial?.texture) ||
    Boolean(existingEntry?.color || existingEntry?.texture || existingEntry?.usdMaterial);

  if (!shouldTrackEntry) {
    return materials;
  }

  const nextEntry: RobotMaterialEntry = {
    ...(existingEntry || {}),
  };
  const nextColor = resolvedVisualMaterial?.color;
  const nextTexture = resolvedVisualMaterial?.texture;
  const preserveUsdMaterial =
    materialValuesEqual(existingEntry?.color, nextColor) &&
    materialValuesEqual(existingEntry?.texture, nextTexture);

  if (nextColor) {
    nextEntry.color = nextColor;
  } else {
    delete nextEntry.color;
  }

  if (resolvedVisualMaterial?.colorRgba) {
    nextEntry.colorRgba = resolvedVisualMaterial.colorRgba;
  } else {
    delete nextEntry.colorRgba;
  }

  if (nextTexture) {
    nextEntry.texture = nextTexture;
  } else {
    delete nextEntry.texture;
  }

  if (!preserveUsdMaterial) {
    delete nextEntry.usdMaterial;
  }

  if (!nextEntry.color && !nextEntry.colorRgba && !nextEntry.texture && !nextEntry.usdMaterial) {
    if (!materials || !Object.prototype.hasOwnProperty.call(materials, nextLink.id)) {
      return materials;
    }

    const nextMaterials = { ...materials };
    delete nextMaterials[nextLink.id];
    return Object.keys(nextMaterials).length > 0 ? nextMaterials : undefined;
  }

  const currentIdEntry = materials?.[nextLink.id];
  if (materialEntriesEqual(currentIdEntry, nextEntry)) {
    return materials;
  }

  return {
    ...(materials || {}),
    [nextLink.id]: nextEntry,
  };
}

export function syncRobotVisualColorsFromMaterials<
  T extends Pick<RobotData, 'links' | 'materials'>,
>(robot: T): T {
  if (!robot.materials || Object.keys(robot.materials).length === 0) {
    return robot;
  }

  let linksChanged = false;
  const nextLinks = Object.fromEntries(
    Object.entries(robot.links).map(([linkId, link]) => {
      const materialColor = normalizeMaterialValue(
        robot.materials?.[link.id]?.color || robot.materials?.[link.name]?.color,
      );
      if (link.visual.type !== GeometryType.NONE && hasMultipleAuthoredMaterials(link.visual)) {
        if (link.visual.color === undefined) {
          return [linkId, link];
        }

        linksChanged = true;
        return [
          linkId,
          {
            ...link,
            visual: {
              ...link.visual,
              color: undefined,
            },
          },
        ];
      }

      if (
        !materialColor ||
        link.visual.type === GeometryType.NONE ||
        materialValuesEqual(link.visual.color, materialColor)
      ) {
        return [linkId, link];
      }

      linksChanged = true;
      return [
        linkId,
        {
          ...link,
          visual: {
            ...link.visual,
            color: materialColor,
          },
        },
      ];
    }),
  ) as T['links'];

  if (!linksChanged) {
    return robot;
  }

  return {
    ...robot,
    links: nextLinks,
  };
}
