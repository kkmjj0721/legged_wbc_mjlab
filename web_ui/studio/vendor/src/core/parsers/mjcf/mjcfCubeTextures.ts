import { BOX_FACE_MATERIAL_ORDER, type BoxFaceMaterialName } from '@/core/robot';
import type { UrdfVisualMaterial } from '@/types';

import type { MJCFTexture } from './mjcfUtils';

const MJCF_CUBE_TEXTURE_FACE_ATTRIBUTE_BY_FACE = {
  right: 'fileright',
  left: 'fileleft',
  up: 'fileup',
  down: 'filedown',
  front: 'filefront',
  back: 'fileback',
} as const satisfies Record<BoxFaceMaterialName, keyof MJCFTexture>;

export const MJCF_CUBE_TEXTURE_FACE_ATTRIBUTE_ORDER = BOX_FACE_MATERIAL_ORDER.map(
  (face) => MJCF_CUBE_TEXTURE_FACE_ATTRIBUTE_BY_FACE[face],
);

export type MjcfCubeTextureFaceRecord = Record<BoxFaceMaterialName, string>;

function normalizeTexturePath(value: string | null | undefined): string | undefined {
  const normalizedValue = String(value || '').trim();
  return normalizedValue ? normalizedValue : undefined;
}

export function getMjcfCubeTextureFaceRecord(
  texture:
    | Pick<
        MJCFTexture,
        'type' | 'fileback' | 'filedown' | 'filefront' | 'fileleft' | 'fileright' | 'fileup'
      >
    | null
    | undefined,
): MjcfCubeTextureFaceRecord | null {
  if (
    String(texture?.type || '')
      .trim()
      .toLowerCase() !== 'cube'
  ) {
    return null;
  }

  const faceRecord = {} as MjcfCubeTextureFaceRecord;
  for (const face of BOX_FACE_MATERIAL_ORDER) {
    const attribute = MJCF_CUBE_TEXTURE_FACE_ATTRIBUTE_BY_FACE[face];
    const texturePath = normalizeTexturePath(texture?.[attribute]);
    if (!texturePath) {
      return null;
    }
    faceRecord[face] = texturePath;
  }

  return faceRecord;
}

export function getMjcfCubeTextureFacePaths(
  texture:
    | Pick<
        MJCFTexture,
        'type' | 'fileback' | 'filedown' | 'filefront' | 'fileleft' | 'fileright' | 'fileup'
      >
    | null
    | undefined,
): string[] {
  const faceRecord = getMjcfCubeTextureFaceRecord(texture);
  if (!faceRecord) {
    return [];
  }

  return BOX_FACE_MATERIAL_ORDER.map((face) => faceRecord[face]);
}

export function buildMjcfCubeAuthoredMaterials(
  faceRecord: MjcfCubeTextureFaceRecord,
  sharedColor?: string,
  sharedColorRgba?: [number, number, number, number],
): UrdfVisualMaterial[] {
  return BOX_FACE_MATERIAL_ORDER.map((face) => ({
    ...(sharedColor ? { color: sharedColor } : {}),
    ...(sharedColorRgba ? { colorRgba: sharedColorRgba } : {}),
    texture: faceRecord[face],
  }));
}
