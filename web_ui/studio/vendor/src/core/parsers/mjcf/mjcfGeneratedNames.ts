const MJCF_GENERATED_BODY_SEGMENT = /(?:^|\/|_)body(?:_(\d+)|\[(\d+)\])$/i;
const MJCF_GENERATED_OBJECT_LEGACY_SEGMENT = /^(.*)::(geom|site)(?:_(\d+)|\[(\d+)\])$/i;
const MJCF_GENERATED_OBJECT_UNDERSCORE_SEGMENT = /^(.*)_(geom|site)_(\d+)$/i;

function parseGeneratedIndex(
  primaryIndex: string | undefined,
  legacyIndex: string | undefined,
): number | null {
  const parsed = Number.parseInt(primaryIndex || legacyIndex || '', 10);
  return Number.isFinite(parsed) ? parsed : null;
}

export function buildGeneratedMjcfBodySegment(siblingIndex: number): string {
  return `body_${siblingIndex}`;
}

function normalizeGeneratedMjcfOwnerName(parentPath: string): string {
  return String(parentPath)
    .trim()
    .replace(/[\\/]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '');
}

export function buildGeneratedMjcfBodyPath(parentPath: string, siblingIndex: number): string {
  const normalizedParent = normalizeGeneratedMjcfOwnerName(parentPath);
  return normalizedParent
    ? `${normalizedParent}_${buildGeneratedMjcfBodySegment(siblingIndex)}`
    : buildGeneratedMjcfBodySegment(siblingIndex);
}

function buildGeneratedMjcfObjectName(
  bodyPath: string,
  objectKind: 'geom' | 'site',
  objectIndex: number,
): string {
  return `${bodyPath}_${objectKind}_${objectIndex}`;
}

export function buildGeneratedMjcfGeomName(bodyPath: string, geomIndex: number): string {
  return buildGeneratedMjcfObjectName(bodyPath, 'geom', geomIndex);
}

export function buildGeneratedMjcfSiteName(bodyPath: string, siteIndex: number): string {
  return buildGeneratedMjcfObjectName(bodyPath, 'site', siteIndex);
}

export function buildGeneratedMjcfJointName(bodyPath: string, jointIndex: number): string {
  return `${bodyPath}_joint_${jointIndex}`;
}

export function parseGeneratedMjcfBodyIndex(name: string | null | undefined): number | null {
  const match = String(name || '')
    .trim()
    .match(MJCF_GENERATED_BODY_SEGMENT);
  if (!match) {
    return null;
  }

  return parseGeneratedIndex(match[1], match[2]);
}

export function parseGeneratedMjcfObjectName(name: string | null | undefined): {
  ownerName: string;
  objectKind: 'geom' | 'site';
  index: number;
} | null {
  const normalizedName = String(name || '').trim();
  if (!normalizedName) {
    return null;
  }

  const legacyMatch = normalizedName.match(MJCF_GENERATED_OBJECT_LEGACY_SEGMENT);
  if (legacyMatch) {
    const index = parseGeneratedIndex(legacyMatch[3], legacyMatch[4]);
    if (index == null) {
      return null;
    }

    const ownerName = legacyMatch[1] || '';
    const objectKind = (legacyMatch[2] || '').toLowerCase() as 'geom' | 'site';
    if (!ownerName || (objectKind !== 'geom' && objectKind !== 'site')) {
      return null;
    }

    return {
      ownerName,
      objectKind,
      index,
    };
  }

  const underscoreMatch = normalizedName.match(MJCF_GENERATED_OBJECT_UNDERSCORE_SEGMENT);
  if (!underscoreMatch) {
    return null;
  }

  const ownerName = underscoreMatch[1] || '';
  const objectKind = (underscoreMatch[2] || '').toLowerCase() as 'geom' | 'site';
  const index = Number.parseInt(underscoreMatch[3] || '', 10);
  if (!ownerName || (objectKind !== 'geom' && objectKind !== 'site')) {
    return null;
  }
  if (!Number.isFinite(index)) {
    return null;
  }

  return {
    ownerName,
    objectKind,
    index,
  };
}
