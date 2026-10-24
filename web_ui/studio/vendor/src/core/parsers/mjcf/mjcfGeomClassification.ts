export interface MJCFGeomClassificationInput {
    name?: string;
    className?: string;
    classQName?: string;
    group?: number;
    contype?: number;
    conaffinity?: number;
}

export interface MJCFGeomClassification {
    isVisual: boolean;
    isCollision: boolean;
}

export interface MJCFBodyGeomRole<TGeom extends MJCFGeomClassificationInput> {
    geom: TGeom;
    classification: MJCFGeomClassification;
    renderVisual: boolean;
    renderCollision: boolean;
}

function tokenizeHint(value: string | undefined): string[] {
    if (!value) {
        return [];
    }

    return value
        .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
        .replace(/[^a-zA-Z0-9]+/g, ' ')
        .toLowerCase()
        .split(/\s+/)
        .filter(Boolean);
}

function buildHintTokenSet(geom: MJCFGeomClassificationInput): Set<string> {
    return new Set([
        ...tokenizeHint(geom.name),
        ...tokenizeHint(geom.className),
        ...tokenizeHint(geom.classQName),
    ]);
}

export function classifyMJCFGeom(geom: MJCFGeomClassificationInput): MJCFGeomClassification {
    const hintTokens = buildHintTokenSet(geom);
    const hasCollisionHint = hintTokens.has('collision') || hintTokens.has('collider') || hintTokens.has('col');
    const hasVisualHint = hintTokens.has('visual') || hintTokens.has('render') || hintTokens.has('virtual');
    const hasNonRenderableHelperHint = hintTokens.has('inertial') || hintTokens.has('fluid');
    const hasVisualGroup = geom.group === 1 || geom.group === 2;
    const hasDedicatedCollisionGroup = typeof geom.group === 'number' && geom.group >= 3;
    const hasContactsDisabled = geom.contype === 0 && geom.conaffinity === 0;

    if (hasCollisionHint) {
        return { isVisual: false, isCollision: true };
    }

    if (hasNonRenderableHelperHint) {
        return { isVisual: false, isCollision: false };
    }

    if (hasVisualHint || hasVisualGroup) {
        return { isVisual: true, isCollision: false };
    }

    if (hasDedicatedCollisionGroup && !hasContactsDisabled) {
        return { isVisual: false, isCollision: true };
    }

    if (hasDedicatedCollisionGroup && hasContactsDisabled) {
        return { isVisual: false, isCollision: false };
    }

    if (hasContactsDisabled) {
        return { isVisual: true, isCollision: false };
    }

    // Plain MJCF geoms without classification hints commonly serve both purposes.
    return { isVisual: true, isCollision: true };
}

export function assignMJCFBodyGeomRoles<TGeom extends MJCFGeomClassificationInput>(
    geoms: TGeom[],
): MJCFBodyGeomRole<TGeom>[] {
    const classifications = geoms.map((geom) => ({
        geom,
        classification: classifyMJCFGeom(geom),
    }));

    const hasDedicatedVisualGeom = classifications.some(
        ({ classification }) => classification.isVisual && !classification.isCollision,
    );

    return classifications.map(({ geom, classification }) => ({
        geom,
        classification,
        renderVisual: classification.isVisual
            && (!hasDedicatedVisualGeom || !classification.isCollision),
        renderCollision: classification.isCollision,
    }));
}
