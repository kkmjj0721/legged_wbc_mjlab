const MISSING_ATTRIBUTE_WHITESPACE_PATTERN = /"(?=[A-Za-z_:][-A-Za-z0-9_:.]*(?:\s*=))/g;

export interface ParsedMJCFXmlDocument {
  doc: Document | null;
  normalizedContent: string;
  parseError: string | null;
}

const SOURCE_ONLY_MJCF_FRAGMENT_ROOTS = new Set([
  'asset',
  'body',
  'compiler',
  'contact',
  'custom',
  'default',
  'deformable',
  'equality',
  'extension',
  'geom',
  'keyframe',
  'option',
  'sensor',
  'site',
  'size',
  'statistic',
  'tendon',
  'visual',
  'worldbody',
]);

const MJCF_FRAGMENT_CONTAINER_CHILDREN = new Set([
  'actuator',
  'compiler',
  'contact',
  'custom',
  'default',
  'deformable',
  'equality',
  'extension',
  'keyframe',
  'option',
  'sensor',
  'size',
  'statistic',
  'tendon',
  'visual',
  'worldbody',
]);

export function normalizeMJCFXmlForDomParsing(content: string): string {
  let normalized = content;

  while (true) {
    const next = normalized.replace(MISSING_ATTRIBUTE_WHITESPACE_PATTERN, '" ');
    if (next === normalized) {
      return normalized;
    }
    normalized = next;
  }
}

export function parseMJCFXmlDocument(content: string): ParsedMJCFXmlDocument {
  const normalizedContent = normalizeMJCFXmlForDomParsing(content);

  try {
    const parser = new DOMParser();
    const doc = parser.parseFromString(normalizedContent, 'text/xml');
    const parseError = doc.querySelector('parsererror');
    if (parseError) {
      return {
        doc: null,
        normalizedContent,
        parseError: parseError.textContent?.trim() || 'Unknown XML parsing error.',
      };
    }

    return {
      doc,
      normalizedContent,
      parseError: null,
    };
  } catch (error) {
    return {
      doc: null,
      normalizedContent,
      parseError:
        error instanceof Error ? error.message : String(error ?? 'Unknown XML parsing error.'),
    };
  }
}

export function isSourceOnlyMJCFDocument(content: string): boolean {
  const { doc } = parseMJCFXmlDocument(content);
  if (!doc) {
    return false;
  }

  const rootTagName = doc.documentElement?.tagName?.toLowerCase() ?? '';
  if (rootTagName === 'mujocoinclude') {
    return true;
  }

  if (SOURCE_ONLY_MJCF_FRAGMENT_ROOTS.has(rootTagName)) {
    return true;
  }

  if (rootTagName !== 'mujoco') {
    return Array.from(doc.documentElement?.children ?? []).some((child) =>
      MJCF_FRAGMENT_CONTAINER_CHILDREN.has(child.tagName.toLowerCase()),
    );
  }

  return !doc.querySelector('worldbody');
}

export function isStandaloneMJCFDocument(content: string): boolean {
  const { doc } = parseMJCFXmlDocument(content);
  if (!doc) {
    return false;
  }

  const rootTagName = doc.documentElement?.tagName?.toLowerCase() ?? '';
  return rootTagName === 'mujoco' && Boolean(doc.querySelector('worldbody'));
}

export function looksLikeMJCFSourceDocument(content: string): boolean {
  return isStandaloneMJCFDocument(content) || isSourceOnlyMJCFDocument(content);
}
