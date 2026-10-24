export interface ColladaUpAxisNormalizationResult {
  content: string;
  normalized: boolean;
}

const Z_UP_PATTERN = /<up_axis>\s*Z_UP\s*<\/up_axis>/i;

/**
 * URDF Studio uses a Z-up robot/world convention.
 * three.js ColladaLoader auto-converts Z_UP assets to Y-up by injecting a
 * root rotation, which makes imported URDF meshes appear 90° off against the
 * rest of the kinematic chain. Rewriting the metadata to Y_UP prevents that
 * extra runtime rotation while preserving authored mesh transforms.
 */
export function normalizeColladaUpAxis(content: string): ColladaUpAxisNormalizationResult {
  if (!Z_UP_PATTERN.test(content)) {
    return {
      content,
      normalized: false,
    };
  }

  return {
    content: content.replace(/<up_axis>\s*Z_UP\s*<\/up_axis>/gi, '<up_axis>Y_UP</up_axis>'),
    normalized: true,
  };
}
