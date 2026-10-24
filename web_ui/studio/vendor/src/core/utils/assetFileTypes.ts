const IMAGE_ASSET_EXTENSIONS = new Set([
  'png',
  'jpg',
  'jpeg',
  'bmp',
  'tga',
  'tiff',
  'tif',
  'webp',
  'hdr',
]);

export function getAssetFileExtension(path: string): string {
  const normalizedPath = String(path || '')
    .trim()
    .replace(/\\/g, '/')
    .split('?')[0]
    .split('#')[0];
  const fileName = normalizedPath.split('/').pop() ?? normalizedPath;
  const lastDotIndex = fileName.lastIndexOf('.');
  return lastDotIndex >= 0 ? fileName.slice(lastDotIndex + 1).toLowerCase() : '';
}

export function isImageAssetPath(path: string): boolean {
  return IMAGE_ASSET_EXTENSIONS.has(getAssetFileExtension(path));
}
