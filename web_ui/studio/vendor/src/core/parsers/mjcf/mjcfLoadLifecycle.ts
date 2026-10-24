import * as THREE from 'three';

export interface MJCFLoadAbortSignal {
  aborted: boolean;
}

export class MJCFLoadAbortedError extends Error {
  constructor(message = 'MJCF load aborted') {
    super(message);
    this.name = 'MJCFLoadAbortedError';
  }
}

export function isMJCFLoadAbortedError(error: unknown): error is MJCFLoadAbortedError {
  return (
    error instanceof MJCFLoadAbortedError ||
    (error instanceof Error && error.name === 'MJCFLoadAbortedError')
  );
}

export function throwIfMJCFLoadAborted(signal?: MJCFLoadAbortSignal): void {
  if (signal?.aborted) {
    throw new MJCFLoadAbortedError();
  }
}

function disposeMaterialTextures(
  material: THREE.Material,
  disposedTextures: Set<THREE.Texture>,
): void {
  for (const value of Object.values(material as unknown as Record<string, unknown>)) {
    if (!(value instanceof THREE.Texture) || disposedTextures.has(value)) {
      continue;
    }

    disposedTextures.add(value);
    value.dispose?.();
  }
}

export function disposeTransientObject3D(root: THREE.Object3D | null | undefined): void {
  if (!root) {
    return;
  }

  if (root.parent) {
    root.parent.remove(root);
  }

  const disposedMaterials = new Set<THREE.Material>();
  const disposedSkeletons = new Set<THREE.Skeleton>();
  const disposedTextures = new Set<THREE.Texture>();

  root.traverse((child) => {
    const skinnedMesh = child as THREE.SkinnedMesh;
    if (
      skinnedMesh.isSkinnedMesh &&
      skinnedMesh.skeleton &&
      !disposedSkeletons.has(skinnedMesh.skeleton)
    ) {
      disposedSkeletons.add(skinnedMesh.skeleton);
      skinnedMesh.skeleton.dispose?.();
    }

    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) {
      return;
    }

    mesh.geometry?.dispose?.();

    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials) {
      if (!material || disposedMaterials.has(material)) {
        continue;
      }

      disposeMaterialTextures(material, disposedTextures);
      material.dispose?.();
      disposedMaterials.add(material);
    }
  });

  root.clear();
}

export function disposeTemporaryTexturePromiseCache(
  textureLoadCache: Map<string, Promise<THREE.Texture | null>>,
): void {
  for (const texturePromise of textureLoadCache.values()) {
    void texturePromise
      .then((texture) => {
        texture?.dispose?.();
      })
      .catch(() => undefined);
  }

  textureLoadCache.clear();
}
