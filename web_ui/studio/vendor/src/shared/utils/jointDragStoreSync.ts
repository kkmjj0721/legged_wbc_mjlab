import { throttle } from './throttle.ts';

export type JointDragSyncMode = 'immediate' | 'throttle' | 'animationFrame';

interface JointDragStoreSyncOptions {
  onDragChange?: (jointName: string, angle: number) => void;
  onDragCommit?: (jointName: string, angle: number) => void;
  throttleChanges?: boolean;
  intervalMs?: number;
  syncMode?: JointDragSyncMode;
}

export interface JointDragStoreSync {
  emit: (jointName: string, angle: number) => void;
  commit: (jointName: string, angle: number) => void;
  dispose: () => void;
}

const DEFAULT_INTERVAL_MS = 33;

export function createJointDragStoreSync({
  onDragChange,
  onDragCommit,
  throttleChanges = false,
  intervalMs = DEFAULT_INTERVAL_MS,
  syncMode,
}: JointDragStoreSyncOptions): JointDragStoreSync {
  const emitImmediate = (jointName: string, angle: number) => {
    onDragChange?.(jointName, angle);
  };

  const emitThrottled = throttle(emitImmediate, intervalMs);
  const resolvedSyncMode = syncMode ?? (throttleChanges ? 'throttle' : 'immediate');
  let pendingAnimationFrame: number | null = null;
  let pendingAnimationFrameArgs: [string, number] | null = null;

  const cancelAnimationFrameEmit = () => {
    if (
      pendingAnimationFrame !== null &&
      typeof globalThis.cancelAnimationFrame === 'function'
    ) {
      globalThis.cancelAnimationFrame(pendingAnimationFrame);
    }
    pendingAnimationFrame = null;
    pendingAnimationFrameArgs = null;
  };

  const emitAnimationFrame = (jointName: string, angle: number) => {
    pendingAnimationFrameArgs = [jointName, angle];

    if (pendingAnimationFrame !== null) {
      return;
    }

    if (typeof globalThis.requestAnimationFrame !== 'function') {
      const args = pendingAnimationFrameArgs;
      pendingAnimationFrameArgs = null;
      if (args) {
        emitImmediate(...args);
      }
      return;
    }

    pendingAnimationFrame = globalThis.requestAnimationFrame(() => {
      pendingAnimationFrame = null;
      const args = pendingAnimationFrameArgs;
      pendingAnimationFrameArgs = null;
      if (args) {
        emitImmediate(...args);
      }
    });
  };

  const emit =
    resolvedSyncMode === 'animationFrame'
      ? emitAnimationFrame
      : resolvedSyncMode === 'throttle'
        ? emitThrottled
        : emitImmediate;

  return {
    emit(jointName, angle) {
      emit(jointName, angle);
    },
    commit(jointName, angle) {
      emitThrottled.cancel();
      cancelAnimationFrameEmit();
      onDragCommit?.(jointName, angle);
    },
    dispose() {
      emitThrottled.cancel();
      cancelAnimationFrameEmit();
    },
  };
}
