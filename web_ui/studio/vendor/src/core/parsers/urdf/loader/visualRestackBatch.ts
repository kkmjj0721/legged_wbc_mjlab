export interface VisualRestackBatch {
  trackLoad: () => (needsRestack: boolean) => void;
  markHierarchyReady: () => void;
  resetAfterImmediateRestack: () => void;
  flush: () => void;
}

export function createVisualRestackBatch(restack: () => void): VisualRestackBatch {
  let pendingLoads = 0;
  let needsRestack = false;
  let hierarchyReady = false;

  const flush = () => {
    if (!needsRestack || !hierarchyReady || pendingLoads > 0) {
      return;
    }

    needsRestack = false;
    restack();
  };

  return {
    trackLoad: () => {
      let completed = false;
      pendingLoads += 1;

      return (shouldRestack: boolean) => {
        if (completed) {
          return;
        }

        completed = true;
        pendingLoads = Math.max(0, pendingLoads - 1);
        needsRestack = needsRestack || shouldRestack;
        flush();
      };
    },
    markHierarchyReady: () => {
      hierarchyReady = true;
      flush();
    },
    resetAfterImmediateRestack: () => {
      needsRestack = false;
    },
    flush,
  };
}
