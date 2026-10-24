import type { AssemblyState } from '@/types';

export interface AssemblyConnectivityGroup {
  componentIds: string[];
}

export interface AssemblyConnectivityAnalysis {
  componentCount: number;
  connectedGroupCount: number;
  connectedGroups: AssemblyConnectivityGroup[];
  isSingleConnectedComponent: boolean;
  hasDisconnectedComponents: boolean;
}

function sortGroup(group: Iterable<string>): string[] {
  return Array.from(group).sort((left, right) => left.localeCompare(right));
}

export function analyzeAssemblyConnectivity(
  assemblyState: AssemblyState | null | undefined,
): AssemblyConnectivityAnalysis {
  const componentIds = sortGroup(Object.keys(assemblyState?.components ?? {}));
  if (componentIds.length === 0) {
    return {
      componentCount: 0,
      connectedGroupCount: 0,
      connectedGroups: [],
      isSingleConnectedComponent: false,
      hasDisconnectedComponents: false,
    };
  }

  const adjacency = new Map<string, Set<string>>();
  componentIds.forEach((componentId) => {
    adjacency.set(componentId, new Set());
  });

  Object.values(assemblyState?.bridges ?? {}).forEach((bridge) => {
    if (!adjacency.has(bridge.parentComponentId) || !adjacency.has(bridge.childComponentId)) {
      return;
    }

    adjacency.get(bridge.parentComponentId)?.add(bridge.childComponentId);
    adjacency.get(bridge.childComponentId)?.add(bridge.parentComponentId);
  });

  const visited = new Set<string>();
  const connectedGroups: AssemblyConnectivityGroup[] = [];

  componentIds.forEach((startComponentId) => {
    if (visited.has(startComponentId)) {
      return;
    }

    const queue = [startComponentId];
    const group = new Set<string>();
    visited.add(startComponentId);

    while (queue.length > 0) {
      const currentComponentId = queue.shift();
      if (!currentComponentId) {
        continue;
      }

      group.add(currentComponentId);
      adjacency.get(currentComponentId)?.forEach((neighborId) => {
        if (visited.has(neighborId)) {
          return;
        }
        visited.add(neighborId);
        queue.push(neighborId);
      });
    }

    connectedGroups.push({
      componentIds: sortGroup(group),
    });
  });

  connectedGroups.sort((left, right) => (
    (left.componentIds[0] ?? '').localeCompare(right.componentIds[0] ?? '')
  ));

  const connectedGroupCount = connectedGroups.length;
  const isSingleConnectedComponent = componentIds.length === 1 || connectedGroupCount === 1;

  return {
    componentCount: componentIds.length,
    connectedGroupCount,
    connectedGroups,
    isSingleConnectedComponent,
    hasDisconnectedComponents: componentIds.length > 1 && !isSingleConnectedComponent,
  };
}
