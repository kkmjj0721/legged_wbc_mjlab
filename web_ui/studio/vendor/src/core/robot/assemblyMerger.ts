import type { AssemblyState, RobotData } from '@/types';

import { createAssemblySceneProjection } from './assemblySceneProjection';

/** Projects canonical source-local components and merges the visible assembly for consumers. */
export function mergeAssembly(assembly: AssemblyState): RobotData {
  return createAssemblySceneProjection(assembly).robotData;
}
