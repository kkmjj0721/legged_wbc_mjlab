import type {
  AssemblyComponent,
  AssemblyState,
  ComponentSourceDraft,
  ComponentSourceFormat,
  RobotData,
} from '@/types';
import { createRobotPersistenceSnapshot } from './semanticSnapshot';

const SOURCE_FORMATS = new Set<ComponentSourceFormat>([
  'urdf',
  'mjcf',
  'usd',
  'xacro',
  'sdf',
]);

function hashSemanticSnapshot(snapshot: string): string {
  let first = 0x811c9dc5;
  let second = 0x9e3779b9;
  for (let index = 0; index < snapshot.length; index += 1) {
    const code = snapshot.charCodeAt(index);
    first = Math.imul(first ^ code, 0x01000193);
    second = Math.imul(second ^ code, 0x85ebca6b);
  }
  first ^= first >>> 16;
  first = Math.imul(first, 0x7feb352d);
  first ^= first >>> 15;
  second ^= second >>> 16;
  second = Math.imul(second, 0x846ca68b);
  second ^= second >>> 16;
  return `robot-semantic-v1:${snapshot.length.toString(36)}:${(first >>> 0)
    .toString(16)
    .padStart(8, '0')}${(second >>> 0).toString(16).padStart(8, '0')}`;
}

export function createSourceSemanticRobotHash(robot: RobotData): string {
  return hashSemanticSnapshot(createRobotPersistenceSnapshot(robot));
}

export function isComponentSourceFormat(value: unknown): value is ComponentSourceFormat {
  return typeof value === 'string' && SOURCE_FORMATS.has(value as ComponentSourceFormat);
}

export function createComponentSourceDraft({
  componentId,
  format,
  content,
  robot,
}: {
  componentId: string;
  format: ComponentSourceFormat;
  content: string;
  robot: RobotData;
}): ComponentSourceDraft {
  if (!componentId.trim()) throw new Error('Component source draft requires a componentId.');
  if (!isComponentSourceFormat(format)) {
    throw new Error(`Unsupported source draft format: ${format}`);
  }
  return {
    componentId,
    format,
    content,
    robotSnapshotHash: createSourceSemanticRobotHash(robot),
  };
}

export function isComponentSourceDraftMatchingRobot(
  draft: ComponentSourceDraft,
  robot: RobotData,
): boolean {
  return draft.robotSnapshotHash === createSourceSemanticRobotHash(robot);
}

export function isComponentSourceDraftMatchingComponent(
  draft: ComponentSourceDraft,
  component: AssemblyComponent,
): boolean {
  return (
    draft.componentId === component.id
    && isComponentSourceDraftMatchingRobot(draft, component.robot)
  );
}

export type SourcePreservingDraftResolution =
  | { status: 'matched'; draft: ComponentSourceDraft }
  | { status: 'regenerate'; reason: 'component-missing' | 'draft-missing' | 'draft-stale' };

/** Never consults the immutable library template. */
export function resolveSourcePreservingComponentDraft({
  workspace,
  componentId,
  drafts,
}: {
  workspace: AssemblyState;
  componentId: string;
  drafts: Record<string, ComponentSourceDraft>;
}): SourcePreservingDraftResolution {
  const component = workspace.components[componentId];
  if (!component) return { status: 'regenerate', reason: 'component-missing' };
  const draft = drafts[componentId];
  if (!draft) return { status: 'regenerate', reason: 'draft-missing' };
  if (!isComponentSourceDraftMatchingComponent(draft, component)) {
    return { status: 'regenerate', reason: 'draft-stale' };
  }
  return { status: 'matched', draft };
}

export function requireSourcePreservingComponentDraft(
  params: Parameters<typeof resolveSourcePreservingComponentDraft>[0],
): ComponentSourceDraft {
  const resolution = resolveSourcePreservingComponentDraft(params);
  if (resolution.status === 'matched') return resolution.draft;
  throw new Error(
    `Cannot preserve component source for "${params.componentId}": ${resolution.reason}; regenerate the source first.`,
  );
}
