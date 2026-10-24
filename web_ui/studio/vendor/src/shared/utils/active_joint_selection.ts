import { resolveViewerJointKey } from '@/shared/utils/jointPanelState';
import { isSingleDofJoint } from '@/shared/utils/jointTypes';

type ViewerSelection = {
  type: 'link' | 'joint' | 'tendon' | null;
  id: string | null;
};

type ViewerJointLike = {
  childLinkId?: string;
  childName?: string;
  name?: string;
  parentLink?: {
    name?: string;
  };
  parentLinkId?: string;
  parentName?: string;
  type?: string;
  jointType?: string;
  child?: {
    name?: string;
  };
  parent?: {
    name?: string;
  };
};

function resolveViewerJointChildLinkIdentity(
  joint: ViewerJointLike | null | undefined,
): string | null {
  return joint?.child?.name ?? joint?.childLinkId ?? joint?.childName ?? null;
}

function resolveViewerJointParentLinkIdentity(
  joint: ViewerJointLike | null | undefined,
): string | null {
  return (
    joint?.parentLink?.name ??
    joint?.parent?.name ??
    joint?.parentLinkId ??
    joint?.parentName ??
    null
  );
}

function resolveNearestControllableParentJointKey(
  joints: Record<string, ViewerJointLike>,
  linkId: string,
): string | null {
  const parentJointKeyByChildLinkId = Object.entries(joints).reduce<Map<string, string>>(
    (parentJointKeys, [jointKey, joint]) => {
      const childLinkId = resolveViewerJointChildLinkIdentity(joint);
      if (childLinkId) {
        parentJointKeys.set(childLinkId, jointKey);
      }
      return parentJointKeys;
    },
    new Map<string, string>(),
  );
  const visitedLinkIds = new Set<string>();
  let currentLinkId: string | null = linkId;

  while (currentLinkId && !visitedLinkIds.has(currentLinkId)) {
    visitedLinkIds.add(currentLinkId);

    const parentJointKey = parentJointKeyByChildLinkId.get(currentLinkId);
    if (!parentJointKey) {
      return null;
    }

    const parentJoint = joints[parentJointKey];
    if (isSingleDofJoint(parentJoint)) {
      return parentJointKey;
    }

    currentLinkId = resolveViewerJointParentLinkIdentity(parentJoint);
  }

  return null;
}

export function resolveActiveViewerJointKeyFromSelection(
  joints: Record<string, ViewerJointLike> | null | undefined,
  selection: ViewerSelection | null | undefined,
): string | null {
  if (!joints || !selection?.type || !selection.id) {
    return null;
  }

  if (selection.type === 'joint') {
    const jointKey = resolveViewerJointKey(joints, selection.id);
    const joint = jointKey ? joints[jointKey] : undefined;
    return isSingleDofJoint(joint) ? jointKey : null;
  }

  if (selection.type === 'tendon') {
    return null;
  }

  return resolveNearestControllableParentJointKey(joints, selection.id);
}
