import type { UrdfJoint, UrdfLink } from '@/types';

export interface DeletionPlan {
  toDeleteLinks: Set<string>;
  toDeleteJoints: Set<string>;
}

/**
 * Compute which links and joints would be removed by deleting the subtree
 * rooted at `targetLinkId`. Returns null if target is the root (cannot delete).
 */
export function buildDeletionPlan(
  targetLinkId: string,
  links: Record<string, UrdfLink>,
  joints: Record<string, UrdfJoint>,
  rootLinkId: string,
): DeletionPlan | null {
  if (targetLinkId === rootLinkId) return null;

  const toDeleteLinks = new Set<string>();
  const toDeleteJoints = new Set<string>();

  const collect = (currentLinkId: string, visited: Set<string>) => {
    if (visited.has(currentLinkId)) return;
    visited.add(currentLinkId);
    toDeleteLinks.add(currentLinkId);

    Object.values(joints).forEach((joint) => {
      if (joint.parentLinkId === currentLinkId) {
        toDeleteJoints.add(joint.id);
        collect(joint.childLinkId, visited);
      }
      if (joint.childLinkId === currentLinkId) {
        toDeleteJoints.add(joint.id);
      }
    });
  };

  collect(targetLinkId, new Set<string>());
  return { toDeleteLinks, toDeleteJoints };
}

/**
 * Apply a deletion plan to produce filtered links and joints records.
 */
export function applyDeletionPlan(
  links: Record<string, UrdfLink>,
  joints: Record<string, UrdfJoint>,
  plan: DeletionPlan,
): { links: Record<string, UrdfLink>; joints: Record<string, UrdfJoint> } {
  const nextLinks: Record<string, UrdfLink> = {};
  for (const [id, link] of Object.entries(links)) {
    if (!plan.toDeleteLinks.has(id)) {
      nextLinks[id] = link;
    }
  }

  const nextJoints: Record<string, UrdfJoint> = {};
  for (const [id, joint] of Object.entries(joints)) {
    if (!plan.toDeleteJoints.has(id)) {
      nextJoints[id] = joint;
    }
  }

  return { links: nextLinks, joints: nextJoints };
}
