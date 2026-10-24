import type { UrdfJoint, UrdfLink } from '@/types';

export function resolveLinkKey(
  links: Record<string, UrdfLink>,
  identity: string | null | undefined,
): string | null {
  if (!identity) return null;
  if (identity in links) return identity;

  for (const [linkId, link] of Object.entries(links)) {
    if (link.name === identity) {
      return linkId;
    }
  }

  return null;
}

export function resolveJointKey(
  joints: Record<string, UrdfJoint>,
  identity: string | null | undefined,
): string | null {
  if (!identity) return null;
  if (identity in joints) return identity;

  for (const [jointId, joint] of Object.entries(joints)) {
    if (joint.name === identity) {
      return jointId;
    }
  }

  return null;
}
