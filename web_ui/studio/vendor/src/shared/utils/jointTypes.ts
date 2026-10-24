const SINGLE_DOF_JOINT_TYPES = new Set(['revolute', 'continuous', 'prismatic']);

type JointTypeLike = {
  jointType?: unknown;
  type?: unknown;
};

function asJointTypeLike(joint: unknown): JointTypeLike | null {
  return typeof joint === 'object' && joint !== null ? (joint as JointTypeLike) : null;
}

export const getJointType = (joint: unknown): string => {
  const jointLike = asJointTypeLike(joint);
  return String(jointLike?.jointType ?? jointLike?.type ?? '').toLowerCase();
};

export const isSingleDofJoint = (joint: unknown): boolean => {
  return SINGLE_DOF_JOINT_TYPES.has(getJointType(joint));
};

export const getSingleDofJointEntries = <TJoint>(
  joints: Record<string, TJoint> | null | undefined,
): Array<[string, TJoint]> => {
  return Object.entries(joints ?? {}).filter(([, joint]) => isSingleDofJoint(joint));
};

export const hasSingleDofJoints = <TJoint>(
  joints: Record<string, TJoint> | null | undefined,
): boolean => {
  return getSingleDofJointEntries(joints).length > 0;
};
