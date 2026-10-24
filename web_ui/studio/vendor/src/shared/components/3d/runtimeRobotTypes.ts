import type * as THREE from 'three';
import type {
  URDFCollider,
  URDFJoint,
  URDFLink,
  URDFVisual,
} from '@/core/parsers/urdf/loader';

export type RuntimeObject3D = THREE.Object3D & {
  isURDFRobot?: boolean;
  isURDFLink?: boolean;
  isURDFJoint?: boolean;
  isURDFVisual?: boolean;
  isURDFCollider?: boolean;
  isURDFMimicJoint?: boolean;
  urdfName?: string;
};

export type RuntimeLinkObject = URDFLink | RuntimeObject3D;
export type RuntimeJointObject = URDFJoint | RuntimeObject3D;
export type RuntimeVisualObject = URDFVisual | RuntimeObject3D;
export type RuntimeColliderObject = URDFCollider | RuntimeObject3D;

export type RuntimeRobotObject = RuntimeObject3D & {
  links?: Record<string, RuntimeLinkObject>;
  joints?: Record<string, RuntimeJointObject>;
  colliders?: Record<string, RuntimeColliderObject>;
  visual?: Record<string, RuntimeVisualObject>;
  visuals?: Record<string, RuntimeVisualObject>;
  frames?: Record<string, THREE.Object3D>;
};

export function asRuntimeObject3D(object: THREE.Object3D): RuntimeObject3D {
  return object as RuntimeObject3D;
}

export function asRuntimeRobotObject(object: THREE.Object3D): RuntimeRobotObject {
  return object as RuntimeRobotObject;
}
