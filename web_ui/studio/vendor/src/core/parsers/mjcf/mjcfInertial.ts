/**
 * MJCF inertial helpers
 *
 * Pure, dependency-free mass/inertia math extracted from mjcfParser.ts:
 * builds an empty link inertial and derives a mass-weighted inertial from a
 * body's geoms (point-mass aggregation about the combined center of mass).
 */

import { UrdfLink } from '@/types';

/**
 * Minimal geom shape needed to derive mass-weighted inertial properties.
 * Structurally compatible with the parser's MJCFGeom, but declared locally so
 * this module stays decoupled from the full geom interface (no import cycle).
 */
export interface MJCFInertialGeomSource {
  mass?: number;
  pos?: { x: number; y: number; z: number };
  fromto?: number[];
}

export function createEmptyLinkInertial(): NonNullable<UrdfLink['inertial']> {
  return {
    mass: 0,
    origin: { xyz: { x: 0, y: 0, z: 0 }, rpy: { r: 0, p: 0, y: 0 } },
    inertia: { ixx: 0, ixy: 0, ixz: 0, iyy: 0, iyz: 0, izz: 0 },
  };
}

function getGeomMassCenter(geom: MJCFInertialGeomSource): { x: number; y: number; z: number } {
  if (geom.pos) {
    return geom.pos;
  }

  if (geom.fromto && geom.fromto.length >= 6) {
    return {
      x: ((geom.fromto[0] ?? 0) + (geom.fromto[3] ?? 0)) / 2,
      y: ((geom.fromto[1] ?? 0) + (geom.fromto[4] ?? 0)) / 2,
      z: ((geom.fromto[2] ?? 0) + (geom.fromto[5] ?? 0)) / 2,
    };
  }

  return { x: 0, y: 0, z: 0 };
}

export function deriveGeomMassInertial(
  geoms: MJCFInertialGeomSource[],
): NonNullable<UrdfLink['inertial']> | null {
  const massGeoms = geoms.filter(
    (geom) => typeof geom.mass === 'number' && Number.isFinite(geom.mass) && (geom.mass ?? 0) > 0,
  );

  if (massGeoms.length === 0) {
    return null;
  }

  const totalMass = massGeoms.reduce((sum, geom) => sum + (geom.mass ?? 0), 0);
  if (!Number.isFinite(totalMass) || totalMass <= 0) {
    return null;
  }

  const weightedCenter = massGeoms.reduce(
    (sum, geom) => {
      const mass = geom.mass ?? 0;
      const center = getGeomMassCenter(geom);
      return {
        x: sum.x + center.x * mass,
        y: sum.y + center.y * mass,
        z: sum.z + center.z * mass,
      };
    },
    { x: 0, y: 0, z: 0 },
  );

  const centerOfMass = {
    x: weightedCenter.x / totalMass,
    y: weightedCenter.y / totalMass,
    z: weightedCenter.z / totalMass,
  };

  const inertia = massGeoms.reduce(
    (sum, geom) => {
      const mass = geom.mass ?? 0;
      const center = getGeomMassCenter(geom);
      const dx = center.x - centerOfMass.x;
      const dy = center.y - centerOfMass.y;
      const dz = center.z - centerOfMass.z;

      return {
        ixx: sum.ixx + mass * (dy * dy + dz * dz),
        ixy: sum.ixy - mass * dx * dy,
        ixz: sum.ixz - mass * dx * dz,
        iyy: sum.iyy + mass * (dx * dx + dz * dz),
        iyz: sum.iyz - mass * dy * dz,
        izz: sum.izz + mass * (dx * dx + dy * dy),
      };
    },
    { ixx: 0, ixy: 0, ixz: 0, iyy: 0, iyz: 0, izz: 0 },
  );

  return {
    mass: totalMass,
    origin: {
      xyz: centerOfMass,
      rpy: { r: 0, p: 0, y: 0 },
    },
    inertia,
  };
}
