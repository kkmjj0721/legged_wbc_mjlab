# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Canonical GPU mode: serial bias-force and subtree-momentum reductions.

Adapted from MuJoCo Warp 3.11.0 smooth._cfrc_backward, _linear_momentum,
_angular_momentum, and _cfrc_ext_contact. Each unchanged tree launch handles one tree
depth at a time, in reverse depth order. Within that depth only node/thread 0
acts for each world and visits body_tree_ in ascending index order.

Bodies in one tree depth are never each other's ancestors, so the existing
non-atomic own-body updates cannot depend on another same-depth body's update.
Contacts must be globally world-sorted before the postconstraint scatter.
The original guards, division placement, and the two distinct angular-momentum
parent additions are retained. This module changes process-local references
only, and complements the three tree reductions from WF-008.

The postconstraint contact scatter uses one thread per world, with ascending
global contact IDs and an explicit world filter. Its upstream naconmax-sized
launch must cover every world (naconmax >= nworld), checked by the experiment's
existing serial_sensor.validate_capacity before graph capture. It makes no
assumption that contact IDs are grouped by world.
"""

import inspect

import warp as wp

from mujoco_warp._src import smooth, support
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import vec5


MODULE_OPTIONS = {"deterministic": wp.DeterministicMode.NOT_GUARANTEED}
wp.set_module_options(MODULE_OPTIONS)


@wp.kernel(enable_backward=False)
def serial_cfrc_backward(
    body_parentid: wp.array[int],
    cfrc_int_in: wp.array2d[wp.spatial_vector],
    body_tree_: wp.array[int],
    cfrc_int_out: wp.array2d[wp.spatial_vector],
):
    worldid, nodeid = wp.tid()
    if nodeid != 0:
        return
    for index in range(body_tree_.shape[0]):
        bodyid = body_tree_[index]
        pid = body_parentid[bodyid]
        if bodyid != 0:
            cfrc_int_out[worldid, pid] = cfrc_int_out[worldid, pid] + cfrc_int_in[worldid, bodyid]


@wp.kernel(enable_backward=False)
def serial_linear_momentum(
    body_parentid: wp.array[int],
    body_subtreemass: wp.array2d[float],
    subtree_linvel_in: wp.array2d[wp.vec3],
    body_tree_: wp.array[int],
    subtree_linvel_out: wp.array2d[wp.vec3],
):
    worldid, nodeid = wp.tid()
    if nodeid != 0:
        return
    for index in range(body_tree_.shape[0]):
        bodyid = body_tree_[index]
        if bodyid:
            pid = body_parentid[bodyid]
            subtree_linvel_out[worldid, pid] = subtree_linvel_out[worldid, pid] + subtree_linvel_in[worldid, bodyid]
        subtree_linvel_out[worldid, bodyid] /= wp.max(MJ_MINVAL, body_subtreemass[worldid % body_subtreemass.shape[0], bodyid])


@wp.kernel(enable_backward=False)
def serial_angular_momentum(
    body_parentid: wp.array[int],
    body_mass: wp.array2d[float],
    body_subtreemass: wp.array2d[float],
    xipos_in: wp.array2d[wp.vec3],
    subtree_com_in: wp.array2d[wp.vec3],
    subtree_linvel_in: wp.array2d[wp.vec3],
    subtree_bodyvel_in: wp.array2d[wp.spatial_vector],
    body_tree_: wp.array[int],
    subtree_angmom_out: wp.array2d[wp.vec3],
):
    worldid, nodeid = wp.tid()
    if nodeid != 0:
        return
    for index in range(body_tree_.shape[0]):
        bodyid = body_tree_[index]
        if bodyid == 0:
            continue
        pid = body_parentid[bodyid]

        xipos = xipos_in[worldid, bodyid]
        com = subtree_com_in[worldid, bodyid]
        com_parent = subtree_com_in[worldid, pid]
        vel = subtree_bodyvel_in[worldid, bodyid]
        linvel = subtree_linvel_in[worldid, bodyid]
        linvel_parent = subtree_linvel_in[worldid, pid]
        mass = body_mass[worldid % body_mass.shape[0], bodyid]
        subtreemass = body_subtreemass[worldid % body_subtreemass.shape[0], bodyid]

        # momentum wrt body i
        dx = xipos - com
        dv = wp.spatial_bottom(vel) - linvel
        dp = dv * mass
        dL = wp.cross(dx, dp)

        # add to subtree i
        subtree_angmom_out[worldid, bodyid] += dL

        # add to parent; preserve the original two separate additions.
        subtree_angmom_out[worldid, pid] = subtree_angmom_out[worldid, pid] + subtree_angmom_out[worldid, bodyid]

        # momentum wrt parent
        dx = com - com_parent
        dv = linvel - linvel_parent
        dv *= subtreemass
        dL = wp.cross(dx, dv)
        subtree_angmom_out[worldid, pid] = subtree_angmom_out[worldid, pid] + dL


@wp.kernel(enable_backward=False)
def serial_cfrc_ext_contact(
    opt_cone: int,
    body_rootid: wp.array[int],
    geom_bodyid: wp.array[int],
    subtree_com_in: wp.array2d[wp.vec3],
    contact_pos_in: wp.array[wp.vec3],
    contact_frame_in: wp.array[wp.mat33],
    contact_friction_in: wp.array[vec5],
    contact_dim_in: wp.array[int],
    contact_geom_in: wp.array[wp.vec2i],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    efc_force_in: wp.array2d[float],
    njmax_in: int,
    nacon_in: wp.array[int],
    cfrc_ext_out: wp.array2d[wp.spatial_vector],
):
    worldid = wp.tid()
    if worldid >= cfrc_ext_out.shape[0]:
        return
    # Canonical mode globally sorts contacts by world. Find this world's
    # interval, retaining ascending conid order without scanning other worlds.
    count = wp.min(nacon_in[0], contact_worldid_in.shape[0])
    lo = int(0)
    hi = count
    while lo < hi:
        mid = (lo + hi) // 2
        if contact_worldid_in[mid] < worldid:
            lo = mid + 1
        else:
            hi = mid
    for contactid in range(lo, count):
        if contact_worldid_in[contactid] != worldid:
            break

        geom = contact_geom_in[contactid]
        id1 = geom_bodyid[geom[0]]
        id2 = geom_bodyid[geom[1]]
        if id1 == 0 and id2 == 0:
            continue

        # contact force in world frame
        force = support.contact_force_fn(
            opt_cone,
            contact_frame_in,
            contact_friction_in,
            contact_dim_in,
            contact_efc_address_in,
            efc_force_in,
            njmax_in,
            nacon_in,
            worldid,
            contactid,
            to_world_frame=True,
        )
        pos = contact_pos_in[contactid]

        # contact force on bodies; preserve each original signed operation.
        if id1:
            com1 = subtree_com_in[worldid, body_rootid[id1]]
            cfrc_ext_out[worldid, id1] = cfrc_ext_out[worldid, id1] - support.transform_force(force, com1 - pos)
        if id2:
            com2 = subtree_com_in[worldid, body_rootid[id2]]
            cfrc_ext_out[worldid, id2] = cfrc_ext_out[worldid, id2] + support.transform_force(force, com2 - pos)


_REPLACEMENTS = {
    "_cfrc_backward": serial_cfrc_backward,
    "_linear_momentum": serial_linear_momentum,
    "_angular_momentum": serial_angular_momentum,
    "_cfrc_ext_contact": serial_cfrc_ext_contact,
}
_ORIGINALS = {}


def install():
    """Validate exact signatures and install all four process-only kernels."""
    if not _ORIGINALS:
        for name, replacement in _REPLACEMENTS.items():
            original = getattr(smooth, name)
            if inspect.signature(original.func) != inspect.signature(replacement.func):
                raise RuntimeError(f"MuJoCo Warp smooth.{name} signature mismatch")
            _ORIGINALS[name] = original
    for name, replacement in _REPLACEMENTS.items():
        setattr(smooth, name, replacement)
    return dict(_ORIGINALS)


def uninstall():
    for name, original in _ORIGINALS.items():
        setattr(smooth, name, original)
