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

"""Canonical GPU mode: deterministic contact sensor matching order.

Adapted from MuJoCo Warp 3.11.0 ``sensor._contact_match``. The matching,
direction and reduction criteria are unchanged. Overflow sets a sticky error
flag; its consumer count is clamped to avoid out-of-bounds reads before failure. Only the
work partition and match-slot allocation change: one thread handles one
(world, contact sensor) and visits matching contacts in ascending global conid.

``install()`` changes a Python kernel reference in this process only. It leaves
the upstream ``_contact_sort`` and installed dependency files untouched.

The default path scans all contacts and explicitly filters world IDs. Native
RUN_TO_RUN atomic ordering can make slots repeat without putting all contacts
of a world next to each other: collision kernels append different geom types.
Use ``install(world_sorted=True)`` only after independently ensuring that the
valid contact prefix is sorted by world ID. That optional path uses two binary
searches and scans just that world's contiguous interval.

The upstream launch remains (nsensorcontact, naconmax); its second coordinate
is interpreted here as worldid. Extra threads return immediately. This relies
on naconmax >= nworld, as in the diagnostic tasks; install validates signature,
and ``validate_capacity(data)`` checks that launch precondition before capture.
"""

import inspect

import warp as wp

from mujoco_warp._src import sensor as _sensor
from mujoco_warp._src import support
from mujoco_warp._src.sensor import _check_match
from mujoco_warp._src.types import ContactType
from mujoco_warp._src.types import ObjType
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import vec5
from mujoco_warp._src.util_misc import inside_geom


MODULE_OPTIONS = {"deterministic": wp.DeterministicMode.NOT_GUARANTEED}
wp.set_module_options(MODULE_OPTIONS)

_ORIGINAL_CONTACT_MATCH = _sensor._contact_match
_KERNELS = {}


def _make_contact_match(world_sorted: bool):
  WORLD_SORTED = world_sorted

  @wp.kernel(module="unique", enable_backward=False, module_options=MODULE_OPTIONS)
  def _contact_match_serial(
    # Model:
    opt_cone: int,
    opt_contact_sensor_maxmatch: int,
    opt_warn_overflow: bool,
    body_parentid: wp.array[int],
    geom_bodyid: wp.array[int],
    site_type: wp.array[int],
    site_size: wp.array[wp.vec3],
    sensor_objtype: wp.array[int],
    sensor_objid: wp.array[int],
    sensor_reftype: wp.array[int],
    sensor_refid: wp.array[int],
    sensor_intprm: wp.array2d[int],
    sensor_contact_adr: wp.array[int],
    # Data in:
    site_xpos_in: wp.array2d[wp.vec3],
    site_xmat_in: wp.array2d[wp.mat33],
    contact_dist_in: wp.array[float],
    contact_pos_in: wp.array[wp.vec3],
    contact_frame_in: wp.array[wp.mat33],
    contact_friction_in: wp.array[vec5],
    contact_dim_in: wp.array[int],
    contact_geom_in: wp.array[wp.vec2i],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    contact_type_in: wp.array[int],
    efc_force_in: wp.array2d[float],
    njmax_in: int,
    nacon_in: wp.array[int],
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    sensor_contact_nmatch_out: wp.array2d[int],
    sensor_contact_matchid_out: wp.array3d[int],
    sensor_contact_criteria_out: wp.array3d[float],
    sensor_contact_direction_out: wp.array3d[float],
  ):
    contactsensorid, worldid = wp.tid()
    if worldid >= sensor_contact_nmatch_out.shape[0]:
      return

    sensorid = sensor_contact_adr[contactsensorid]
    # sensor information
    objtype = sensor_objtype[sensorid]
    objid = sensor_objid[sensorid]
    reftype = sensor_reftype[sensorid]
    refid = sensor_refid[sensorid]
    reduce = sensor_intprm[sensorid, 1]

    start = int(0)
    # The original fixed-capacity launch visits at most this many slots even
    # when a contact overflow has made nacon larger than its backing arrays.
    end = wp.min(nacon_in[0], contact_worldid_in.shape[0])
    contact_end = end
    if wp.static(WORLD_SORTED):
      # Lower bound: first contact with world >= this world.
      lo = int(0)
      hi = contact_end
      while lo < hi:
        mid = (lo + hi) // 2
        if contact_worldid_in[mid] < worldid:
          lo = mid + 1
        else:
          hi = mid
      start = lo
      # Upper bound: first contact with world > this world.
      hi = contact_end
      while lo < hi:
        mid = (lo + hi) // 2
        if contact_worldid_in[mid] <= worldid:
          lo = mid + 1
        else:
          hi = mid
      end = lo

    nmatch = int(0)
    for contactid in range(start, end):
      if contact_worldid_in[contactid] != worldid:
        continue
      if not contact_type_in[contactid] & ContactType.CONSTRAINT:
        continue

      # site filter
      if objtype == ObjType.SITE:
        if not inside_geom(
          site_xpos_in[worldid, objid], site_xmat_in[worldid, objid], site_size[objid], site_type[objid], contact_pos_in[contactid]
        ):
          continue

      # unknown-unknown match
      if objtype == ObjType.UNKNOWN and reftype == ObjType.UNKNOWN:
        dir = 1.0
      else:
        # contact information
        geom = contact_geom_in[contactid]
        geom1 = geom[0]
        geom2 = geom[1]
        body1 = geom_bodyid[geom1]
        body2 = geom_bodyid[geom2]

        # check match of sensor objects with contact objects
        match11 = _check_match(body_parentid, body1, geom1, objtype, objid)
        match12 = _check_match(body_parentid, body2, geom2, objtype, objid)
        match21 = _check_match(body_parentid, body1, geom1, reftype, refid)
        match22 = _check_match(body_parentid, body2, geom2, reftype, refid)

        # if a sensor object is specified, it must be involved in the contact
        if not match11 and not match12:
          continue
        if not match21 and not match22:
          continue

        # determine direction
        dir = 1.0
        if objtype != ObjType.UNKNOWN and reftype != ObjType.UNKNOWN:
          # both obj1 and obj2 specified: direction depends on order
          order_regular = match11 and match22
          order_reverse = match12 and match21
          if not order_regular and not order_reverse:
            continue
          if order_reverse and not order_regular:
            dir = -1.0
        elif objtype != ObjType.UNKNOWN:
          if not match11:
            dir = -1.0
        elif reftype != ObjType.UNKNOWN:
          if not match22:
            dir = -1.0

      # Preserve the total match count, including matches beyond capacity.
      contactmatchid = nmatch
      nmatch += 1
      if contactmatchid >= opt_contact_sensor_maxmatch:
        if opt_warn_overflow:
          wp.printf("contact match overflow: please increase Option.contact_sensor_maxmatch to %u\n", contactmatchid)
        wp.atomic_or(overflow_out, worldid, OverflowType.CONTACT_MATCH)
        continue

      sensor_contact_matchid_out[worldid, contactsensorid, contactmatchid] = contactid

      if reduce == 1:  # mindist
        sensor_contact_criteria_out[worldid, contactsensorid, contactmatchid] = contact_dist_in[contactid]
      elif reduce == 2:  # maxforce
        contact_force = support.contact_force_fn(
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
          False,
        )
        force_magnitude = (
          contact_force[0] * contact_force[0] + contact_force[1] * contact_force[1] + contact_force[2] * contact_force[2]
        )
        sensor_contact_criteria_out[worldid, contactsensorid, contactmatchid] = -force_magnitude

      # contact direction
      sensor_contact_direction_out[worldid, contactsensorid, contactmatchid] = dir

    sensor_contact_nmatch_out[worldid, contactsensorid] = wp.min(nmatch, opt_contact_sensor_maxmatch)

  return _contact_match_serial


def validate_capacity(data):
  """Check the reused upstream launch covers every world before graph capture."""
  if data.naconmax < data.nworld:
    raise ValueError(
      "serial sensor matching reuses the upstream naconmax launch and requires "
      f"naconmax >= nworld, got {data.naconmax} < {data.nworld}"
    )


def install(world_sorted=False):
  """Install this process-only experiment; default input may interleave worlds."""
  key = bool(world_sorted)
  if key not in _KERNELS:
    kernel = _make_contact_match(key)
    original = inspect.signature(_ORIGINAL_CONTACT_MATCH.func)
    replacement = inspect.signature(kernel.func)
    if original != replacement:
      raise RuntimeError(f"MuJoCo Warp contact matcher signature mismatch: {original} != {replacement}")
    _KERNELS[key] = kernel
  _sensor._contact_match = _KERNELS[key]
  return {"contact_match": "serial_per_world_sensor", "world_sorted": key}


def uninstall():
  """Restore the kernel reference that was present when this module loaded."""
  _sensor._contact_match = _ORIGINAL_CONTACT_MATCH
