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

"""Experimental, process-local serial tree reductions for the canonical GPU mode.

Call ``install()`` before constructing any simulation or capturing CUDA graphs.
This module changes only Python kernel references; installed dependencies and
project source files remain untouched. Each replacement keeps the existing 2-D
launch shape and lets node 0 perform all node updates for its world in ascending
``body_tree_`` index order. The callers still visit tree depths in reverse order.

Original kernels in mujoco-warp 3.11.0:
* ``mujoco_warp/_src/smooth.py:700`` ``_subtree_com_acc`` updates
  ``Data.subtree_com`` (mass-weighted positions before the existing division).
* ``mujoco_warp/_src/smooth.py:1030`` ``_crb_accumulate`` updates
  ``Data.crb`` (10-component composite body inertias).
* ``mujoco_warp/_src/io.py:2754`` ``_accumulate_subtreemass`` updates
  ``Model.body_subtreemass`` during ``set_const_fixed``.

The three original formulas and their body/parent guards are retained. Only the
order of floating-point additions changes. This is a diagnostic intervention,
not a guarantee that the complete physics pipeline becomes deterministic.
"""

import warp as wp

from mujoco_warp._src import io, smooth
from mujoco_warp._src.types import vec10

wp.set_module_options({"deterministic": wp.DeterministicMode.NOT_GUARANTEED})


@wp.kernel(enable_backward=False)
def serial_subtree_com_acc(
    body_parentid: wp.array[int],
    subtree_com_in: wp.array2d[wp.vec3],
    body_tree_: wp.array[int],
    subtree_com_out: wp.array2d[wp.vec3],
):
    worldid, nodeid = wp.tid()
    if nodeid != 0:
        return
    for index in range(body_tree_.shape[0]):
        bodyid = body_tree_[index]
        pid = body_parentid[bodyid]
        if bodyid != 0:
            subtree_com_out[worldid, pid] = (
                subtree_com_out[worldid, pid] + subtree_com_in[worldid, bodyid]
            )


@wp.kernel(enable_backward=False)
def serial_crb_accumulate(
    body_parentid: wp.array[int],
    crb_in: wp.array2d[vec10],
    body_tree_: wp.array[int],
    crb_out: wp.array2d[vec10],
):
    worldid, nodeid = wp.tid()
    if nodeid != 0:
        return
    for index in range(body_tree_.shape[0]):
        bodyid = body_tree_[index]
        pid = body_parentid[bodyid]
        if pid != 0:
            crb_out[worldid, pid] = crb_out[worldid, pid] + crb_in[worldid, bodyid]


@wp.kernel(enable_backward=False)
def serial_accumulate_subtreemass(
    body_parentid: wp.array[int],
    body_subtreemass_io: wp.array2d[float],
    body_tree_: wp.array[int],
):
    worldid, nodeid = wp.tid()
    if nodeid != 0:
        return
    body_subtreemass_id = worldid % body_subtreemass_io.shape[0]
    for index in range(body_tree_.shape[0]):
        bodyid = body_tree_[index]
        parentid = body_parentid[bodyid]
        if bodyid != 0:
            body_subtreemass_io[body_subtreemass_id, parentid] = (
                body_subtreemass_io[body_subtreemass_id, parentid]
                + body_subtreemass_io[body_subtreemass_id, bodyid]
            )


_ORIGINALS = {}


def install():
    """Install the three replacements once and return the original references."""
    if not _ORIGINALS:
        _ORIGINALS.update({
            "smooth._subtree_com_acc": smooth._subtree_com_acc,
            "smooth._crb_accumulate": smooth._crb_accumulate,
            "io._accumulate_subtreemass": io._accumulate_subtreemass,
        })
    smooth._subtree_com_acc = serial_subtree_com_acc
    smooth._crb_accumulate = serial_crb_accumulate
    io._accumulate_subtreemass = serial_accumulate_subtreemass
    return dict(_ORIGINALS)
