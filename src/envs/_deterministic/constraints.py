"""Canonical GPU mode: canonical EFC rows for the rigid dense CG path.

Call after make_constraint(), with contacts already in canonical order, and
before any consumer of the new EFC layout. Keys include the world for both
active and padding rows, so a global fixed-capacity radix sort keeps each
world's njmax-sized segment intact. Data array allocations never change.

This helper only permutes completed constraint data and its contact reverse
references. It does not change any physical formula or claim that other Warp
reductions are deterministic. All scratch arrays are allocated at construction.
"""

from functools import lru_cache

import warp as wp

from mujoco_warp._src.types import ConeType
from mujoco_warp._src.types import ConstraintType
from mujoco_warp._src.types import ContactType
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import SolverType


EFC_ROW_FIELDS = (
    "type", "id", "J", "pos", "margin", "D", "vel", "aref",
    "frictionloss", "Jqvel", "force", "state", "island",
)
BAD_COUNT = 1
BAD_IDENTITY = 2
BAD_COMPONENT = 4
BAD_PERMUTATION = 8
DUPLICATE_IDENTITY = 16
BAD_CONTACT_ADDRESS = 32
BAD_LAYOUT = 64
MODULE_OPTIONS = {"deterministic": wp.DeterministicMode.NOT_GUARANTEED}
wp.set_module_options(MODULE_OPTIONS)


@lru_cache(maxsize=None)
def _keys_kernel(nworld: int, njmax: int, ncontact: int, nv: int,
                 njnt: int, nmaxpyramid: int, id_bits: int,
                 component_bits: int):
    id_shift = component_bits
    type_shift = component_bits + id_bits
    world_shift = type_shift + 4

    @wp.kernel(module="unique", enable_backward=False,
               module_options=MODULE_OPTIONS)
    def make_keys(
        nefc: wp.array[int], efc_type: wp.array2d[int],
        efc_id: wp.array2d[int], nacon: wp.array[int],
        contact_world: wp.array[int], contact_type: wp.array[int],
        contact_dim: wp.array[int], contact_address: wp.array2d[int],
        keys: wp.array[wp.uint64], permutation: wp.array[wp.int32],
        bad_flags: wp.array[wp.int32],
    ):
        flat = wp.tid()
        world = flat // njmax
        row = flat % njmax
        permutation[flat] = flat
        # World-local padding preserves njmax rows per world after global sort.
        key = ((wp.uint64(world) << wp.uint64(world_shift))
               | (wp.uint64(15) << wp.uint64(type_shift))
               | (wp.uint64(row) << wp.uint64(id_shift)))
        if row < nefc[world]:
            kind = efc_type[world, row]
            object_id = efc_id[world, row]
            component = int(0)
            valid = True
            if kind == ConstraintType.FRICTION_DOF:
                if object_id < 0 or object_id >= nv:
                    valid = False
                    bad_flags[flat] = bad_flags[flat] | BAD_IDENTITY
            elif kind == ConstraintType.LIMIT_JOINT:
                if object_id < 0 or object_id >= njnt:
                    valid = False
                    bad_flags[flat] = bad_flags[flat] | BAD_IDENTITY
            elif kind >= ConstraintType.CONTACT_FRICTIONLESS and kind <= ConstraintType.CONTACT_ELLIPTIC:
                if object_id < 0 or object_id >= nacon[0] or object_id >= ncontact:
                    valid = False
                    bad_flags[flat] = bad_flags[flat] | BAD_IDENTITY
                else:
                    if contact_world[object_id] != world or not (contact_type[object_id] & ContactType.CONSTRAINT):
                        valid = False
                        bad_flags[flat] = bad_flags[flat] | BAD_IDENTITY
                    dim = contact_dim[object_id]
                    count = int(0)
                    if dim == 1:
                        count = 1
                        if kind != ConstraintType.CONTACT_FRICTIONLESS:
                            valid = False
                            bad_flags[flat] = bad_flags[flat] | BAD_COMPONENT
                    elif dim == 3 or dim == 4 or dim == 6:
                        count = 2 * (dim - 1)
                        if kind != ConstraintType.CONTACT_PYRAMIDAL:
                            valid = False
                            bad_flags[flat] = bad_flags[flat] | BAD_COMPONENT
                    else:
                        valid = False
                        bad_flags[flat] = bad_flags[flat] | BAD_COMPONENT
                    base = contact_address[object_id, 0]
                    component = row - base
                    if base < 0 or component < 0 or component >= count or component >= nmaxpyramid:
                        valid = False
                        bad_flags[flat] = bad_flags[flat] | BAD_COMPONENT
                    elif contact_address[object_id, component] != row:
                        valid = False
                        bad_flags[flat] = bad_flags[flat] | BAD_CONTACT_ADDRESS
            else:
                # Equality, tendon, and elliptic paths are outside this helper.
                valid = False
                bad_flags[flat] = bad_flags[flat] | BAD_IDENTITY
            if valid:
                key = ((wp.uint64(world) << wp.uint64(world_shift))
                       | (wp.uint64(kind) << wp.uint64(type_shift))
                       | (wp.uint64(object_id) << wp.uint64(id_shift))
                       | wp.uint64(component))
        keys[flat] = key

    return make_keys


@wp.kernel(enable_backward=False)
def _check_counts(
    ne: wp.array[int], nf: wp.array[int], nl: wp.array[int],
    nefc: wp.array[int], jtdaj_nblock: wp.array[int],
    njmax: int, flag_offset: int, bad_flags: wp.array[wp.int32],
):
    world = wp.tid()
    flag = flag_offset + world
    if nefc[world] < 0 or nefc[world] > njmax:
        bad_flags[flag] = bad_flags[flag] | BAD_COUNT
    if ne[world] != 0 or nf[world] < 0 or nl[world] < 0 or nf[world] + nl[world] > nefc[world] or jtdaj_nblock[world] != 0:
        bad_flags[flag] = bad_flags[flag] | BAD_LAYOUT


@wp.kernel(enable_backward=False)
def _build_inverse(
    permutation: wp.array[wp.int32], njmax: int,
    inverse: wp.array2d[wp.int32], bad_flags: wp.array[wp.int32],
):
    flat = wp.tid()
    world = flat // njmax
    row = flat % njmax
    old = permutation[flat]
    if old < world * njmax or old >= (world + 1) * njmax:
        bad_flags[flat] = bad_flags[flat] | BAD_PERMUTATION
    else:
        inverse[world, old % njmax] = row


@wp.kernel(enable_backward=False)
def _check_inverse_and_duplicates(
    keys: wp.array[wp.uint64], permutation: wp.array[wp.int32],
    inverse: wp.array2d[wp.int32], nefc: wp.array[int],
    njmax: int, bad_flags: wp.array[wp.int32],
):
    flat = wp.tid()
    world = flat // njmax
    row = flat % njmax
    newrow = inverse[world, row]
    if newrow < 0 or newrow >= njmax:
        bad_flags[flat] = bad_flags[flat] | BAD_PERMUTATION
    elif permutation[world * njmax + newrow] != flat:
        bad_flags[flat] = bad_flags[flat] | BAD_PERMUTATION
    if row > 0 and row < nefc[world]:
        if keys[flat] == keys[flat - 1]:
            bad_flags[flat] = bad_flags[flat] | DUPLICATE_IDENTITY


@lru_cache(maxsize=None)
def _field_kernels(dtype, ndim: int):
    if ndim == 2:
        @wp.kernel(module="unique", enable_backward=False,
                   module_options=MODULE_OPTIONS)
        def gather(src: wp.array2d(dtype=dtype), old: wp.array[wp.int32],
                   njmax: int, dst: wp.array2d(dtype=dtype)):
            world, row = wp.tid()
            oldflat = old[world * njmax + row]
            if oldflat >= world * njmax and oldflat < (world + 1) * njmax:
                dst[world, row] = src[world, oldflat % njmax]
            else:
                dst[world, row] = src[world, row]

        @wp.kernel(module="unique", enable_backward=False,
                   module_options=MODULE_OPTIONS)
        def scatter(src: wp.array2d(dtype=dtype), dst: wp.array2d(dtype=dtype)):
            world, row = wp.tid()
            dst[world, row] = src[world, row]
    else:
        @wp.kernel(module="unique", enable_backward=False,
                   module_options=MODULE_OPTIONS)
        def gather(src: wp.array3d(dtype=dtype), old: wp.array[wp.int32],
                   njmax: int, dst: wp.array3d(dtype=dtype)):
            world, row, column = wp.tid()
            oldflat = old[world * njmax + row]
            if oldflat >= world * njmax and oldflat < (world + 1) * njmax:
                dst[world, row, column] = src[world, oldflat % njmax, column]
            else:
                dst[world, row, column] = src[world, row, column]

        @wp.kernel(module="unique", enable_backward=False,
                   module_options=MODULE_OPTIONS)
        def scatter(src: wp.array3d(dtype=dtype), dst: wp.array3d(dtype=dtype)):
            world, row, column = wp.tid()
            dst[world, row, column] = src[world, row, column]
    return gather, scatter


@wp.kernel(enable_backward=False)
def _remap_contact_addresses(
    nacon: wp.array[int], contact_world: wp.array[int],
    contact_type: wp.array[int], nefc: wp.array[int],
    efc_type: wp.array2d[int], efc_id: wp.array2d[int],
    inverse: wp.array2d[wp.int32], nworld: int, njmax: int,
    ncontact: int, flag_offset: int,
    address: wp.array2d[int], bad_flags: wp.array[wp.int32],
):
    contactid = wp.tid()
    flag = flag_offset + contactid
    if contactid == 0:
        if nacon[0] < 0 or nacon[0] > ncontact:
            bad_flags[flag_offset + ncontact] = bad_flags[flag_offset + ncontact] | BAD_COUNT
    if contactid >= nacon[0]:
        return
    world = contact_world[contactid]
    if world < 0 or world >= nworld:
        bad_flags[flag] = bad_flags[flag] | BAD_IDENTITY
        return
    for component in range(address.shape[1]):
        oldrow = address[contactid, component]
        if oldrow >= 0:
            if not (contact_type[contactid] & ContactType.CONSTRAINT) or oldrow >= njmax or oldrow >= nefc[world]:
                bad_flags[flag] = bad_flags[flag] | BAD_CONTACT_ADDRESS
            else:
                newrow = inverse[world, oldrow]
                if newrow < 0 or newrow >= njmax or newrow >= nefc[world]:
                    bad_flags[flag] = bad_flags[flag] | BAD_PERMUTATION
                elif efc_id[world, newrow] != contactid or efc_type[world, newrow] < ConstraintType.CONTACT_FRICTIONLESS or efc_type[world, newrow] > ConstraintType.CONTACT_ELLIPTIC:
                    bad_flags[flag] = bad_flags[flag] | BAD_CONTACT_ADDRESS
                else:
                    address[contactid, component] = newrow


@wp.kernel(enable_backward=False)
def _check_no_contacts(nacon: wp.array[int], flag: int,
                       bad_flags: wp.array[wp.int32]):
    if nacon[0] != 0:
        bad_flags[flag] = bad_flags[flag] | BAD_COUNT


class CanonicalConstraints:
    """Capture-safe EFC permutation with sticky, single-writer device flags.

    Supported: rigid models, dense Jacobian, CG, pyramidal cone, no equality,
    no tendons, no sleeping. Contacts must already have canonical slot IDs.
    ``validate`` synchronizes outside a graph and reports malformed data rather
    than silently accepting a partial or ambiguous ordering.
    """

    def __init__(self, m, d):
        unsupported = []
        if bool(m.is_sparse):
            unsupported.append("sparse Jacobian")
        if int(m.opt.solver) != int(SolverType.CG):
            unsupported.append("non-CG solver")
        if int(m.opt.cone) != int(ConeType.PYRAMIDAL):
            unsupported.append("non-pyramidal cone")
        if int(m.nflex):
            unsupported.append("flex")
        if int(m.neq):
            unsupported.append("equality constraints")
        if int(m.ntendon):
            unsupported.append("tendons")
        if int(m.opt.enableflags) & int(EnableBit.SLEEP):
            unsupported.append("sleeping")
        if unsupported:
            raise NotImplementedError("CanonicalConstraints does not support: " + ", ".join(unsupported))
        self.model = m
        self.data = d
        self.world_count = int(d.nworld)
        self.njmax = int(d.njmax)
        self.contact_capacity = int(d.naconmax)
        self.capacity = self.world_count * self.njmax
        if self.world_count <= 0 or self.njmax < 0 or self.contact_capacity < 0:
            raise ValueError("Invalid world, constraint, or contact capacity")
        if max(self.capacity, self.contact_capacity) > (1 << 31) - 1:
            raise ValueError("EFC and contact capacities must fit nonnegative int32 indices")
        if d.contact.efc_address.shape[1] < 1:
            raise ValueError("Contact.efc_address must have at least one component")
        self.component_count = int(d.contact.efc_address.shape[1])
        if self.component_count < int(m.nmaxpyramid):
            raise ValueError("Contact.efc_address is narrower than nmaxpyramid")
        self.id_bits = max(1, (max(self.contact_capacity, int(m.nv), int(m.njnt), self.njmax, 1) - 1).bit_length())
        self.component_bits = max(1, (max(self.component_count, 1) - 1).bit_length())
        self.key_bits = max(1, (self.world_count - 1).bit_length()) + 4 + self.id_bits + self.component_bits
        if self.key_bits > 63:
            raise ValueError(f"Constraint identity needs {self.key_bits} bits; maximum is 63")
        self.device = d.nefc.device
        for name in ("J_rownnz", "J_rowadr", "J_colind"):
            if getattr(d.efc, name).size != 0:
                raise ValueError(f"Dense CG requires empty EFC.{name}")
        self.keys = wp.empty(2 * self.capacity, dtype=wp.uint64, device=self.device)
        self.permutation = wp.empty(2 * self.capacity, dtype=wp.int32, device=self.device)
        self.inverse = wp.full((self.world_count, self.njmax), -1,
                               dtype=wp.int32, device=self.device)
        self._world_flag_offset = self.capacity + self.contact_capacity + 1
        self.bad_flags = wp.zeros(self._world_flag_offset + self.world_count,
                                  dtype=wp.int32, device=self.device)
        self._keys_kernel = _keys_kernel(
            self.world_count, self.njmax, self.contact_capacity,
            int(m.nv), int(m.njnt), int(m.nmaxpyramid),
            self.id_bits, self.component_bits,
        )
        self._fields = []
        for name in EFC_ROW_FIELDS:
            src = getattr(d.efc, name)
            ndim = 3 if name == "J" else 2
            if src.device != self.device or src.ndim != ndim or not src.is_contiguous:
                raise ValueError(f"EFC.{name} has incompatible device, dimensions, or strides")
            if src.shape[0] != self.world_count or src.shape[1] < self.njmax:
                raise ValueError(f"EFC.{name} has incompatible world or row capacity")
            if name == "J" and src.shape[2] < int(m.nv):
                raise ValueError("Dense EFC.J has too few DOF columns")
            shape = (self.world_count, self.njmax) + src.shape[2:]
            scratch = wp.empty(shape, dtype=src.dtype, device=self.device)
            gather, scatter = _field_kernels(src.dtype, src.ndim)
            self._fields.append((name, src, scratch, gather, scatter))
        self._contact_address = d.contact.efc_address

    def prepare(self):
        """Compile owned kernels before capture; execute no simulation steps."""
        modules = {
            _check_counts.module, _build_inverse.module,
            _check_inverse_and_duplicates.module,
            _remap_contact_addresses.module, _check_no_contacts.module,
        }
        if self.capacity:
            modules.add(self._keys_kernel.module)
        for _, _, _, gather, scatter in self._fields:
            modules.update((gather.module, scatter.module))
        for module in modules:
            module.load(self.device)
        return self

    def warmup(self):
        """Sort a completed valid layout once before capturing sort/scan scratch."""
        self.prepare()
        self.sort()
        return self.validate()

    def sort(self):
        """Canonicalize row fields and contact addresses without host data reads."""
        d = self.data
        if d.contact.efc_address is not self._contact_address:
            raise RuntimeError("Contact.efc_address allocation changed; recreate sorter and graphs")
        with wp.ScopedDevice(self.device):
            wp.launch(_check_counts, dim=self.world_count, inputs=[
                d.ne, d.nf, d.nl, d.nefc, d.efc.jtdaj_nblock,
                self.njmax, self._world_flag_offset, self.bad_flags,
            ])
            if self.capacity:
                wp.launch(self._keys_kernel, dim=self.capacity, inputs=[
                    d.nefc, d.efc.type, d.efc.id, d.nacon,
                    d.contact.worldid, d.contact.type, d.contact.dim,
                    d.contact.efc_address, self.keys, self.permutation,
                    self.bad_flags,
                ])
                wp.utils.radix_sort_pairs(self.keys, self.permutation, self.capacity)
                wp.launch(_build_inverse, dim=self.capacity, inputs=[
                    self.permutation, self.njmax, self.inverse, self.bad_flags,
                ])
                wp.launch(_check_inverse_and_duplicates, dim=self.capacity, inputs=[
                    self.keys, self.permutation, self.inverse, d.nefc,
                    self.njmax, self.bad_flags,
                ])
                for name, src, scratch, gather, scatter in self._fields:
                    if getattr(d.efc, name) is not src:
                        raise RuntimeError(f"EFC.{name} allocation changed; recreate sorter and graphs")
                    wp.launch(gather, dim=scratch.shape, inputs=[
                        src, self.permutation, self.njmax,
                    ], outputs=[scratch])
                    wp.launch(scatter, dim=scratch.shape,
                              inputs=[scratch], outputs=[src])
            if self.contact_capacity:
                wp.launch(_remap_contact_addresses, dim=self.contact_capacity, inputs=[
                    d.nacon, d.contact.worldid, d.contact.type, d.nefc,
                    d.efc.type, d.efc.id,
                    self.inverse, self.world_count, self.njmax,
                    self.contact_capacity, self.capacity,
                    d.contact.efc_address, self.bad_flags,
                ])
            else:
                wp.launch(_check_no_contacts, dim=1, inputs=[
                    d.nacon, self.capacity, self.bad_flags,
                ])

    def clear_bad_flags(self):
        self.bad_flags.zero_()

    def validate(self):
        """Synchronize outside capture and reject any sticky device failure."""
        combined = 0
        slots = []
        for slot, value in enumerate(self.bad_flags.numpy()):
            if int(value):
                combined |= int(value)
                if len(slots) < 16:
                    slots.append((slot, int(value)))
        if combined:
            labels = (
                (BAD_COUNT, "constraint/contact count overflow"),
                (BAD_IDENTITY, "invalid constraint identity"),
                (BAD_COMPONENT, "invalid constraint component"),
                (BAD_PERMUTATION, "invalid row inverse/permutation"),
                (DUPLICATE_IDENTITY, "duplicate constraint identity"),
                (BAD_CONTACT_ADDRESS, "invalid contact EFC address"),
                (BAD_LAYOUT, "unsupported EFC layout/counts"),
            )
            reasons = [label for bit, label in labels if combined & bit]
            raise RuntimeError(f"Canonical EFC validation failed: {', '.join(reasons)}; slots={slots}")
        return {"valid": True, "capacity": self.capacity,
                "world_count": self.world_count, "njmax": self.njmax,
                "key_bits": self.key_bits}
