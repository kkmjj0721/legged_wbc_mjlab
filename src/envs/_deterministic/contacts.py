"""Canonical GPU mode: reorder rigid contacts before constraint construction.

The collision kernels and their scratch-memory behavior remain unchanged.  This
helper sorts only their completed Contact output, using the ordered geometry
pair and each pair's collision-local index as physical identity.  It must run
after collision() and before make_constraint(); EFC references are not remapped.
All Contact arrays retain their original allocations and pointers.
"""

from functools import lru_cache

import mujoco
import warp as wp


CONTACT_FIELDS = (
    "dist", "pos", "frame", "includemargin", "friction", "solref",
    "solreffriction", "solimp", "dim", "geom", "flex", "elem", "vert",
    "efc_address", "worldid", "type", "geomcollisionid",
)
UINT64_MAX = (1 << 64) - 1
BAD_COUNT = 1
BAD_IDENTITY = 2
DUPLICATE_IDENTITY = 4
MODULE_OPTIONS = {"deterministic": wp.DeterministicMode.NOT_GUARANTEED}
wp.set_module_options(MODULE_OPTIONS)


@lru_cache(maxsize=None)
def _key_kernel(world_count: int, geom_count: int, collision_count: int,
                geom_bits: int, collision_bits: int):
    geom1_shift = collision_bits
    geom0_shift = collision_bits + geom_bits
    world_shift = collision_bits + 2 * geom_bits

    @wp.kernel(module="unique", enable_backward=False,
               module_options=MODULE_OPTIONS)
    def keys_kernel(
        nacon: wp.array[int], world: wp.array[int], geom: wp.array[wp.vec2i],
        collision: wp.array[int], capacity: int,
        keys: wp.array[wp.uint64], permutation: wp.array[wp.int32],
        bad_flags: wp.array[wp.int32],
    ):
        tid = wp.tid()
        count = nacon[0]
        if tid == 0:
            if count < 0 or count > capacity:
                bad_flags[capacity] = bad_flags[capacity] | BAD_COUNT
        permutation[tid] = tid
        key = wp.uint64(UINT64_MAX)
        if tid < count:
            w = world[tid]
            pair = geom[tid]
            local_id = collision[tid]
            valid = (w >= 0 and w < world_count
                     and pair[0] >= 0 and pair[0] < geom_count
                     and pair[1] >= 0 and pair[1] < geom_count
                     and local_id >= 0 and local_id < collision_count)
            if valid:
                key = ((wp.uint64(w) << wp.uint64(world_shift))
                       | (wp.uint64(pair[0]) << wp.uint64(geom0_shift))
                       | (wp.uint64(pair[1]) << wp.uint64(geom1_shift))
                       | wp.uint64(local_id))
            else:
                bad_flags[tid] = bad_flags[tid] | BAD_IDENTITY
        keys[tid] = key

    return keys_kernel


@wp.kernel(enable_backward=False)
def _check_duplicates(keys: wp.array[wp.uint64], bad_flags: wp.array[wp.int32]):
    tid = wp.tid()
    if tid > 0:
        if keys[tid] != wp.uint64(UINT64_MAX) and keys[tid] == keys[tid - 1]:
            bad_flags[tid] = bad_flags[tid] | DUPLICATE_IDENTITY


@wp.kernel(enable_backward=False)
def _check_empty_count(nacon: wp.array[int], bad_flags: wp.array[wp.int32]):
    if nacon[0] != 0:
        bad_flags[0] = bad_flags[0] | BAD_COUNT


@lru_cache(maxsize=None)
def _gather_kernel(dtype, ndim: int):
    if ndim == 1:
        @wp.kernel(module="unique", enable_backward=False,
                   module_options=MODULE_OPTIONS)
        def gather(src: wp.array(dtype=dtype), old: wp.array[wp.int32],
                   dst: wp.array(dtype=dtype)):
            tid = wp.tid()
            dst[tid] = src[old[tid]]
    else:
        @wp.kernel(module="unique", enable_backward=False,
                   module_options=MODULE_OPTIONS)
        def gather(src: wp.array2d(dtype=dtype), old: wp.array[wp.int32],
                   dst: wp.array2d(dtype=dtype)):
            tid, column = wp.tid()
            dst[tid, column] = src[old[tid], column]
    return gather


class CanonicalContacts:
    """Capture-safe stable contact ordering for the installed rigid backend.

    ``bad_flags`` is a sticky device array: each thread owns one slot, so no
    atomic is required.  Call ``validate()`` outside graph capture whenever a
    synchronization is acceptable; it raises on overflow, malformed identities,
    or repeated contact keys.  ``clear_bad_flags()`` explicitly starts a fresh
    validation interval.
    """

    def __init__(self, m, d):
        if int(m.nflex) != 0:
            raise NotImplementedError("CanonicalContacts supports rigid models only")
        self.model = m
        self.data = d
        self.capacity = int(d.naconmax)
        self.world_count = int(d.nworld)
        self.geom_count = int(m.ngeom)
        self.collision_count = int(mujoco.mjMAXCONPAIR)
        if not 0 <= self.capacity <= (1 << 31) - 1:
            raise ValueError("Contact capacity must fit nonnegative int32 indices")
        if self.world_count <= 0 or self.geom_count < 0 or self.collision_count <= 0:
            raise ValueError("Invalid world, geometry, or per-pair contact count")
        self.world_bits = max(1, (self.world_count - 1).bit_length())
        self.geom_bits = max(1, (max(self.geom_count, 1) - 1).bit_length())
        self.collision_bits = max(1, (self.collision_count - 1).bit_length())
        self.key_bits = self.world_bits + 2 * self.geom_bits + self.collision_bits
        # Reserve one bit so no valid key can equal the invalid-tail sentinel.
        if self.key_bits > 63:
            raise ValueError(f"Contact identity needs {self.key_bits} bits; maximum is 63")
        actual_fields = tuple(d.contact.__dataclass_fields__)
        if set(actual_fields) != set(CONTACT_FIELDS):
            raise ValueError(f"Unexpected Contact schema: {actual_fields!r}")

        self.device = d.nacon.device
        self.keys = wp.empty(2 * self.capacity, dtype=wp.uint64, device=self.device)
        self.permutation = wp.empty(2 * self.capacity, dtype=wp.int32, device=self.device)
        self.bad_flags = wp.zeros(self.capacity + 1, dtype=wp.int32, device=self.device)
        self._fields = []
        for name in CONTACT_FIELDS:
            src = getattr(d.contact, name)
            if name in ("flex", "elem", "vert") and not src.size:
                continue  # Rigid-only backend omits these unused arrays.
            if src.device != self.device or src.shape[0] != self.capacity:
                raise ValueError(f"Contact.{name} has incompatible device or capacity")
            expected_ndim = 2 if name == "efc_address" else 1
            if src.ndim != expected_ndim or not src.is_contiguous:
                raise ValueError(f"Contact.{name} must be contiguous with ndim={expected_ndim}")
            scratch = wp.empty_like(src)
            self._fields.append((name, src, scratch, _gather_kernel(src.dtype, src.ndim)))
        self._keys_kernel = _key_kernel(
            self.world_count, self.geom_count, self.collision_count,
            self.geom_bits, self.collision_bits,
        )

    def sort(self) -> None:
        """Reorder the complete fixed-capacity Contact buffers in place."""
        with wp.ScopedDevice(self.device):
            if self.capacity == 0:
                wp.launch(_check_empty_count, dim=1,
                          inputs=[self.data.nacon, self.bad_flags])
                return
            contact = self.data.contact
            wp.launch(self._keys_kernel, dim=self.capacity, inputs=[
                self.data.nacon, contact.worldid, contact.geom,
                contact.geomcollisionid, self.capacity, self.keys,
                self.permutation, self.bad_flags,
            ])
            wp.utils.radix_sort_pairs(self.keys, self.permutation, self.capacity)
            wp.launch(_check_duplicates, dim=self.capacity,
                      inputs=[self.keys, self.bad_flags])
            for name, src, scratch, gather in self._fields:
                if getattr(contact, name) is not src:
                    raise RuntimeError(f"Contact.{name} allocation changed; recreate sorter and graphs")
                wp.launch(gather, dim=src.shape,
                          inputs=[src, self.permutation], outputs=[scratch])
                wp.copy(src, scratch)

    def clear_bad_flags(self) -> None:
        self.bad_flags.zero_()

    def validate(self) -> dict:
        """Synchronize and reject invalid ordering rather than silently continue."""
        flags = self.bad_flags.numpy()
        combined = 0
        slots = []
        for slot, value in enumerate(flags):
            if int(value):
                combined |= int(value)
                if len(slots) < 16:
                    slots.append((slot, int(value)))
        if combined:
            reasons = []
            for bit, label in ((BAD_COUNT, "contact count overflow"),
                               (BAD_IDENTITY, "invalid contact identity"),
                               (DUPLICATE_IDENTITY, "duplicate contact identity")):
                if combined & bit:
                    reasons.append(label)
            raise RuntimeError(f"Canonical contact validation failed: {', '.join(reasons)}; slots={slots}")
        return {"valid": True, "capacity": self.capacity, "key_bits": self.key_bits}
