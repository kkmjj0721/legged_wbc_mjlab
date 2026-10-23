"""Preallocated sort workspaces and device-side validity checks."""
import numpy as np
import warp as wp
from mujoco_warp._src.types import BroadphaseType, IntegratorType, SensorType, TrnType
from .contacts import CanonicalContacts
from .constraints import CanonicalConstraints
from .sensor import validate_capacity

wp.set_module_options({"deterministic": wp.DeterministicMode.NOT_GUARANTEED})


@wp.kernel(enable_backward=False)
def _collect_status(contact: wp.array[int], efc: wp.array[int],
                    overflow: wp.array[int], status: wp.array[int]):
    i = wp.tid()
    if i < contact.shape[0] and contact[i] != 0:
        wp.atomic_or(status, 0, contact[i])
    if i < efc.shape[0] and efc[i] != 0:
        wp.atomic_or(status, 1, efc[i])
    if i < overflow.shape[0] and overflow[i] != 0:
        wp.atomic_or(status, 2, overflow[i])


def _validate_model(m, d):
    validate_capacity(d)
    # Larger/more general models exercise additional solver/actuator atomics.
    # Extend this list only alongside numerical replay/rollout evidence.
    if int(m.nv) > 32 or int(m.nv) < 1:
        raise NotImplementedError("Canonical GPU mode requires 1 <= nv <= 32")
    if int(m.opt.integrator) != int(IntegratorType.IMPLICITFAST):
        raise NotImplementedError("Canonical GPU mode currently requires implicitfast")
    if int(m.opt.broadphase) != int(BroadphaseType.NXN):
        raise NotImplementedError("Canonical GPU mode currently requires nxn broadphase")
    if int(m.opt.enableflags):
        raise NotImplementedError("Canonical GPU mode currently requires no extra MuJoCo enableflags")
    if np.any(m.body_gravcomp.numpy() != 0) or np.any(m.qLD_block_adr.numpy() < 0):
        raise NotImplementedError("Canonical GPU mode does not cover gravity compensation or sparse mass factorization")
    if np.any(m.opt.density.numpy() != 0) or np.any(m.opt.viscosity.numpy() != 0):
        raise NotImplementedError("Canonical GPU mode does not cover fluid forces")
    allowed = {int(getattr(SensorType, name)) for name in (
        "GYRO", "VELOCIMETER", "ACCELEROMETER", "SUBTREEANGMOM", "CONTACT")}
    if set(m.sensor_type.numpy().tolist()) - allowed:
        raise NotImplementedError("Canonical GPU mode only covers the RI-4438 IMU, momentum and contact sensors")
    transmissions = m.actuator_trntype.numpy()
    joints = m.actuator_trnid.numpy()[:, 0]
    if np.any(transmissions != int(TrnType.JOINT)) or len(set(joints.tolist())) != len(joints):
        raise NotImplementedError("Canonical GPU mode requires distinct joint transmissions")
    if any(value is not None for value in vars(m.callback).values()):
        raise NotImplementedError("Canonical GPU mode does not cover custom MuJoCo callbacks")


class CanonicalState:
    def __init__(self, m, d):
        _validate_model(m, d)
        self.data = d
        self.contacts = CanonicalContacts(m, d)
        self.constraints = CanonicalConstraints(m, d)
        self.status = wp.zeros(3, dtype=int, device=d.nacon.device)
        self.constraints.prepare()
        for _, _, _, kernel in self.contacts._fields:
            kernel.module.load(self.contacts.device)
        self.contacts._keys_kernel.module.load(self.contacts.device)
        # CUB may reserve temporary storage: warm it on owned scratch, without
        # changing contacts/constraints or advancing a single physics step.
        with wp.ScopedDevice(d.nacon.device):
            for sorter in (self.contacts, self.constraints):
                if sorter.capacity:
                    sorter.keys.zero_()
                    sorter.permutation.zero_()
                    wp.utils.radix_sort_pairs(sorter.keys, sorter.permutation, sorter.capacity)
            _collect_status.module.load(d.nacon.device)

    def collect(self):
        with wp.ScopedDevice(self.data.nacon.device):
            wp.launch(_collect_status, dim=max(self.contacts.bad_flags.size,
                      self.constraints.bad_flags.size, self.data.overflow.size),
                      inputs=[self.contacts.bad_flags, self.constraints.bad_flags,
                              self.data.overflow], outputs=[self.status])

    def validate(self):
        self.collect()
        flags = self.status.numpy()
        if flags[0]:
            self.contacts.validate()
        if flags[1]:
            self.constraints.validate()
        if flags[2]:
            raise RuntimeError(f"Canonical GPU mode rejected backend overflow (bitmask={int(flags[2])}); increase the indicated simulation buffer capacity")
