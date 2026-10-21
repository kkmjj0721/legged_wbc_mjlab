"""HIM lifecycle with explicit symmetry experiment and export contracts."""

from dataclasses import fields, is_dataclass
from enum import Enum
import inspect
import json
import os
from pathlib import Path
import tempfile

import onnx
import torch

from src.tasks.locomotion.ri_4438_him.rl.runner import HIMOnPolicyRunner
from .layouts import get_layout, validate_env_cfg
from .symmetry import MAPPING_VERSION, augment_phase47, get_adapter


def _plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if callable(value):
        item = value if inspect.isfunction(value) or inspect.isclass(value) else type(value)
        return f"{item.__module__}:{item.__qualname__}"
    if isinstance(value, slice):
        return [value.start, value.stop, value.step]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported experiment contract value: {type(value)}")


class HimSymmetryRunner(HIMOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.layout = get_layout(train_cfg["observation_layout"])
        validate_env_cfg(env.unwrapped.cfg, self.layout)
        symmetry_cfg = train_cfg["algorithm"].get("symmetry_cfg")
        reference_update = train_cfg["algorithm"].get("reference_update", False)
        reference_mode = reference_update or (symmetry_cfg and symmetry_cfg.get("implementation") == "homie")
        self._symmetry_contract = {
            "version": 2 if reference_mode else 1,
            "layout": _plain(self.layout.descriptor()),
            "mapping_version": MAPPING_VERSION,
            "symmetry": _plain(symmetry_cfg),
            "rewards": {name: {"weight": term.weight, "func": _plain(term.func),
                                "params": _plain(term.params)}
                        for name, term in env.unwrapped.cfg.rewards.items()},
        }
        if reference_mode:
            self._symmetry_contract["algorithm"] = {
                "reference_update": bool(reference_update),
                "fused_optimizer": bool(train_cfg["algorithm"].get("fused_optimizer", False)),
                "rollout_steps": train_cfg["num_steps_per_env"],
                "normalization": "real_rollout_only",
            }
        super().__init__(env, train_cfg, log_dir, device)
        actor = self.alg.get_policy()
        if (actor.num_one_step_obs != self.layout.frame_size
                or actor.history_size != self.layout.history_size
                or tuple(actor.action_observation_slice) != self.layout.action_slice
                or tuple(self.alg.estimator_velocity_slice) != self.layout.velocity_slice
                or self.alg.estimator_target_slices != self.layout.target_slices):
            raise ValueError("HIM model and supervision do not match observation_layout")
        if self.alg.symmetry is not None:
            if self.alg.symmetry.data_augmentation_func is not augment_phase47:
                raise ValueError("Symmetry augmentation function does not match observation_layout")
            get_adapter(env, self.layout)

    def save(self, path, infos=None):
        super().save(path, {**(infos or {}), "him_symmetry": self._symmetry_contract})

    def warm_start(self, path):
        """Initialize a fresh experiment from weights, without old optimizer/curriculum state."""
        if (self.current_learning_iteration != 0 or self.env.unwrapped.common_step_counter != 0
                or self.alg.optimizer.state
                or self.alg.estimator.optimizer.state):
            raise ValueError("Warm-start requires a fresh HIM symmetry runner")
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        infos = checkpoint.get("infos") or {}
        saved = infos.get("him_symmetry")
        if saved is not None and saved["layout"] != self._symmetry_contract["layout"]:
            raise ValueError("HIM observation layout differs from warm-start checkpoint")
        numerical = infos.get("him_numerics")
        if numerical is not None and numerical != self._numerical_contract():
            raise ValueError("Warm-start HIM numerical contract differs")
        # HIMPPO's estimator loader also restores its own optimizer when present.
        # Remove both states, and bypass parent load() which restores curricula.
        checkpoint.pop("optimizer_state_dict", None)
        checkpoint.pop("estimator_optimizer_state_dict", None)
        self.alg.load(checkpoint, {
            "actor": True, "critic": True, "estimator": True,
            "optimizer": False, "iteration": False,
        }, strict=True)
        self._validate_numerics()
        self.env.unwrapped.common_step_counter = 0
        with torch.inference_mode():
            self.env.reset()

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        # Check before loading any model/optimizer or resetting the environment.
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        infos = checkpoint.get("infos") or {}
        saved = infos.get("him_symmetry")
        if saved is not None and saved.get("symmetry") is not None:
            symmetry = saved["symmetry"]
            # Moving the implementation does not change the training contract.
            if symmetry.get("data_augmentation_func") == (
                "src.tasks.locomotion.ri_4438_symmetry_common.symmetry:augment_phase47"
            ):
                saved = {**saved, "symmetry": {
                    **symmetry, "data_augmentation_func": _plain(augment_phase47),
                }}
        full_resume = load_cfg is None or any(load_cfg.get(k, False) for k in ("optimizer", "iteration"))
        if full_resume and saved != self._symmetry_contract:
            raise ValueError("HIM symmetry experiment differs (algorithm, layout, mapping, rewards or switches); cannot resume")
        if saved is not None and saved["layout"] != self._symmetry_contract["layout"]:
            raise ValueError("HIM observation layout differs from checkpoint")
        numerical = infos.get("him_numerics")
        if numerical is not None and numerical != self._numerical_contract():
            raise ValueError("HIM checkpoint numerical contract differs from the current configuration")
        return super().load(path, load_cfg, strict, map_location)

    def export_policy_to_onnx(self, path, filename="policy.onnx", verbose=False):
        super().export_policy_to_onnx(path, filename, verbose)
        destination = Path(path) / filename
        model = onnx.load(destination)
        metadata = {entry.key: entry.value for entry in model.metadata_props}
        metadata.update({
            "him_observation_layout": self.layout.name,
            "him_observation_terms": json.dumps(self.layout.deployment_terms),
            "him_action_joint_names": json.dumps(list(self.env.unwrapped.action_manager.get_term("joint_pos").target_names)),
            "him_mapping_version": MAPPING_VERSION,
            "him_action_scale": json.dumps(torch.as_tensor(
                self.env.unwrapped.action_manager.get_term("joint_pos").scale
            ).expand(self.env.num_envs, 12)[0].cpu().tolist()),
            "him_action_offset": json.dumps(torch.as_tensor(
                self.env.unwrapped.action_manager.get_term("joint_pos").offset
            ).expand(self.env.num_envs, 12)[0].cpu().tolist()),
            "him_control_dt": str(self.env.unwrapped.step_dt),
            "him_phase_period": json.dumps(self.env.unwrapped.cfg.rewards["foot_gait"].params["period"]
                                           if self.layout.with_phase else None),
            "him_phase_command_threshold": json.dumps(
                self.env.unwrapped.cfg.rewards["foot_gait"].params["command_threshold"]
                if self.layout.with_phase else None),
        })
        onnx.helper.set_model_props(model, metadata)
        onnx.checker.check_model(model)
        descriptor, temporary = tempfile.mkstemp(prefix=".symmetry_", suffix=".onnx", dir=path)
        os.close(descriptor)
        try:
            onnx.save(model, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
