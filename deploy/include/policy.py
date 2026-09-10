from __future__ import annotations

from pathlib import Path

import numpy as np


class OnnxPolicy:
    """ONNX actor adapter shared by simulation and a future real backend."""

    def __init__(self, path: Path | None, observation_dim: int, action_dim: int):
        self._session = None
        self._input_name = None
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        if path is None:
            return
        if not path.exists():
            raise FileNotFoundError(f"Policy not found: {path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required; use --no-policy for backend smoke tests") from exc
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or not outputs:
            raise ValueError("Expected one ONNX input and at least one output")
        self._input_name = inputs[0].name
        if inputs[0].shape[-1] not in (observation_dim, None, "None"):
            raise ValueError(f"Policy input shape {inputs[0].shape} does not end in {observation_dim}")
        if outputs[0].shape[-1] not in (action_dim, None, "None"):
            raise ValueError(f"Policy output shape {outputs[0].shape} does not end in {action_dim}")

    @property
    def available(self) -> bool:
        return self._session is not None

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if self._session is None:
            return np.zeros(self.action_dim, dtype=np.float64)
        observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        if observation.shape[1] != self.observation_dim:
            raise ValueError(f"Expected [1,{self.observation_dim}], got {observation.shape}")
        action = np.asarray(self._session.run(None, {self._input_name: observation})[0]).reshape(-1)
        if action.size != self.action_dim or not np.all(np.isfinite(action)):
            raise ValueError("Policy produced a non-finite or incorrectly sized action")
        return np.clip(action.astype(np.float64), -1.0, 1.0)
