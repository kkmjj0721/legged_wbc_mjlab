"""Go2 HIM runner with the MjLab ONNX export lifecycle."""

from src.tasks.locomotion.ri_4438_him.rl.runner import (
  HIMOnPolicyRunner as _Ri4438HIMOnPolicyRunner,
)


class HIMOnPolicyRunner(_Ri4438HIMOnPolicyRunner):
  """Task-local name for the shared modern HIM runner."""

