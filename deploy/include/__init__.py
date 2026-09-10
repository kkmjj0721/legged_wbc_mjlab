"""Reusable deployment components.

The package is deliberately backend-agnostic at the controller boundary:
``fsm`` consumes a ``RobotBackend`` and ``inputs`` produces commands.  A real
robot backend can therefore replace ``mujoco_bridge`` without changing the
task configuration or state machine.
"""
