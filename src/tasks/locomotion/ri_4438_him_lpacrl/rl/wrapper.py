"""Proxy an already initialized vector wrapper; never reset it a second time."""
from ..mdp.curriculums import get_controller


class TimeoutMaskWrapper:
    def __init__(self, env):
        self.env = env
        self._geometry_monitor = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def step(self, actions):
        controller = get_controller(self.unwrapped)
        if controller.mode == "train":
            controller.note_step()
        obs, reward, done, extras = self.env.step(actions)
        if not controller.terrain_map.plane:
            from ..mdp.geometry_checks import GeometryMonitor
            if self._geometry_monitor is None:
                self._geometry_monitor = GeometryMonitor(self.unwrapped, controller)
            self._geometry_monitor.observe(done.bool())
            extras.setdefault("log", {}).update({"LPACRL/geometry/"+k: v for k, v in self._geometry_monitor.summary().items()})
        if "time_outs" in extras:
            extras = dict(extras)
            extras["time_outs"] = self.unwrapped.reset_time_outs & ~self.unwrapped.reset_terminated
        return obs, reward, done, extras

    def reset(self):
        with get_controller(self.unwrapped).discard_episodes():
            return self.env.reset()
