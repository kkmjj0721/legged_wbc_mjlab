"""Measure actual feet/scan containment on nonterminal finite-patch states."""
import torch


class GeometryMonitor:
    def __init__(self, env, controller):
        self.env, self.controller = env, controller
        self.scan_crossings = self.foot_crossings = self.checked_states = 0
        self.max_relative_extent = 0.0
        self._site_ids = None

    def observe(self, terminal_mask):
        if self.controller.terrain_map.plane:
            return
        env = self.env
        valid = ~terminal_mask
        if not valid.any():
            return
        robot = env.scene["robot"]
        if self._site_ids is None:
            self._site_ids, _ = robot.find_sites(("FL", "FR", "RL", "RR"))
            if len(self._site_ids) != 4:
                raise ValueError("Expected four named foot sites for geometry validation")
        feet = robot.data.site_pos_w[:, self._site_ids, :2]
        scan = env.scene["terrain_scan"].data.hit_pos_w[..., :2]
        origins = env.scene.env_origins[:, None, :2]
        root = robot.data.root_link_pos_w[:, None, :2]
        half = torch.tensor(self.controller.cfg.geometry.patch_size_m, device=env.device) / 2
        scan_cross = ((scan - origins).abs() > half).any(dim=-1).any(dim=-1)
        foot_cross = ((feet - origins).abs() > half).any(dim=-1).any(dim=-1)
        self.scan_crossings += int((scan_cross & valid).sum())
        self.foot_crossings += int((foot_cross & valid).sum())
        self.checked_states += int(valid.sum())
        extent = max(float((scan[valid] - root[valid]).abs().max()), float((feet[valid] - root[valid]).abs().max()))
        self.max_relative_extent = max(self.max_relative_extent, extent)
        if ((scan_cross | foot_cross) & valid).any():
            raise RuntimeError(
                f"Nonterminal feet/scan crossed assigned patch: {self.summary()}. "
                f"Increase boundary_margin_m to at least {self.max_relative_extent + 0.1:.3f} and revalidate with a new protocol.")

    def summary(self):
        return dict(checked_states=self.checked_states, scan_crossings=self.scan_crossings,
                    foot_crossings=self.foot_crossings, max_relative_extent_m=self.max_relative_extent)
