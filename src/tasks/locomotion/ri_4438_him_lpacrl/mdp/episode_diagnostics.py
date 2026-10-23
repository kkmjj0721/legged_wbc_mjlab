"""V7 physical crossings and command events, independent of reward and reset.

Support is measured from actual foot/terrain contacts (point, force and normal).
The contact point supplies the local surface height, including adjacent tiles.
"""

import torch
from mjlab.utils.lab_api.math import wrap_to_pi

from ..rl.deployment import DeploymentMetrics


class ConnectedTraversal:
    SURFACES = ("flat", "rough", "stairs", "slope", "mixed", "unsupported")
    def __init__(self, env, controller):
        self.env, self.controller = env, controller
        self.map = controller.terrain_map
        self.robot = env.scene["robot"]
        self.contact = env.scene["feet_support"]
        names = [name.rsplit("/", 1)[-1].removesuffix("_foot_collision")
                 for name in self.contact.primary_names]
        self.foot_ids, _ = self.robot.find_sites(names, preserve_order=True)
        n, device = env.num_envs, env.device
        self.birth_origins = torch.zeros(n, 3, device=device)
        self.birth_tiles = torch.zeros(n, dtype=torch.long, device=device)
        self.target_tiles = torch.zeros_like(self.birth_tiles)
        self.target_entered = torch.ones(n, dtype=torch.bool, device=device)
        self.body_tiles = torch.zeros_like(self.birth_tiles)
        self.foot_tiles = torch.zeros(n, 4, dtype=torch.long, device=device)
        self.support_tiles = torch.full_like(self.birth_tiles, -3)
        self.mixed_support = torch.zeros(n, dtype=torch.bool, device=device)
        self.passed = torch.zeros(n, dtype=torch.bool, device=device)
        self.first_pass_time = torch.full((n,), -1., device=device)
        self.confirm_seconds = torch.zeros(n, device=device)
        self.target_confirm_seconds = torch.zeros(n, device=device)
        count = len(self.map.tiles)
        self.visited = torch.zeros(n, count, dtype=torch.bool, device=device)
        self.traversed = torch.zeros_like(self.visited)
        self.support_seconds = torch.zeros(n, count + 3, device=device)
        self.surface_seconds = torch.zeros(n, len(self.SURFACES), device=device)
        self.support_surface = torch.full_like(self.birth_tiles, 5)
        self.exit_extent = torch.tensor([t["exit_half_extent"] for t in self.map.tiles], device=device)
        kinds, inner, outer = [], [], []
        for tile in self.map.tiles:
            family, p = tile["terrain"], tile["parameters"]
            extent = [s/2-p.get("border_width", 0) for s in self.map.size]
            if family.startswith("stairs"):
                steps = int(min((2*e-p["platform_width"])/(2*p["step_width"]) for e in extent))
                kinds.append(2); inner.append([e-steps*p["step_width"] for e in extent])
            elif family.startswith("slope") and p["gradient"] > 0:
                kinds.append(3); inner.append([p["platform_width"]/2]*2)
            elif family == "random_rough":
                kinds.append(1); inner.append([-1., -1.])
            else:
                kinds.append(0); inner.append([0., 0.])
            outer.append(extent)
        self.tile_kind = torch.tensor(kinds, device=device)
        self.tile_inner = torch.tensor(inner, device=device)
        self.tile_outer = torch.tensor(outer, device=device)
        self.reset(torch.arange(n, device=device))

    def reset(self, ids):
        self.birth_origins[ids] = self.env.scene.env_origins[ids]
        self.birth_tiles[ids] = self.controller.episode_geometry_id[ids]
        self.target_tiles[ids] = self.birth_tiles[ids]
        self.target_entered[ids] = True
        self.body_tiles[ids] = self.birth_tiles[ids]
        self.passed[ids] = False
        self.first_pass_time[ids] = -1
        self.confirm_seconds[ids] = 0
        self.target_confirm_seconds[ids] = 0
        self.visited[ids] = False
        self.traversed[ids] = False
        self.support_seconds[ids] = 0
        self.surface_seconds[ids] = 0

    def _outside(self, points, tiles):
        origins = self.map.origins.reshape(-1, 3)[tiles, :2]
        extent = self.exit_extent[tiles] + self.controller.cfg.geometry.pass_margin_m
        return ((points - origins[:, None, :]).abs() >= extent[:, None, :]).any(-1)

    def observe(self, active, failed):
        feet = self.robot.data.site_pos_w[:, self.foot_ids]
        data = self.contact.data
        points = data.pos.reshape(self.env.num_envs, 4, 3)
        normals = data.normal.reshape(self.env.num_envs, 4, 3)
        force = data.force.reshape(self.env.num_envs, 4, 3).norm(dim=-1)
        clearance = feet[..., 2] - points[..., 2]
        support = ((force > 1.0) & (normals[..., 2].abs() > .3)
                   & (clearance >= -.005) & (clearance <= .04)
                   & torch.isfinite(points).all(-1))
        self.body_tiles.copy_(self.map.locate(self.robot.data.root_link_pos_w[:, :2]))
        self.foot_tiles.copy_(self.map.locate(points[..., :2]))
        support &= self.foot_tiles >= -1
        low = torch.where(support, self.foot_tiles, len(self.map.tiles)).amin(-1)
        high = torch.where(support, self.foot_tiles, -2).amax(-1)
        self.mixed_support.copy_(support.any(-1) & (low != high))
        self.support_tiles.copy_(torch.where(support.any(-1), torch.where(self.mixed_support, -4, low), -3))
        tile = self.foot_tiles.clamp_min(0)
        local = (points[..., :2] - self.map.origins.reshape(-1, 3)[tile, :2]).abs()
        on_obstacle = (local >= self.tile_inner[tile]).any(-1) & (local < self.tile_outer[tile]).all(-1) & (self.foot_tiles >= 0)
        kind = torch.where(on_obstacle, self.tile_kind[tile], 0)
        low_kind = torch.where(support, kind, 5).amin(-1)
        high_kind = torch.where(support, kind, -1).amax(-1)
        self.support_surface.copy_(torch.where(~support.any(-1), 5,
                                               torch.where(self.mixed_support | (low_kind != high_kind), 4, low_kind)))
        # Exposure bins: physical tiles, frame, mixed, unsupported.
        exposure = torch.where(self.support_tiles >= 0, self.support_tiles,
                               torch.where(self.support_tiles == -1, len(self.map.tiles),
                                           torch.where(self.mixed_support, len(self.map.tiles)+1, len(self.map.tiles)+2)))
        ids = active.nonzero(as_tuple=False).flatten()
        self.support_seconds[ids, exposure[ids]] += self.env.step_dt
        self.surface_seconds[ids, self.support_surface[ids]] += self.env.step_dt
        in_tile = active & (self.body_tiles >= 0)
        rows = in_tile.nonzero(as_tuple=False).flatten()
        self.visited[rows, self.body_tiles[rows]] = True

        def confirmation(tiles, clock, eligible=None):
            outside = self._outside(feet[..., :2], tiles).all(-1)
            grounded_outside = (support & self._outside(points[..., :2], tiles)).sum(-1) >= 2
            evidence = outside & grounded_outside & ~failed
            if eligible is not None:
                evidence &= eligible
            clock[active] = torch.where(evidence[active], clock[active] + self.env.step_dt, 0.)
            return active & (clock >= self.controller.cfg.geometry.pass_confirm_s - 1e-6)

        birth_pass = confirmation(self.birth_tiles, self.confirm_seconds)
        first = birth_pass & ~self.passed
        self.first_pass_time[first] = self.env.episode_length_buf[first].float() * self.env.step_dt
        self.passed |= birth_pass
        self.target_entered |= active & ~self._outside(feet[..., :2], self.target_tiles).any(-1)
        passed = confirmation(self.target_tiles, self.target_confirm_seconds, self.target_entered)
        rows = passed.nonzero(as_tuple=False).flatten()
        self.traversed[rows, self.target_tiles[rows]] = True
        # Move diagnostic references only after leaving the previous target;
        # the immutable birth task/origin and its latch remain untouched.
        move = passed & (self.body_tiles >= 0) & (self.body_tiles != self.target_tiles)
        self.target_tiles[move] = self.body_tiles[move]
        self.target_confirm_seconds[move] = 0
        self.target_entered[move] = False

    def trial(self, slot):
        tile = self.map.tiles[int(self.birth_tiles[slot])]
        obstacle = not (tile["terrain"].startswith("slope") and tile["parameters"].get("gradient", 0) == 0)
        return {"traversal_version": 7, "birth_tile": int(self.birth_tiles[slot]),
                "terrain_passed": bool(self.passed[slot]),
                "first_pass_time": float(self.first_pass_time[slot]) if self.passed[slot] else None,
                "obstacle_eligible": obstacle,
                "body_tile": int(self.body_tiles[slot]),
                "support_tile": int(self.support_tiles[slot]),
                "support_surface": self.SURFACES[int(self.support_surface[slot])],
                "mixed_support": bool(self.mixed_support[slot]),
                "visited_tiles": self.visited[slot].nonzero().flatten().tolist(),
                "traversed_tiles": self.traversed[slot].nonzero().flatten().tolist(),
                "support_seconds_by_tile": self.support_seconds[slot].tolist(),
                "support_seconds_by_surface": dict(zip(self.SURFACES, self.surface_seconds[slot].tolist()))}


class EpisodeDiagnostics:
    """Training counters collected once per real step, before auto reset."""

    NAMES = ("environment_steps", "episodes", "failures", "map_truncations", "zero_switch_episodes",
             "resamplings", "significant_switches", "switch_passed", "switch_failed", "switch_unobserved",
             "terrain_passed", "passed_without_failure", "strict_process_passed")

    def __init__(self, env, controller):
        self.env, self.controller = env, controller
        self.deployment = DeploymentMetrics(env.num_envs, env.device, env.step_dt, controller.cfg.evaluation)
        self.traversal = ConnectedTraversal(env, controller) if controller.terrain_map.connected else None
        self.mode_names = controller.family_names if controller.compact else ("fixed", "random_process", "coverage")
        mode_count = len(self.mode_names)
        if controller.compact:
            self.task_families = torch.tensor(
                [self.mode_names.index(name) for name in controller.catalog.task_families], device=env.device)
        # fixed, random process, coverage; explicit totals survive PPO update boundaries.
        self.totals = torch.zeros(mode_count, len(self.NAMES), dtype=torch.long, device=env.device)
        self.exposure = torch.zeros(mode_count, len(controller.terrain_map.tiles)+3, device=env.device)
        self.switch_exposure = torch.zeros_like(self.exposure, dtype=torch.long)
        self.failure_regions = torch.zeros_like(self.exposure, dtype=torch.long)
        self.surface_steps = torch.zeros(mode_count, 6, dtype=torch.long, device=env.device)
        self.complete_switch_surfaces = torch.zeros_like(self.surface_steps)
        self._synced = {name: torch.zeros_like(getattr(self, name)) for name in
                        ("totals", "exposure", "switch_exposure", "failure_regions", "surface_steps", "complete_switch_surfaces")}
        behavior = getattr(controller.cfg, "behavior_monitor", None)
        self.monitor_enabled = behavior is not None and behavior.enabled
        if self.monitor_enabled:
            self.behavior_names = ("linear_active_steps", "yaw_active_steps", "grace_steps",
                                   "linear_stall_steps", "yaw_stall_steps", "penalty_steps", "stall_terminations")
            self.behavior_totals = torch.zeros(mode_count, len(self.behavior_names), dtype=torch.long, device=env.device)
            self.behavior_costs = torch.zeros(mode_count, dtype=torch.float64, device=env.device)
            for name in ("behavior_totals", "behavior_costs"):
                self._synced[name] = torch.zeros_like(getattr(self, name))
        if controller._saved_diagnostics is not None:
            for name in self._synced:
                value = torch.as_tensor(controller._saved_diagnostics[name], device=env.device,
                                        dtype=getattr(self, name).dtype)
                if value.shape != getattr(self, name).shape or not torch.isfinite(value).all() or (value < 0).any():
                    raise ValueError("Invalid saved interaction statistics")
                getattr(self, name).copy_(value)
                self._synced[name].copy_(value)

    def synchronize(self):
        for name, previous in self._synced.items():
            value = getattr(self, name)
            delta = value - previous
            torch.distributed.all_reduce(delta)
            value.copy_(previous + delta)
            previous.copy_(value)

    def state_dict(self):
        return {name: getattr(self, name).cpu().tolist() for name in self._synced}

    def modes(self):
        c = self.controller
        if c.compact:
            return self.task_families[c.episode_task_id.clamp_min(0)]
        return torch.where(c.episode_sampling_mode == 1, 2, c.process_tasks[c.episode_task_id].long())

    def reset(self, ids):
        self.deployment.reset(ids)
        if self.traversal is not None:
            self.traversal.reset(ids)

    def observe(self):
        env, c = self.env, self.controller
        active = c.episode_task_id >= 0
        failed = env.termination_manager.terminated
        modes = self.modes()
        self.totals[:, 0] += torch.bincount(modes[active], minlength=len(self.mode_names))
        if self.monitor_enabled:
            from .behavior_monitor import get_behavior_monitor

            monitor = get_behavior_monitor(env).ensure_step()
            values = torch.stack((c.commands[:, :2].norm(dim=-1) >= monitor.linear_command_threshold,
                                  c.commands[:, 2].abs() >= monitor.angular_command_threshold,
                                  monitor.grace_active,
                                  monitor.linear_stalled, monitor.yaw_stalled,
                                  monitor.cost > 0, monitor.terminated), -1).long()
            self.behavior_totals.index_add_(0, modes[active], values[active])
            self.behavior_costs.index_add_(0, modes[active], monitor.cost[active].double())
        support = None
        if self.traversal is not None:
            self.traversal.observe(active, failed)
            support = self.traversal.support_tiles
            count = len(c.terrain_map.tiles)
            region = torch.where(support >= 0, support,
                                 torch.where(support == -1, count, torch.where(support == -4, count+1, count+2)))
            flat = modes * (count+3) + region
            self.exposure.view(-1).index_add_(0, flat[active], torch.full_like(flat[active], env.step_dt, dtype=self.exposure.dtype))
            significant = c.command_event & (c.command_event_delta.abs() >= self.deployment.change_thresholds).any(-1)
            selected = active & significant
            self.switch_exposure.view(-1).index_add_(0, flat[selected], torch.ones_like(flat[selected]))
            selected = active & failed
            self.failure_regions.view(-1).index_add_(0, flat[selected], torch.ones_like(flat[selected]))
            surface = modes * 6 + self.traversal.support_surface
            self.surface_steps.view(-1).index_add_(0, surface[active], torch.ones_like(surface[active]))
        robot = env.scene["robot"]
        actual = torch.cat((robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]), -1)
        self.deployment.observe(actual, c.commands, active, failed, event=c.command_event,
                                event_delta=c.command_event_delta, heading_delta=c.command_heading_delta,
                                support_tile=support,
                                support_surface=self.traversal.support_surface if self.traversal is not None else None,
                                heading_error=wrap_to_pi(c.heading_targets - robot.data.heading_w) if c.catalog.heading_command else None)

    def finish(self, ids):
        d, c, env = self.deployment, self.controller, self.env
        d.finish(ids)
        failed = env.termination_manager.terminated[ids]
        boundary_name = "boundary_truncation" if c.compact else "map_boundary_truncation"
        boundary = (env.termination_manager.get_term(boundary_name)[ids] & ~failed
                    if boundary_name in env.termination_manager.active_terms else torch.zeros_like(failed))
        passed = self.traversal.passed[ids] if self.traversal is not None else torch.zeros_like(failed)
        full = (env.episode_length_buf[ids] >= env.max_episode_length) & ~failed & ~boundary
        tracking = (d.trial_error_sum[ids] / d.trial_steps[ids, None].clamp_min(1) < 1).all(-1)
        strict = (full & tracking & (d.max_stall[ids] * env.step_dt < c.cfg.evaluation.deployment_stall_s)
                  & (d.transition_count[ids] > 0) & (d.transition_passed[ids] == d.transition_count[ids]))
        values = torch.stack((torch.zeros_like(ids), torch.ones_like(ids), failed, boundary,
                              d.resampling_count[ids] == 0, d.resampling_count[ids], d.transition_count[ids],
                              d.transition_passed[ids], d.transition_failed[ids], d.transition_unobserved[ids],
                              passed, passed & full, strict & c.scheduled_tasks[c.episode_task_id[ids]]), -1).long()
        self.totals.index_add_(0, self.modes()[ids], values)
        if self.traversal is not None:
            events = d.event_data[ids]
            complete = (events[..., 9] > 0) & (events[..., 11] >= d.check_steps)
            surface = self.modes()[ids, None] * 6 + events[..., 13].long()
            self.complete_switch_surfaces.view(-1).index_add_(0, surface[complete], torch.ones_like(surface[complete]))

    def summary(self):
        result = {}
        for i, mode in enumerate(self.mode_names):
            result.update({f"{mode}/{name}": int(self.totals[i, j]) for j, name in enumerate(self.NAMES)
                           if not (self.controller.compact and self.traversal is None
                                   and name in {"terrain_passed", "passed_without_failure"})})
            steps = max(int(self.totals[i, 0]), 1)
            # The compact isolated map has no connected-tile contact survey.
            # Do not report absent measurements as zero exposure or zero passes.
            if not self.controller.compact or self.traversal is not None:
                result[f"{mode}/outer_flat_fraction"] = float(self.exposure[i, len(self.controller.terrain_map.tiles)]) / (steps*self.env.step_dt)
                for surface, name in enumerate(ConnectedTraversal.SURFACES):
                    result[f"{mode}/support_{name}_steps"] = int(self.surface_steps[i, surface])
                    result[f"{mode}/complete_{name}_switches_per_million_steps"] = int(self.complete_switch_surfaces[i, surface]) * 1e6 / steps
            if self.monitor_enabled:
                result.update({f"{mode}/behavior/{name}": int(self.behavior_totals[i, j])
                               for j, name in enumerate(self.behavior_names)})
                result[f"{mode}/behavior/cost_mean"] = float(self.behavior_costs[i]) / steps
            if self.controller.compact:
                result[f"{mode}/allocated_environments"] = int(self.controller.family_counts[i])
                result[f"{mode}/step_fraction"] = int(self.totals[i, 0]) / max(int(self.totals[:, 0].sum()), 1)
        return result


def observe_episode(env, env_ids=None):
    from .curriculums import get_controller

    c = get_controller(env)
    if c.episode_metrics is None:
        # PPO collects in inference_mode, but checkpoint/reset runs outside it.
        # Persistent diagnostic buffers must remain ordinary mutable tensors.
        with torch.inference_mode(False):
            c.episode_metrics = EpisodeDiagnostics(env, c)
    c.episode_metrics.observe()
