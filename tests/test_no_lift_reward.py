"""No-lift reward audit; production files and checkpoints are never changed.

CPU counterexamples and regression checks:
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest tests.test_no_lift_reward -v

Frozen-checkpoint rollout with paired reward managers on the SAME states:
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m tests.test_no_lift_reward \
    --checkpoint /path/to/model_4500.pt --seconds 12 --num-envs 24 \
    --candidate phase --forward-speed 0.6 --clearance-weight -0.25

The rollout prints JSON to stdout and writes no reports or model files. It tests
reward arithmetic and activation, NOT the outcome of retraining a policy.

Audit findings (2026-09-30, 2026-09-29_21-37-06/model_4500.pt):
* The contact-timeout candidate is REJECTED as a complete fix: the frozen actor
  can remain at the first stair while brief contact breaks keep the cost zero.
* The phase candidate was checked at weight -0.5 on 5 terrains x 3 seeds x 24
  episodes, up to 12 seconds each. All 20 other reward terms matched bitwise
  between the paired managers. Resets/failures are excluded from episode stats.
* Weight -0.25 is the conservative candidate: under the current diagonal gait,
  its weighted rate is bounded in [-0.5, 0]. The existing checkpoint saved
  clearance=-0.25 and pose=0.65; current config uses -0.5 and 0.5 respectively.
* These are nominal-physics counterfactual scores on a FROZEN policy. They do
  not establish improved stair success, unchanged gait after learning, or an
  optimal weight. A/B retraining is still needed to measure those outcomes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from src.tasks.locomotion.ri_4438_him.mdp import rewards as live


def feet_gait_contact_timeout(
    env, period, offset, threshold, command_threshold, command_name, sensor_name
):
    """Rejected standalone candidate, retained to reproduce its counterexample.

    At weight 0.5, the added cost lies in [-0.5, 0] reward-rate units.
    Support <= period is unchanged; cost ramps to its cap over another period.
    This detects prolonged contact, not swing height or forward progress.
    """
    sensor = env.scene[sensor_name]
    contact = sensor.data.current_contact_time > 0
    period_steps = int(round(period / env.step_dt))
    phase = ((env.episode_length_buf % period_steps) / period_steps).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=phase.dtype).view(1, -1)
    desired = ((phase + offsets) % 1.0) < threshold
    reward = (desired == contact).float().mean(dim=1)

    overdue = ((sensor.data.current_contact_time - period) / period).clamp(0.0, 1.0)
    reward = reward - overdue.mean(dim=1)

    command = env.command_manager.get_command(command_name)
    if command is not None:
        active = (torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold).float()
        reward = reward * active
    return reward


def phase_clearance_cost(height, leg_phase, height_range=(0.10, 0.16), threshold=0.56):
    """Site-center clearance, smooth lower bound, bounded cost per swing foot."""
    minimum, maximum = height_range
    swing = leg_phase >= threshold
    swing_progress = ((leg_phase - threshold) / (1.0 - threshold)).clamp(0, 1)
    required_height = minimum * torch.sin(torch.pi * swing_progress)
    error = (required_height - height).clamp_min(0) + (height - maximum).clamp_min(0)
    return ((error / minimum).clamp(max=1.0) * swing).sum(dim=1)


def feet_clearance_phase(
    env, height_range, period, offset, threshold, command_name="twist",
    command_threshold=0.1, height_sensor_name="feet_terrain_height",
):
    """Test-only replacement: expected swing phase x terrain-relative height.

    Both phase offsets and sensor frames must be FL, FR, RL, RR. No contact mask
    or foot-velocity factor; those would let a stationary foot escape the cost.
    Height is site-center height, consistent with the current height_range.
    """
    period_steps = int(round(period / env.step_dt))
    phase = ((env.episode_length_buf % period_steps) / period_steps).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=phase.dtype).view(1, -1)
    leg_phase = (phase + offsets) % 1.0
    height = env.scene[height_sensor_name].data.heights
    cost = phase_clearance_cost(height, leg_phase, height_range, threshold)
    command = env.command_manager.get_command(command_name)
    active = torch.linalg.norm(command[:, :2], dim=1) + command[:, 2].abs() > command_threshold
    return cost * active


GAIT = dict(period=0.6, offset=[0.0, 0.5, 0.5, 0.0], threshold=0.56,
            command_threshold=0.1, command_name="twist", sensor_name="feet_ground_contact")
ASSET = SimpleNamespace(name="robot", site_ids=slice(None))
CLEARANCE = dict(height_range=(0.10, 0.16), command_name="twist",
                 command_threshold=0.1, asset_cfg=ASSET)


def fake_env(n=30):
    command = torch.tensor([[0.6, 0.0, 0.0]]).repeat(n, 1)
    robot = SimpleNamespace(data=SimpleNamespace(
        site_pos_w=torch.zeros(n, 4, 3), site_lin_vel_w=torch.zeros(n, 4, 3),
        root_link_lin_vel_b=torch.zeros(n, 3), root_link_ang_vel_b=torch.zeros(n, 3)))
    robot.data.site_pos_w[..., 2] = 0.03
    sensor = SimpleNamespace(data=SimpleNamespace(current_contact_time=torch.full((n, 4), 2.0)))
    height_sensor = SimpleNamespace(data=SimpleNamespace(heights=torch.full((n, 4), 0.01573)))
    return SimpleNamespace(device="cpu", step_dt=0.02, num_envs=n,
                           episode_length_buf=torch.arange(n), command=command,
                           command_manager=SimpleNamespace(get_command=lambda _: command),
                           scene={"robot": robot, "feet_ground_contact": sensor,
                                  "feet_terrain_height": height_sensor})


def desired_contacts(env):
    phase = (env.episode_length_buf % 30).float().unsqueeze(1) / 30
    return ((phase + torch.tensor(GAIT["offset"])) % 1) < GAIT["threshold"]


class ExistingLoopholes(unittest.TestCase):
    def test_stationary_low_feet_have_zero_clearance_cost(self):
        env = fake_env()
        self.assertTrue(torch.equal(live.feet_clearance(env, **CLEARANCE), torch.zeros(30)))

    def test_all_contact_collects_over_half_the_max_gait_reward(self):
        env = fake_env()
        value = live.feet_gait(env, **GAIT)
        self.assertAlmostEqual(value.mean().item() * 0.5, 17 / 60, places=6)
        self.assertEqual(set(value.tolist()), {0.5, 1.0})

    def test_speed_floor_charges_legitimate_stationary_stance(self):
        env = fake_env()
        stance = desired_contacts(env)
        z = env.scene["robot"].data.site_pos_w[..., 2]
        z[:] = torch.where(stance, 0.03, 0.13)
        v = env.scene["robot"].data.site_lin_vel_w[..., 0]
        v[:] = torch.where(stance, 0.0, 1.0)
        self.assertTrue(torch.equal(live.feet_clearance(env, **CLEARANCE), torch.zeros(30)))
        floored = ((0.10 - z).clamp_min(0) + (z - 0.16).clamp_min(0)) * v.clamp_min(0.2)
        self.assertTrue((floored.sum(1) > 0).all())

    def test_measured_swing_gate_still_lets_all_contact_escape(self):
        env = fake_env()
        swing = env.scene["feet_ground_contact"].data.current_contact_time == 0
        cost = ((0.1 - env.scene["robot"].data.site_pos_w[..., 2]).clamp_min(0) * swing).sum(1)
        self.assertTrue(torch.equal(cost, torch.zeros(30)))

    def test_world_height_is_not_terrain_clearance(self):
        env = fake_env()
        env.scene["robot"].data.site_pos_w[..., 2] = 0.13
        env.scene["robot"].data.site_lin_vel_w[..., 0] = 1.0
        flat_cost = live.feet_clearance(env, **CLEARANCE)
        env.scene["robot"].data.site_pos_w[..., 2] += 0.30
        elevated_cost = live.feet_clearance(env, **CLEARANCE)
        self.assertTrue((elevated_cost > flat_cost).all())

    def test_centering_gait_does_not_change_equal_horizon_ranking(self):
        env = fake_env()
        stuck = live.feet_gait(env, **GAIT)
        env.scene["feet_ground_contact"].data.current_contact_time[:] = desired_contacts(env) * 0.30
        walking = live.feet_gait(env, **GAIT)
        torch.testing.assert_close(walking - stuck, (walking - 0.5) - (stuck - 0.5))

    def test_standing_tracking_is_not_full_linear_tracking(self):
        env = fake_env()
        linear = live.track_linear_velocity(env, 0.5, "twist") * 3
        angular = live.track_angular_velocity(env, 0.5, "twist") * 2
        self.assertAlmostEqual(linear[0].item(), 3 * math.exp(-0.6**2 / 0.25), places=6)
        self.assertEqual(angular[0].item(), 2.0)


class TimeoutCandidate(unittest.TestCase):
    def test_normal_trot_is_bitwise_unchanged_for_entire_period(self):
        env = fake_env()
        env.scene["feet_ground_contact"].data.current_contact_time[:] = desired_contacts(env) * 0.34
        self.assertTrue(torch.equal(live.feet_gait(env, **GAIT), feet_gait_contact_timeout(env, **GAIT)))

    def test_extended_support_up_to_a_full_period_is_unchanged(self):
        env = fake_env(5)
        env.scene["feet_ground_contact"].data.current_contact_time[:] = torch.tensor([0, 0.34, 0.50, 0.59, 0.60])[:, None]
        self.assertTrue(torch.equal(live.feet_gait(env, **GAIT), feet_gait_contact_timeout(env, **GAIT)))

    def test_timeout_is_ramped_and_bounded(self):
        env = fake_env(6)
        env.scene["feet_ground_contact"].data.current_contact_time[:] = torch.tensor([0.6, 0.75, 0.9, 1.2, 10, 1e6])[:, None]
        difference = live.feet_gait(env, **GAIT) - feet_gait_contact_timeout(env, **GAIT)
        torch.testing.assert_close(difference, torch.tensor([0, 0.25, 0.5, 1, 1, 1.]))

    def test_stalled_gait_becomes_negative_without_foot_motion(self):
        env = fake_env()
        self.assertAlmostEqual(feet_gait_contact_timeout(env, **GAIT).mean().item() * 0.5, -13 / 60, places=6)

    def test_each_stuck_foot_counts_even_if_others_move(self):
        env = fake_env(4)
        env.scene["feet_ground_contact"].data.current_contact_time[:] = torch.eye(4) * 2
        difference = live.feet_gait(env, **GAIT) - feet_gait_contact_timeout(env, **GAIT)
        torch.testing.assert_close(difference * 0.5, torch.full((4,), 0.125))

    def test_standing_turning_and_command_threshold_match_existing_gate(self):
        env = fake_env(7)
        env.command[:] = torch.tensor([[0, 0, 0], [0.099, 0, 0], [0.1, 0, 0],
                                      [0.1001, 0, 0], [0, 0, 0.5], [0.06, 0, 0.05], [-0.6, 0, 0]])
        old, new = live.feet_gait(env, **GAIT), feet_gait_contact_timeout(env, **GAIT)
        torch.testing.assert_close(old - new, torch.tensor([0, 0, 0, 1, 1, 1, 1.]))
        self.assertTrue(torch.equal(new[:3], torch.zeros(3)))

    def test_height_and_foot_velocity_do_not_change_timeout(self):
        env = fake_env()
        initial = feet_gait_contact_timeout(env, **GAIT)
        env.scene["robot"].data.site_pos_w += 100
        env.scene["robot"].data.site_lin_vel_w += 10
        self.assertTrue(torch.equal(initial, feet_gait_contact_timeout(env, **GAIT)))

    def test_no_new_state_to_leak_across_reset(self):
        env = fake_env()
        feet_gait_contact_timeout(env, **GAIT)
        env.scene["feet_ground_contact"].data.current_contact_time.zero_()
        self.assertTrue(torch.equal(live.feet_gait(env, **GAIT), feet_gait_contact_timeout(env, **GAIT)))

    def test_candidate_does_not_mutate_shared_inputs(self):
        env = fake_env()
        contact = env.scene["feet_ground_contact"].data.current_contact_time.clone()
        command, steps = env.command.clone(), env.episode_length_buf.clone()
        feet_gait_contact_timeout(env, **GAIT)
        self.assertTrue(torch.equal(contact, env.scene["feet_ground_contact"].data.current_contact_time))
        self.assertTrue(torch.equal(command, env.command))
        self.assertTrue(torch.equal(steps, env.episode_length_buf))

    def test_documented_limit_tiny_contact_break_can_clear_timer(self):
        # This test intentionally records the remaining loophole, not success:
        # contact duration alone cannot enforce 10 cm of swing clearance.
        env = fake_env()
        env.scene["feet_ground_contact"].data.current_contact_time[:] = desired_contacts(env) * 0.02
        self.assertTrue(torch.equal(live.feet_gait(env, **GAIT), feet_gait_contact_timeout(env, **GAIT)))


PHASE_PARAMS = {key: value for key, value in GAIT.items() if key != "sensor_name"}
PHASE_PARAMS["height_range"] = (0.10, 0.16)


class PhaseClearanceCandidate(unittest.TestCase):
    def test_stationary_feet_are_charged_every_scheduled_swing(self):
        env = fake_env()
        cost = feet_clearance_phase(env, **PHASE_PARAMS)
        self.assertGreater(cost.mean().item(), 0.7)
        self.assertLessEqual(cost.max().item(), 2.0)

    def test_contact_flicker_cannot_clear_cost(self):
        env = fake_env()
        before = feet_clearance_phase(env, **PHASE_PARAMS)
        env.scene["feet_ground_contact"].data.current_contact_time.zero_()
        self.assertTrue(torch.equal(before, feet_clearance_phase(env, **PHASE_PARAMS)))

    def test_foot_velocity_cannot_clear_cost(self):
        env = fake_env()
        before = feet_clearance_phase(env, **PHASE_PARAMS)
        env.scene["robot"].data.site_lin_vel_w.normal_()
        self.assertTrue(torch.equal(before, feet_clearance_phase(env, **PHASE_PARAMS)))

    def test_normal_swing_profile_and_stationary_stance_are_free(self):
        env = fake_env()
        phases = ((env.episode_length_buf[:, None] % 30) / 30 + torch.tensor(GAIT["offset"])) % 1
        s = ((phases - 0.56) / 0.44).clamp(0, 1)
        h = env.scene["feet_terrain_height"].data.heights
        h[:] = torch.maximum(torch.full_like(h, 0.01573), 0.10 * torch.sin(torch.pi * s) + 1e-6)
        self.assertTrue(torch.equal(feet_clearance_phase(env, **PHASE_PARAMS), torch.zeros(30)))

    def test_liftoff_and_touchdown_lower_bound_tends_to_zero(self):
        eps = 1e-6
        phase = torch.tensor([[0.56, 0.56 + eps, 1 - eps, 0.0]])
        cost = phase_clearance_cost(torch.zeros(1, 4), phase)
        self.assertLess(cost.item(), 2e-5)
        # A fixed lower bound during all of swing would instead charge ~2.

    def test_support_is_not_charged_even_on_a_high_step(self):
        phase = torch.full((1, 4), 0.3)
        self.assertEqual(phase_clearance_cost(torch.full((1, 4), 0.3), phase).item(), 0.0)

    def test_higher_world_terrain_does_not_change_cost(self):
        env = fake_env()
        before = feet_clearance_phase(env, **PHASE_PARAMS)
        env.scene["robot"].data.site_pos_w[..., 2] += 1.0
        self.assertTrue(torch.equal(before, feet_clearance_phase(env, **PHASE_PARAMS)))

    def test_stand_is_free_but_turning_activates_cost(self):
        env = fake_env(2)
        env.episode_length_buf[:] = 8
        env.command[:] = torch.tensor([[0, 0, 0], [0, 0, 0.5]])
        values = feet_clearance_phase(env, **PHASE_PARAMS)
        self.assertEqual(values[0].item(), 0.0)
        self.assertGreater(values[1].item(), 0.0)

    def test_excessive_lift_still_has_a_bounded_cost(self):
        phase = torch.tensor([[0.78, 0.28, 0.28, 0.78]])
        self.assertGreater(phase_clearance_cost(torch.full((1, 4), 0.2), phase).item(), 0.0)
        self.assertEqual(phase_clearance_cost(torch.full((1, 4), 100.0), phase).item(), 2.0)

    def test_at_mid_swing_actual_lift_is_preferred_to_shuffling(self):
        phase = torch.tensor([[0.78, 0.28, 0.28, 0.78]])
        low = phase_clearance_cost(torch.full((1, 4), 0.01573), phase)
        lifted = phase_clearance_cost(torch.tensor([[0.10, 0.01573, 0.01573, 0.10]]), phase)
        self.assertGreater(low.item(), 1.6)
        self.assertEqual(lifted.item(), 0.0)

    def test_configured_trot_cost_is_bounded_at_both_relevant_weights(self):
        env = fake_env()
        env.scene["feet_terrain_height"].data.heights[:] = -10
        cost = feet_clearance_phase(env, **PHASE_PARAMS)
        # threshold > 0.5 with the configured diagonal offsets: at most 2 swings.
        self.assertLessEqual(cost.max().item(), 2.0)
        for weight in (-0.25, -0.5):
            self.assertGreaterEqual((weight * cost).min().item(), 2 * weight)

    def test_height_matches_gait_config_and_sensor_order(self):
        from src.tasks.locomotion.ri_4438_him.config.env_cfgs import ri_4438_rough_env_cfg
        cfg = ri_4438_rough_env_cfg()
        self.assertEqual(cfg.rewards["foot_gait"].params, GAIT)
        self.assertEqual(cfg.rewards["foot_clearance"].params["height_range"], PHASE_PARAMS["height_range"])
        self.assertEqual(cfg.rewards["foot_clearance"].params["asset_cfg"].site_names, ("FL", "FR", "RL", "RR"))


def audit_checkpoint(args):
    """Compare complete reward vectors with a frozen policy, no training/writes."""
    import numpy as np
    import yaml
    from tensordict import TensorDict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardManager
    from mjlab.tasks.registry import load_env_cfg
    from src.utils.training_logs import ConfigLoader
    from src.utils.training_report import load_checkpoint_actor
    from src.utils.policy_evaluation import EvalOptions, make_env_cfg
    from src.utils.evaluation_metrics import COMMANDS
    from src.utils.evaluation_scenarios import STAIR_COMMANDS

    torch.set_num_threads(4)
    checkpoint = Path(args.checkpoint).resolve()
    config_dir = checkpoint.parent / "params"
    run = SimpleNamespace(path=checkpoint.parent,
                          env=yaml.load((config_dir / "env.yaml").read_text(), Loader=ConfigLoader),
                          agent=yaml.load((config_dir / "agent.yaml").read_text(), Loader=ConfigLoader))
    options = EvalOptions(num_envs=args.num_envs, duration=args.seconds, seeds=tuple(args.seeds))
    options.validate()
    result = {"scope": "frozen policy, nominal physics, paired reward evaluation; NOT retraining",
              "candidate": args.candidate,
              "clearance_weight_override": args.clearance_weight,
              "checkpoint": str(checkpoint), "seconds": args.seconds, "seeds": args.seeds,
              "num_envs": args.num_envs, "config_differences": {}, "rows": []}
    current = load_env_cfg("Ri-4438-HIM-Rough")
    for name, term in current.rewards.items():
        saved = run.env["rewards"].get(name, {})
        if saved.get("weight") != term.weight:
            result["config_differences"][name] = {"saved_weight": saved.get("weight"), "current_weight": term.weight}
    print("CONFIG", json.dumps(result["config_differences"]), flush=True)
    for terrain in args.terrains:
        cfg, _ = make_env_cfg(run, options, terrain)
        cfg.rewards = deepcopy(current.rewards)
        if args.candidate == "phase":
            from mjlab.sensor import GridPatternCfg, ObjRef, TerrainHeightSensorCfg
            cfg.scene.sensors += (TerrainHeightSensorCfg(
                name="feet_terrain_height",
                frame=tuple(ObjRef(type="site", name=name, entity="robot") for name in ("FL", "FR", "RL", "RR")),
                pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1),
                ray_alignment="world", max_distance=2.0,
                exclude_parent_body=True, include_geom_groups=(0,), reduction="min",
            ),)
        # A second score using saved weights is reported, but saved source code
        # is not available: this is NOT a reconstruction of historical training.
        env = ManagerBasedRlEnv(cfg, device=args.device)
        try:
            alt_cfg = deepcopy(cfg.rewards)
            changed_term = "foot_gait" if args.candidate == "timeout" else "foot_clearance"
            if args.candidate == "timeout":
                alt_cfg[changed_term].func = feet_gait_contact_timeout
            else:
                alt_cfg[changed_term].func = feet_clearance_phase
                alt_cfg[changed_term].params = {
                    key: value for key, value in cfg.rewards["foot_gait"].params.items() if key != "sensor_name"}
                alt_cfg[changed_term].params["height_range"] = cfg.rewards["foot_clearance"].params["height_range"]
                if args.clearance_weight is not None:
                    if not math.isfinite(args.clearance_weight) or args.clearance_weight >= 0:
                        raise ValueError("clearance weight must be finite and negative")
                    alt_cfg[changed_term].weight = args.clearance_weight
            shadow = RewardManager(alt_cfg, env)
            baseline_compute = env.reward_manager.compute
            names = list(env.reward_manager.active_terms)
            gait_index = names.index("foot_gait")
            changed_index = names.index(changed_term)
            other_indices = [i for i, name in enumerate(names) if name != changed_term]
            max_other_error = 0.0

            def paired_compute(dt):
                nonlocal max_other_error
                baseline = baseline_compute(dt).clone()
                alternate = shadow.compute(dt).clone()
                before = env.reward_manager._step_reward
                after = shadow._step_reward
                error = (before[:, other_indices] - after[:, other_indices]).abs().max().item()
                max_other_error = max(max_other_error, error)
                torch.testing.assert_close(before[:, other_indices], after[:, other_indices], rtol=0, atol=0)
                torch.testing.assert_close(alternate - baseline,
                                           (after[:, changed_index] - before[:, changed_index]) * dt,
                                           rtol=1e-4, atol=2e-6)
                return baseline

            env.reward_manager.compute = paired_compute
            commands = list(STAIR_COMMANDS if terrain.startswith("stairs_") else COMMANDS)
            if args.forward_speed is not None:
                for c, (name, label, value) in enumerate(commands):
                    if name == "forward":
                        commands[c] = (name, label, (args.forward_speed, 0.0, 0.0))
                        env.command_manager.get_command("twist")[c::len(commands), 0] = args.forward_speed
            observations, _ = env.reset(seed=args.seeds[0])
            shadow.reset(torch.arange(env.num_envs, device=env.device))
            actor, _, action_dim, _ = load_checkpoint_actor(
                run, checkpoint, TensorDict({"actor": observations["actor"]}, batch_size=[env.num_envs]), args.device)
            assert action_dim == env.action_manager.total_action_dim
            robot = env.scene["robot"]
            gait_params = env.reward_manager.get_term_cfg("foot_gait").params
            gait_period = gait_params["period"]
            saved_weights = torch.tensor([run.env["rewards"][name]["weight"] for name in names], device=env.device)
            weights = torch.tensor([env.reward_manager.get_term_cfg(name).weight for name in names], device=env.device)
            for seed in args.seeds:
                env._evaluation_seed = seed
                env.common_step_counter = 0
                observations, _ = env.reset(seed=seed)
                shadow.reset(torch.arange(env.num_envs, device=env.device))
                alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
                records = []
                start = robot.data.root_link_pos_w.clone()
                last_position = start.clone()
                failures = torch.zeros_like(alive)
                for step in range(round(args.seconds / env.step_dt)):
                    with torch.no_grad():
                        actions = actor(TensorDict({"actor": observations["actor"]}, batch_size=[env.num_envs]))
                        clip = run.agent.get("clip_actions")
                        if clip is not None:
                            actions = actions.clamp(-clip, clip)
                        observations, _, terminated, truncated, _ = env.step(actions)
                    current_contact = env.scene["feet_ground_contact"].data.current_contact_time
                    command = env.command_manager.get_command("twist")
                    is_active = torch.linalg.norm(command[:, :2], dim=1) + command[:, 2].abs() > 0.1
                    stuck = (is_active
                             & (torch.linalg.norm(robot.data.root_link_lin_vel_b[:, :2], dim=1) < 0.1)
                             & (robot.data.root_link_ang_vel_b[:, 2].abs() < 0.1))
                    rates = env.reward_manager._step_reward
                    alt = shadow._step_reward
                    # Skip initial landing transient; never count reset episodes.
                    valid = alive & (step * env.step_dt >= 1.0)
                    values = torch.stack([
                        valid.float(), stuck.float(), ((current_contact > gait_period).any(1) & is_active).float(),
                        rates[:, gait_index], alt[:, gait_index], rates[:, names.index("foot_clearance")],
                        rates.sum(1), alt.sum(1), (rates / weights * saved_weights).sum(1),
                        torch.linalg.norm(command[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=1),
                        (rates[:, changed_index] - alt[:, changed_index]),
                        current_contact.max(1).values,
                        ((current_contact > gait_period).all(1) & is_active).float(),
                        alt[:, names.index("foot_clearance")],
                    ], dim=1)
                    records.append(values.detach().cpu().numpy())
                    last_position[alive] = robot.data.root_link_pos_w[alive]
                    failures |= alive & terminated
                    alive &= ~(terminated | truncated)
                    ids = (terminated | truncated).nonzero().flatten()
                    if ids.numel():
                        observations, _ = env.reset(env_ids=ids)
                        shadow.reset(ids)
                    if not alive.any():
                        break
                data = np.stack(records)
                for c, (name, _, command_value) in enumerate(commands):
                    indices = np.arange(c, env.num_envs, len(commands))
                    subset = data[:, indices].reshape(-1, data.shape[-1])
                    subset = subset[subset[:, 0] > 0]
                    if not len(subset):
                        continue
                    stalled = subset[subset[:, 1] > 0]
                    motion = subset[subset[:, 1] == 0]
                    row = dict(terrain=terrain, seed=seed, command=name, command_value=command_value,
                               samples=len(subset), failures=int(failures[indices].sum().item()),
                               progress_m=float((last_position - start)[indices, 0].mean().item()),
                               stall_fraction=float(subset[:, 1].mean()),
                               timeout_fraction=float(subset[:, 2].mean()),
                               gait_before=float(subset[:, 3].mean()), gait_after=float(subset[:, 4].mean()),
                               clearance=float(subset[:, 5].mean()), total_before=float(subset[:, 6].mean()),
                               total_after=float(subset[:, 7].mean()), total_saved_weights=float(subset[:, 8].mean()),
                               linear_error=float(subset[:, 9].mean()), added_cost=float(subset[:, 10].mean()),
                               max_contact_s=float(subset[:, 11].max()),
                               all_feet_overdue_fraction=float(subset[:, 12].mean()),
                               clearance_after=float(subset[:, 13].mean()),
                               stalled_added_cost=float(stalled[:, 10].mean()) if len(stalled) else None,
                               moving_added_cost=float(motion[:, 10].mean()) if len(motion) else None,
                               other_reward_max_error=max_other_error)
                    result["rows"].append(row)
                    print("ROW", json.dumps(row), flush=True)
        finally:
            env.close()
    print("RESULT", json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate", choices=["timeout", "phase"], default="phase")
    parser.add_argument("--forward-speed", type=float, default=None)
    parser.add_argument("--clearance-weight", type=float, default=None,
                        help="Override only the candidate clearance weight; baseline retains current config")
    parser.add_argument("--seeds", nargs=3, type=int, default=[0, 1, 2])
    parser.add_argument("--terrains", nargs="+", default=["flat", "rough", "stairs_up_10cm", "stairs_down_10cm", "stairs_up_15cm"])
    audit_checkpoint(parser.parse_args())
