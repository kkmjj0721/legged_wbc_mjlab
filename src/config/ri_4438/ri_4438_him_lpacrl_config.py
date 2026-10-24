"""RI-4438 HIM + LPACRL 的独立配置，不继承其他任务的配置。

结构与 Ri4438Cfg / Ri4438CFGHimPPO 一致：
- Ri4438HimLpacrlCfg：环境、机器人、命令分箱、地形、在线 LP 和离线评估。
- Ri4438HimLpacrlCfgPPO：HIM 网络、PPO 算法和训练轮数。

这里声明默认值；task 的配置工厂会复制它们，训练和评估不会修改这些类。
teacher 的 signed LP/softmax 参考 FIBO 5813fc3；阶段统计和覆盖采样是本项目扩展。
"""

import math

# HIM 数值边界；独立保留，避免其他任务调整后影响本任务。
ACTION_CLIP = 10.0
OBSERVATION_CLIP = 100.0
RAW_ACTION_WARN = 20.0
RAW_OBSERVATION_ABORT = 1000.0
ACTION_OBSERVATION_SLICE = (35, 47)


class Ri4438HimLpacrlCfg:
    """环境与课程配置；奖励项在 task 的 *_env_cfg.py 中定义。"""

    class env:
        num_envs = 1024
        num_actions = 12
        num_one_step_obs = 3 + 3 + 3 + 2 + num_actions * 3  # 47
        history_size = 6
        env_spacing = 3.0
        episode_length_s = 20.0

    class control:
        stiffness = {"hip": 32.5065, "thigh": 32.5065, "calf": 32.5065}
        damping = {"hip": 2.0694, "thigh": 2.0694, "calf": 2.0694}
        effort_limit = {"hip": 10.0, "thigh": 10.0, "calf": 10.0}
        action_scale = 0.5
        decimation = 4  # 控制周期 = 仿真步长 × decimation。
        hip_reduction = 0.5
        armature = 0.008234
        delay_min_lag = 0
        delay_max_lag = 3
        delay_hold_prob = 0.4
        delay_update_period = 10

    class init_state:
        pos = [0.0, 0.0, 0.27]
        default_joint = {
            "FL_hip_joint": -0.0,
            "RL_hip_joint": -0.0,
            "FR_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "RL_thigh_joint": 0.9,
            "FR_thigh_joint": 0.9,
            "RR_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "RL_calf_joint": -1.8,
            "FR_calf_joint": -1.8,
            "RR_calf_joint": -1.8,
        }

    class reset:
        base_offset = {
            "pose_range": {
                "x": (-0.1, 0.1),  # 配合 2.5 m 楼梯平台，避免出生时足端跨台阶。
                "y": (-0.1, 0.1),
                "z": (-0.0, 0.0),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        }
        joint_offset = {
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        }

    class domain_rand:
        base_mass_range = [0.85, 1.15]  # 躯干质量/惯量倍率，与腿部分开。
        link_mass_range = [0.9, 1.1]
        KpKd_factor_range = [0.8, 1.2]
        com_displacement_range = [-0.05, 0.05]  # 每个方向的质心偏移，m。
        motor_offset_range = [-0.02, 0.02]
        effort_limit_range = [0.8, 1.0]
        joint_friction_range = [0.5, 1.5]
        joint_armature_range = [0.8, 1.2]
        joint_damping_range = [0.5, 1.5]
        imu_gyro_bias_range = [-0.02, 0.02]  # rad/s。
        imu_mount_angle_range = [-0.02045329252, 0.02045329252]  # rad。
        external_force_range = [-6.0, 6.0]  # 世界坐标系各方向的力，N。
        external_force_duration_s = [0.1, 0.3]
        external_force_cooldown_s = [6.0, 10.0]
        foot_contact_time_constant_range = [0.015, 0.03]

    class commands:
        # 默认任务 = 4 个 Ax × 4 个 Ay × 地形类型 × 该类型的等级。
        # 固定任务分箱供历史 A0～A3 和独立评估使用。
        fixed_tasks = False
        heading_command = True
        heading_control_stiffness = 0.5  # 与独立 HIM 相同：wz = k * wrap(target-current)。
        ang_vel_z_range = (-1.0, 1.0)  # 最终交给策略的角速度限幅，rad/s。
        axes = {
            "vx": {
                "mode": "magnitude_bins",
                "edges": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # m/s。
            },
            "vy": {"mode": "continuous", "range": (-1.0, 1.0)},  # m/s。
            "wz": {
                "mode": "magnitude_bins",
                "edges": [0.0, 1 / 6, 2 / 6, 0.5, 4 / 6, 5 / 6, 1.0],  # rad/s。
            },
            "heading": {
                "mode": "magnitude_bins",
                "edges": [i * math.pi / 6 for i in range(7)],  # 相对出生朝向的误差，rad。
            },
        }
        # 固定任务只在 reset 采样；随机过程使用独立的 process_hold_s。
        resampling_interval_s = None
        # Independent nested velocity ranges, world-frame heading targets.
        random_process = True
        process_amplitudes = (0.2, 0.5, 0.8, 1.0)
        process_heading_range = (-3.14, 3.14)
        process_hold_s = (2.0, 10.0)
        # 固定任务默认箱内连续采样；随机过程始终按矩形范围连续采样。
        sample_bin_centers = False
        # 每个轴的完整默认值；axes 中的同名字段覆盖这里。
        axis_defaults = {
            "mode": "fixed",
            "value": 0.0,
            "range": None,
            "edges": None,
            "bins": None,
            "signs": "both",
            "positive_probability": 0.5,
            "sign_as_task_axis": False,
        }
        include = []  # 按 vx/vy/heading（直接模式为 wz）、terrain、level 筛选任务。
        exclude = []

    class terrain:
        # 默认 Rough：4 个 Ax × 4 个 Ay × (8 类楼梯 × 5 档 + 3 类其他地形 × 4 档)
        # = 832 个任务及 teacher 组。楼梯类别由踏步深度和上下方向共同确定。
        # 每个任务的几何难度固定；LP 只改变任务被采样的概率。
        # rough 按高度上限分级，每张图采样 [0, height_max_m]；这是项目定义。
        # 楼梯保留 RI-4438 的 0.05–0.16 m 范围，论文使用 0.05–0.20 m。
        # gradient = 高差 / 水平距离，不是角度：0、0.4/3、0.8/3、0.4。
        # 恢复第一版的四档等间隔坡度比，对应约 0°/7.59°/14.93°/21.80°。
        # 两类坡地的 l0 都是平地，保留各自任务标签；不另加 flat 地形类别。
        levels = {
            "random_rough": {
                "l0": {"height_max_m": 0.02},
                "l1": {"height_max_m": 0.03},
                "l2": {"height_max_m": 0.045},
                "l3": {"height_max_m": 0.06},
            },
            # 同一类别的深度固定，只按高度分五档；018 表示深度 0.18 m。
            # 8 m 地块分别放置 10/8/7/6 层踏步，配套边宽保持中心平台 2.5 m。
            # 边宽减去 1 nm，避免浮点取整少生成一层踏步而扩大平台。
            **{
                f"stairs_{label}_outward_{direction}": {
                    f"l{i}": {
                        "step_height_m": height,
                        "step_width": depth,
                        "border_width": border - 1e-9,
                    }
                    for i, height in enumerate((0.05, 0.08, 0.10, 0.12, 0.16))
                }
                for label, depth, border in (
                    ("018", 0.18, 0.95),
                    ("021", 0.21, 1.07),
                    ("025", 0.25, 1.00),
                    ("028", 0.28, 1.07),
                )
                for direction in ("up", "down")
            },
            "slope_outward_up": {
                "l0": {"gradient": 0.0},
                "l1": {"gradient": 0.4 / 3},
                "l2": {"gradient": 0.8 / 3},
                "l3": {"gradient": 0.4},
            },
            "slope_outward_down": {
                "l0": {"gradient": 0.0},
                "l1": {"gradient": 0.4 / 3},
                "l2": {"gradient": 0.8 / 3},
                "l3": {"gradient": 0.4},
            },
        }
        # 地形类别的基础几何参数，levels 中的字段覆盖基础参数；长度单位为 m。
        defaults = {
            "flat": {},
            "random_rough": {
                "height_max_m": 0.05,
                "noise_step": 0.005,
                "horizontal_scale": 0.1,
                "vertical_scale": 0.005,
                "downsampled_scale": None,
                "border_width": 0.25,
            },
            "stairs": {
                "step_height_m": 0.10,
                "step_width": 0.18,  # 各楼梯类别使用 levels 中声明的固定深度。
                "platform_width": 2.5,
                "border_width": 0.95,  # 各深度的配套边宽在 levels 中覆盖。
                "holes": False,
            },
            "slope": {
                "gradient": 0.20,  # 高差 / 水平距离，对应约 11.31°。
                "platform_width": 2.0,
                "border_width": 0.25,
                "horizontal_scale": 0.1,
                "vertical_scale": 0.001,  # 1 mm 高度量化。
            },
        }
        replicas = 1  # 仅 legacy_v6 使用副本行；V7 使用显式核心区出生列表。
        patch_size_m = (8.0, 8.0)
        boundary_margin_m = 1.1
        seed = 0
        map_protocol = "connected_v7"  # legacy_v6 is an explicit historical control.
        core_rows = 5
        surrounding_rings = 1  # 5×11 核心区外扩一圈：7×13，共 91 块。
        outer_flat_width_m = 16.0
        pass_margin_m = 0.05
        pass_confirm_s = 0.2

    class height_scan:
        size_m = (1.6, 1.0)
        resolution_m = 0.1  # 对应 187 条射线，与 HIM critic 的观测维度一致。

    class compact:
        """Explicit body-velocity tasks; selected only by the Compact presets."""

        family_fractions = {"steady": 0.35, "stairs": 0.25, "sequence": 0.30, "random": 0.10}
        uniform_families = ("random",)
        sequence_hold_s = 4.0
        process_hold_s = (2.0, 10.0)
        interaction_metrics_version = 2
        interaction_linear_threshold = 0.05  # Observe 0.1 m/s starts and stalls.
        interaction_linear_tolerance = 0.05  # Stationary is not a successful low-speed switch.
        # Eight terrain conditions, with four broad treads and a 2.5 m platform.
        # Stair exit is 2.45 m from the centre, before the 2.9 m root boundary.
        # Keep the 1.1 m boundary margin so the complete height scan stays inside.
        terrain_levels = {
            "flat": {"base": {}},
            "random_rough": {"low": {"height_max_m": 0.02}},
            **{
                f"stairs_outward_{direction}": {
                    f"h{round(height * 100):02d}": {
                        "step_height_m": height,
                        "step_width": 0.30,
                        "platform_width": 2.5,
                        "border_width": 1.55 - 1e-9,
                    }
                    for height in (0.05, 0.08, 0.10)
                }
                for direction in ("up", "down")
            },
        }
        # Each sign is a separate steady task and teacher group.
        steady_flat = {
            "stand": (0.0, 0.0, 0.0),
            **{f"vx_{speed:+g}": (speed, 0.0, 0.0)
               for speed in (0.1, -0.1, 0.2, -0.2, 0.5, -0.5)},
            "vy_+0.2": (0.0, 0.2, 0.0),
            "vy_-0.2": (0.0, -0.2, 0.0),
            "wz_+0.5": (0.0, 0.0, 0.5),
            "wz_-0.5": (0.0, 0.0, -0.5),
        }
        steady_rough = {
            f"vx_{speed:+g}": (speed, 0.0, 0.0)
            for speed in (0.2, -0.2, 0.5, -0.5)
        }
        stairs_speeds = (0.3, 0.5)
        # Five 4 s segments cover the complete default 20 s episode.
        sequences = {
            "vx_speed_changes": tuple((v, 0.0, 0.0) for v in (0.1, 0.3, 0.5, 0.2, 0.1)),
            **{
                f"vx_start_stop_{speed:g}": tuple((v, 0.0, 0.0) for v in (0.0, speed, 0.0, -speed, 0.0))
                for speed in (0.1, 0.2)
            },
            **{
                f"vx_reverse_{speed:g}": tuple((v, 0.0, 0.0) for v in (speed, -speed, speed, -speed, 0.0))
                for speed in (0.2, 0.5)
            },
            **{
                f"vy_start_stop_{speed:g}": tuple((0.0, v, 0.0) for v in (0.0, speed, 0.0, -speed, 0.0))
                for speed in (0.1, 0.2)
            },
            "vy_reverse_0.2": tuple((0.0, v, 0.0) for v in (0.2, -0.2, 0.2, -0.2, 0.0)),
            "wz_start_stop_0.5": tuple((0.0, 0.0, v) for v in (0.0, 0.5, 0.0, -0.5, 0.0)),
            "wz_reverse_0.5": tuple((0.0, 0.0, v) for v in (0.5, -0.5, 0.5, -0.5, 0.0)),
        }
        random_ranges = ((-0.5, 0.5), (-0.3, 0.3), (-0.5, 0.5))

    class compact_rough:
        """Extend Compact skills to 1.0 on the unchanged full Rough map.

        Existing low-speed skills and family quotas come from ``compact``.
        Both slope l0 conditions are physically flat; use one for flat skills
        without adding a terrain column to the frozen connected map.
        """

        flat_terrain = ("slope_outward_up", "l0")
        random_ranges = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        steady_flat_extra = {
            f"{axis}_{speed:+g}": tuple(speed if i == index else 0.0 for i in range(3))
            for index, axis in enumerate(("vx", "vy", "wz"))
            for speed in (1.0, -1.0)
        }
        steady_rough_extra = {
            f"vx_{speed:+g}": (speed, 0.0, 0.0) for speed in (1.0, -1.0)
        }
        stairs_speeds = (0.3, 0.5, 1.0)
        sequences_extra = {
            "vx_speed_changes_1": tuple((v, 0.0, 0.0) for v in (0.1, 0.5, 1.0, 0.5, 0.1)),
            **{
                f"{axis}_{behavior}_1": tuple(
                    tuple(v if i == index else 0.0 for i in range(3)) for v in values
                )
                for index, axis in enumerate(("vx", "vy", "wz"))
                for behavior, values in (
                    ("start_stop", (0.0, 1.0, 0.0, -1.0, 0.0)),
                    ("reverse", (1.0, -1.0, 1.0, -1.0, 0.0)),
                )
            },
        }

    class motion_tracking:
        """Current all-terrain motion semantics; disable for the old protocol.

        Translation thresholds/std use m/s; angular threshold uses rad/s.
        These are training candidates, not calibrated hardware limits.
        """

        enabled = True
        linear_command_threshold = 0.05
        angular_command_threshold = 0.05
        low_speed_std = 0.05
        low_speed_threshold = 0.1
        high_speed_threshold = 0.5
        low_speed_progress_ratio = 0.7
        low_speed_full_command = 0.2
        low_speed_end_command = 0.5

    class behavior_monitor:
        # Legacy presets preserve their exact reward and termination contracts.
        enabled = False
        penalty_weight = -0.5
        termination_enabled = False
        linear_command_threshold = 0.08
        angular_command_threshold = 0.1
        linear_progress_floor = 0.02
        angular_progress_floor = 0.02
        progress_ratio = 0.2
        smoothing_s = 0.3
        grace_s = 0.8
        penalty_sustain_s = 0.4
        termination_sustain_s = 3.0
        command_change_thresholds = (0.08, 0.08, 0.1)
        heading_change_threshold = 0.2

    class curriculum:
        method = "lp"  # 有符号进步；uniform 为相同覆盖条件下的均匀对照。
        # 线速度/角速度跟踪核按两项正奖励权重归一化。
        # 物理失败按完整回合预算记分；正常超时/地形出界按实际片段平均。
        # teacher 分组 = Ax × Ay × 出生地形类型 × 等级；方向和航向在组内采样。
        temperature = 0.025
        uniform_mix = 0.20
        stage_length_updates = 50
        min_episodes_per_group = 8  # 不足时保留样本和旧基线，禁止空组填零。
        seed_bin = 0  # A0 保留旧 2:1 初始先验；V7 连通地图从常规组均匀分布开始。
        # 默认所有回合由 teacher 抽样；历史 A0～A3 单独恢复 25% 直爬覆盖。
        forward_stairs_fraction = 0.0
        forward_stairs_velocity = 0.5
        forward_stairs_heights_m = (0.05, 0.08, 0.10, 0.12)

    class evaluation:
        protocol_version = 7  # 连通地图、完整出口与显式指令事件。
        trials_per_task = 8  # 每个任务固定试验数，与并行环境数独立。
        num_envs = 512  # 默认 Rough：832 × 8 = 6656 次试验，分 13 批评估。
        horizon_s = 20.0
        seed = 314159
        # 变指令/零指令下的项目约定：逐步 |误差| / max(|指令|, floor)，
        # 对存活段求均值后截到 [0, 1]；再按论文公式补足失败后的最坏误差。
        relative_error_floors = (0.1, 0.1, 0.1)  # vx/vy: m/s；wz: rad/s。
        success_alive_fraction = 0.9  # 严格大于；1000 步试验对应 >900 存活步。
        success_tracking_error = 0.3  # vx、wz 的 EPTE-SP 均须严格小于此值。
        # 部署指标是项目扩展；不改变论文 success 或训练指令采样。
        # 参考 Go2 LP-ACRL 的 max(绝对容差, 相对容差*|指令|)，避免零指令百分比失真。
        deployment_tracking_abs = (0.1, 0.15, 0.2)  # vx/vy: m/s；wz: rad/s。
        deployment_tracking_rel = (0.3, 0.3, 0.3)
        deployment_stall_s = 2.0  # 连续低进度达到此时间判失败。
        deployment_stall_window_s = 0.5  # 速度滑动平均，避免原地摆动掩盖停滞。
        deployment_stall_command = (0.2, 0.2)  # 平面速度 m/s、角速度 rad/s。
        deployment_stall_speed = (0.05, 0.05)  # 沿指令方向的最低实际进度。
        deployment_change_thresholds = (0.2, 0.2, 0.3)  # vx/vy/wz 的显著指令变化。
        deployment_transition_grace_s = 1.0
        deployment_transition_window_s = 1.0  # 宽限之后单独检查每次切换，防止均值掩盖。
        deployment_segment_s = 3.0  # 独立平地验收序列每段长度。


class Ri4438HimLpacrlCfgPPO:
    """HIM 策略、估计器、PPO 与 runner 默认参数。"""

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        obs_normalization = True
        num_one_step_obs = Ri4438HimLpacrlCfg.env.num_one_step_obs
        history_size = Ri4438HimLpacrlCfg.env.history_size
        history_term_dims = (3, 3, 3, 2, 12, 12, 12)
        history_order = "frame_major_oldest_first"
        encoder_hidden_dims = (128, 64, 16)
        target_hidden_dims = (128, 64)
        num_prototype = 32
        temperature = 3.0  # HIM 原型学习温度，与 curriculum.temperature 独立。
        sinkhorn_eps = 0.05
        sinkhorn_iters = 3

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        estimator_learning_rate = 1.0e-3
        estimator_max_grad_norm = 10.0
        estimator_velocity_slice = (47, 50)
        estimator_target_slices = ((0, 6), (9, 47), (47, 50))

    class runner:
        num_steps_per_env = 100
        max_iterations = 20000  # 本次新增的 PPO 更新数，续训同样如此。
        save_interval = 200
        experiment_name = "ri_4438_him_lpacrl"
        seed = 0
