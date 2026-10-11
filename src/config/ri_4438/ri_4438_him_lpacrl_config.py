"""RI-4438 HIM + LPACRL 的独立配置，不继承其他任务的配置。

结构与 Ri4438Cfg / Ri4438CFGHimPPO 一致：
- Ri4438HimLpacrlCfg：环境、机器人、命令分箱、地形和 LP 评估。
- Ri4438HimLpacrlCfgPPO：HIM 网络、PPO 算法和训练轮数。

这里声明默认值；task 的配置工厂会复制它们，训练和评估不会修改这些类。
LP 评估参数是本项目的工程设置，不代表论文未公开的超参数。
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
                "x": (-0.1, 0.1),  # 配合 1.2 m 楼梯平台，避免出生时足端跨台阶。
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
        # 任务 = 命令分箱 × 地形类型 × 该类型的等级。
        # 保留论文的 5 × 6 分箱结构，但 vx/wz 的范围都限制为 [-1, 1]。
        # |vx| 档宽 0.2 m/s；|wz| 档宽 1/6 rad/s。
        # 正负方向在箱内各占一半；vy 连续采样，不增加任务数量。
        axes = {
            "vx": {
                "mode": "magnitude_bins",
                "edges": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # m/s。
            },
            "vy": {"mode": "continuous", "range": (-0.5, 0.5)},  # m/s。
            "wz": {
                "mode": "magnitude_bins",
                "edges": [0.0, 1 / 6, 2 / 6, 0.5, 4 / 6, 5 / 6, 1.0],  # rad/s。
            },
        }
        resampling_interval_s = 2.0  # 回合内只更新箱内命令，任务/地形保持不变。
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
        include = []  # 可按 vx/vy/wz、terrain、level 筛选任务；空列表保留全部。
        exclude = []

    class terrain:
        # 默认 Rough：5 个 vx 分箱 × 6 个 wz 分箱 × 5 类地形 × 4 档 = 600 个任务。
        # 每个任务的几何难度固定；LP 只改变任务被采样的概率。
        # rough 按高度上限分级，每张图采样 [0, height_max_m]；这是项目定义。
        # 楼梯保留 RI-4438 的 0.05–0.16 m 范围，论文使用 0.05–0.20 m。
        # 坡角按 0°/0.10°/0.20°/0.30° 分级；生成器使用 tan(坡角) 作为梯度。
        # 两类坡地的 l0 都是平地，保留各自任务标签；不另加 flat 地形类别。
        levels = {
            "random_rough": {
                "l0": {"height_max_m": 0.02},
                "l1": {"height_max_m": 0.03},
                "l2": {"height_max_m": 0.045},
                "l3": {"height_max_m": 0.06},
            },
            "stairs_outward_up": {
                "l0": {"step_height_m": 0.05},
                "l1": {"step_height_m": 0.08},
                "l2": {"step_height_m": 0.12},
                "l3": {"step_height_m": 0.16},
            },
            "stairs_outward_down": {
                "l0": {"step_height_m": 0.05},
                "l1": {"step_height_m": 0.08},
                "l2": {"step_height_m": 0.12},
                "l3": {"step_height_m": 0.16},
            },
            "slope_outward_up": {
                "l0": {"gradient": 0.0},
                "l1": {"gradient": math.tan(math.radians(0.10))},
                "l2": {"gradient": math.tan(math.radians(0.20))},
                "l3": {"gradient": math.tan(math.radians(0.30))},
            },
            "slope_outward_down": {
                "l0": {"gradient": 0.0},
                "l1": {"gradient": math.tan(math.radians(0.10))},
                "l2": {"gradient": math.tan(math.radians(0.20))},
                "l3": {"gradient": math.tan(math.radians(0.30))},
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
                "step_width": 0.30,
                "platform_width": 1.2,
                "border_width": 1.0,
                "holes": False,
            },
            "slope": {
                "gradient": math.tan(math.radians(0.20)),  # 默认坡角 0.20°。
                "platform_width": 2.0,
                "border_width": 0.25,
                "horizontal_scale": 0.1,
                "vertical_scale": 0.0001,  # 0.1 mm 量化，保留小坡角的高度差。
            },
        }
        replicas = 1  # 每档地形的副本数；地形图的列对应等级，行对应副本。
        patch_size_m = (8.0, 8.0)
        boundary_margin_m = 1.1
        seed = 0

    class height_scan:
        size_m = (1.6, 1.0)
        resolution_m = 0.1  # 对应 187 条射线，与 HIM critic 的观测维度一致。

    class curriculum:
        method = "lp"  # lp：有符号进步；alp：进步绝对值；uniform：均匀采样。
        temperature = 1.0  # LP softmax 温度；越小越集中。
        uniform_mix = 0.0  # 混入均匀分布的比例，范围 [0, 1)。

    class evaluation:
        protocol_version = 2  # 固定时长；出界重置同一任务继续评估，物理失败结束。
        interval_updates = 50  # 按 PPO 更新次数触发，不是 episode 数。
        trials_per_task = 8  # 每个任务固定试验数，与并行环境数独立。
        num_envs = 512  # 600 × 8 = 4800 次试验，分 10 批评估；128 环境需 38 批。
        horizon_s = 20.0
        seed = 314159
        max_consecutive_failures = 3


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
