"""Independent RI-4438 HIM LP-ACRL defaults.

Behavior values copied from the workspace RI-4438 HIM configuration on 2026-10-10.
No inheritance from, or imports of, another task's behavior configuration.
LP/evaluation defaults are engineering choices, not unpublished paper parameters.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any






@dataclass
class RobotBehaviorCfgControl:
    action_scale: float = 0.5
    armature: float = 0.008234
    damping: dict = field(default_factory=lambda: {'hip': 2.0694, 'thigh': 2.0694, 'calf': 2.0694})
    decimation: int = 4
    delay_hold_prob: float = 0.4
    delay_max_lag: int = 3
    delay_min_lag: int = 0
    delay_update_period: int = 10
    effort_limit: dict = field(default_factory=lambda: {'hip': 10.0, 'thigh': 10.0, 'calf': 10.0})
    hip_reduction: float = 0.5
    stiffness: dict = field(default_factory=lambda: {'hip': 32.5065, 'thigh': 32.5065, 'calf': 32.5065})


@dataclass
class RobotBehaviorCfgDomainRand:
    KpKd_factor_range: list = field(default_factory=lambda: [0.8, 1.2])
    base_mass_range: list = field(default_factory=lambda: [0.85, 1.15])
    com_displacement_range: list = field(default_factory=lambda: [-0.05, 0.05])
    effort_limit_range: list = field(default_factory=lambda: [0.8, 1.0])
    external_force_cooldown_s: list = field(default_factory=lambda: [6.0, 10.0])
    external_force_duration_s: list = field(default_factory=lambda: [0.1, 0.3])
    external_force_range: list = field(default_factory=lambda: [-6.0, 6.0])
    foot_contact_time_constant_range: list = field(default_factory=lambda: [0.015, 0.03])
    imu_gyro_bias_range: list = field(default_factory=lambda: [-0.02, 0.02])
    imu_mount_angle_range: list = field(default_factory=lambda: [-0.02045329252, 0.02045329252])
    joint_armature_range: list = field(default_factory=lambda: [0.8, 1.2])
    joint_damping_range: list = field(default_factory=lambda: [0.5, 1.5])
    joint_friction_range: list = field(default_factory=lambda: [0.5, 1.5])
    link_mass_range: list = field(default_factory=lambda: [0.9, 1.1])
    motor_offset_range: list = field(default_factory=lambda: [-0.02, 0.02])


@dataclass
class RobotBehaviorCfgEnv:
    env_spacing: float = 3.0
    history_size: int = 6
    num_actions: int = 12
    num_one_step_obs: int = 47


@dataclass
class RobotBehaviorCfgInitState:
    default_joint: dict = field(default_factory=lambda: {'FL_hip_joint': -0.0, 'RL_hip_joint': -0.0, 'FR_hip_joint': 0.0, 'RR_hip_joint': 0.0, 'FL_thigh_joint': 0.9, 'RL_thigh_joint': 0.9, 'FR_thigh_joint': 0.9, 'RR_thigh_joint': 0.9, 'FL_calf_joint': -1.8, 'RL_calf_joint': -1.8, 'FR_calf_joint': -1.8, 'RR_calf_joint': -1.8})
    pos: list = field(default_factory=lambda: [0.0, 0.0, 0.27])




@dataclass
class RobotBehaviorCfgReset:
    base_offset: dict = field(default_factory=lambda: {'pose_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5), 'z': (-0.0, 0.0), 'roll': (-0.3, 0.3), 'pitch': (-0.3, 0.3), 'yaw': (-3.14, 3.14)}, 'velocity_range': {}})
    joint_offset: dict = field(default_factory=lambda: {'position_range': (-0.0, 0.0), 'velocity_range': (-0.0, 0.0)})




@dataclass
class RobotBehaviorCfg:
    control: RobotBehaviorCfgControl = field(default_factory=RobotBehaviorCfgControl)
    domain_rand: RobotBehaviorCfgDomainRand = field(default_factory=RobotBehaviorCfgDomainRand)
    env: RobotBehaviorCfgEnv = field(default_factory=RobotBehaviorCfgEnv)
    init_state: RobotBehaviorCfgInitState = field(default_factory=RobotBehaviorCfgInitState)
    reset: RobotBehaviorCfgReset = field(default_factory=RobotBehaviorCfgReset)


@dataclass
class HimLearningCfgAlgorithm:
    clip_param: float = 0.2
    desired_kl: float = 0.01
    entropy_coef: float = 0.01
    estimator_learning_rate: float = 0.001
    estimator_max_grad_norm: float = 10.0
    estimator_target_slices: tuple = ((0, 6), (9, 47), (47, 50))
    estimator_velocity_slice: tuple = (47, 50)
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 0.001
    max_grad_norm: float = 1.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    schedule: str = 'adaptive'
    use_clipped_value_loss: bool = True
    value_loss_coef: float = 1.0


@dataclass
class HimLearningCfgPolicy:
    activation: str = 'elu'
    actor_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    encoder_hidden_dims: tuple = (128, 64, 16)
    history_order: str = 'frame_major_oldest_first'
    history_size: int = 6
    history_term_dims: tuple = (3, 3, 3, 2, 12, 12, 12)
    init_noise_std: float = 1.0
    num_one_step_obs: int = 47
    num_prototype: int = 32
    obs_normalization: bool = True
    sinkhorn_eps: float = 0.05
    sinkhorn_iters: int = 3
    target_hidden_dims: tuple = (128, 64)
    temperature: float = 3.0




@dataclass
class HimLearningCfg:
    algorithm: HimLearningCfgAlgorithm = field(default_factory=HimLearningCfgAlgorithm)
    policy: HimLearningCfgPolicy = field(default_factory=HimLearningCfgPolicy)


@dataclass
class CommandAxisCfg:
    mode: str = "fixed"
    value: float = 0.0
    range: tuple[float, float] | None = None
    edges: list[float] | None = None
    bins: dict[str, tuple[float, float]] | None = None
    signs: str = "both"
    positive_probability: float = 0.5
    sign_as_task_axis: bool = False


@dataclass
class TerrainFamilyCfg:
    levels: dict[str, dict[str, Any]] = field(default_factory=lambda: {"base": {}})


@dataclass
class TaskSpaceCfg:
    command_axes: dict[str, CommandAxisCfg] = field(default_factory=lambda: {
        "vx": CommandAxisCfg(mode="magnitude_bins", edges=[0.0, 0.5, 1.0, 1.5]),
        "vy": CommandAxisCfg(), "wz": CommandAxisCfg(),
    })
    terrains: dict[str, TerrainFamilyCfg] = field(default_factory=lambda: {
        "flat": TerrainFamilyCfg(),
    })
    include: list[dict[str, str | list[str]]] = field(default_factory=list)
    exclude: list[dict[str, str | list[str]]] = field(default_factory=list)


@dataclass
class GeometryCfg:
    replicas: int = 1
    patch_size_m: tuple[float, float] = (8.0, 8.0)
    boundary_margin_m: float = 1.1
    seed: int = 0


@dataclass
class CurriculumCfg:
    method: str = "lp"
    temperature: float = 1.0
    uniform_mix: float = 0.0


@dataclass
class EvaluationCfg:
    interval_updates: int = 50
    trials_per_task: int = 8
    num_envs: int = 128
    horizon_s: float = 20.0
    seed: int = 314159
    max_consecutive_failures: int = 3


@dataclass
class TrainingCfg:
    num_envs: int = 1024
    rollout_steps: int = 100
    max_iterations: int = 20000
    save_interval: int = 200
    horizon_s: float = 20.0
    seed: int = 0
    device: str = "cuda:0"
    output_dir: str = "src/tasks/locomotion/ri_4438_him_lpacrl/outputs"
    resume: str | None = None
    warm_start: str | None = None
    video: bool = False
    video_interval: int = 2000
    video_length: int = 200


@dataclass
class ExperimentCfg:
    task_space: TaskSpaceCfg = field(default_factory=TaskSpaceCfg)
    geometry: GeometryCfg = field(default_factory=GeometryCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
    evaluation: EvaluationCfg = field(default_factory=EvaluationCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    robot: RobotBehaviorCfg = field(default_factory=RobotBehaviorCfg)
    learning: HimLearningCfg = field(default_factory=HimLearningCfg)
    reward_weights: dict[str, float] = field(default_factory=dict)
    reward_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    scan_size_m: tuple[float, float] = (1.6, 1.0)
    scan_resolution_m: float = 0.1
