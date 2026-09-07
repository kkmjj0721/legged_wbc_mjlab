import math
from dataclasses import replace


from mjlab.envs.mdp import events as event_fns, dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.sensor import RayCastSensorCfg, ContactSensorCfg

from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.envs.mdp import observations as obs_fns

from mjlab.envs.mdp.actions import JointPositionActionCfg

from mjlab.envs.mdp import rewards
from mjlab.managers.reward_manager import RewardTermCfg

