from dataclasses import dataclass
from enum import Enum
from typing import Any

from hydra.core.config_store import ConfigStore

from habitat.config.default_structured_configs import HabitatConfig


class AgentType(Enum):
    """valid agent types"""

    sem_exp = "sem_exp"
    objnav = "objnav"


@dataclass
class AgentConfig:
    """base agent config"""

    eval: int = 0
    num_training_frames: int = 10_000_000
    num_eval_episodes: int = 400
    num_train_episodes: int = 10_000
    log_interval: int = 10
    model_save_interval: int = 1
    model_save_frequency: int = 500_000
    agent_type: AgentType = AgentType.sem_exp
    learning_rate: float = 2.5e-5
    global_hidden_states: int = 256
    epsilon: float = 1e-5
    alpha: float = 0.99
    gamma: float = 0.99
    use_gae: bool = False
    tau: float = 0.95
    entropy_coef: float = 0.001
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_global_steps: int = 20
    num_ppo_epochs: int = 4
    num_mini_batches: int = 32
    clip_threshold: float = 0.2
    use_recurrent_global: bool = False
    num_local_steps: int = 25
    intrinsic_reward_coef: float = 0.05
    num_sem_categories: int = 16


@dataclass
class SemMapConfig:
    """base semantic map config"""

    vision_range: float = 100
    map_resolution: int = 5
    du_scale: int = 1
    map_size_cm: int = 4800
    cat_pred_threshold: float = 5.0
    map_pred_threshold: float = 1.0
    exp_pred_threshold: float = 1.0
    collision_threshold: float = 0.10
    use_gt_sem: bool = False


@dataclass
class EnvConfig:
    """base environment config"""

    frame_height: int = 480
    frame_width: int = 640
    frame_height_downscale: int = 120
    frame_width_downscale: int = 160
    max_episode_length: int = 500
    camera_height_meter: float = 0.88
    field_of_view: float = 79.0
    turn_angle: int = 30
    min_depth: float = 0.5
    max_depth: float = 5.0
    success_dist: float = 1.0
    floor_threshold: float = 50.0
    min_dis_to_goal: float = 1.5
    max_dis_to_goal: float = 100.0


@dataclass
class FullConfig:
    """base full config"""

    habitat: HabitatConfig
    agent: AgentConfig
    sem_map: SemMapConfig
    environment: EnvConfig
    general: Any
    path: Any
    visualize: Any
    data: Any


cs = ConfigStore.instance()
cs.store(
    name="base_config",
    node=FullConfig,
)
cs.store(
    group="environment",
    name="base_environment",
    node=EnvConfig,
)
cs.store(
    group="agent",
    name="base_agent",
    node=AgentConfig,
)
cs.store(
    group="sem_map",
    name="base_sem_map",
    node=SemMapConfig,
)
