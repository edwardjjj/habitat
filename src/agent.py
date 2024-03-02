from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from envs.habitat.env_factory import VectorEnvWrapper
from habitat.core.dataset import Dataset, Episode
from model import SemanticPredMaskRCNN


class LlmAgent:
    envs: VectorEnvWrapper
    device: torch.device
    config: DictConfig

    # image processing objects
    image_resize: Any
    selem: np.ndarray
    dialation_kernel: np.ndarray

    # agent observation and info
    eve_angle: float
    observation: np.ndarray
    observation_shape: Tuple
    collision_map: np.ndarray
    explored_map: np.ndarray
    col_width: int
    curr_pose: np.ndarray
    last_pose: np.ndarray
    last_action: Dict[str, int]
    replan_count: int
    collision_n: int
    fail_case: Dict[str, int]

    # Semantic Segmentation model
    sem_pred: SemanticPredMaskRCNN

    def __init__(
        self,
        full_config: DictConfig,
        envs: VectorEnvWrapper,
    ) -> None:
        self.envs = envs
        self.config = full_config
        self.device = full_config.general.agent_gpu_id

        # Initialize image processing objects
        self.selem = skimage.morphology.disk(3)
        self.dialation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Initialize Segmentation Models
        self.mrcnn_sem_pred = SemanticPredMaskRCNN(full_config)
        self.fail_case = {
            "collision": 0,
            "detection": 0,
            "exploration": 0,
            "success": 0,
        }

        self.eve_angle = 0.0

    def reset(self) -> Tuple[torch.Tensor, Sequence[Dict]]:
        observations_list, info_list = self.envs.reset()
        observations_tensor = self._preprocess_observation(observations_list)
        self.observation_shape = observations_tensor.shape

        map_shape = (
            self.config.sem_map.map_size_cm // self.config.sem_map.map_resolution,
            self.config.sem_map.map_size_cm // self.config.sem_map.map_resolution,
        )
        self.collision_map = np.zeros(map_shape)
        self.explored_map = np.zeros(map_shape)
        self.col_width = 1
        self.replan_count = 0
        self.curr_pose = np.asarray(
            [
                self.config.sem_map.map_size_cm / 100 / 2,
                self.config.sem_map.map_size_cm / 100 / 2,
                0.0,
            ]
        )
        self.last_pose = np.asarray(
            [
                self.config.sem_map.map_size_cm / 100 / 2,
                self.config.sem_map.map_size_cm / 100 / 2,
                0.0,
            ]
        )
        self.eve_angle = 0
        return observations_tensor, info_list

    def _preprocess_observation(
        self, observations_list: Sequence[np.ndarray]
    ) -> torch.Tensor:
        semantic_labels = self.get_semantic_labels(observations_list)
        return torch.zeros(1)

    def get_semantic_labels(self, observations_list: Sequence[np.ndarray]):
        return self.mrcnn_sem_pred(observations_list)

        return labels
