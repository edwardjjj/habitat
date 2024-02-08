from objnav_env import ObjNavEnv
import numpy as np


class LlmAgent:
    env: ObjNavEnv
    mrcnn_sem_pred: SemanticPredMaskRCNN
    rednet_sem_pred: RedNet
    device: torch.device
    image_resize: Any
    selem: np.ndarray
    observation: np.ndarray
    observation_shape: np.ndarray
    collision_map: np.ndarray
    explored_map: np.ndarray
    col_width: int
    curr_pose: np.ndarray
    last_pose: np.ndarray
    last_action: Dict[str, int]
    replan_count: int
    collision_n: int
    dialation_kernel: np.ndarray
    fail_case: Dict[str, int]
