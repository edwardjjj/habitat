from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Protocol, Sequence, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

from habitat.core.vector_env import VectorEnv
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from habitat.datasets.registration import make_dataset

from .objnav_env import ObjNavEnv


class CustomEnv(Protocol):
    def seed(self, *args, **kwargs): ...


@dataclass
class VectorEnvWrapper:
    """Wrapper class of VectorEnv that unpacks a list of tuples to a tuple of list of observations, infos ..."""

    vec_env: VectorEnv

    @property
    def num_envs(self):
        return self.vec_env.num_envs

    @property
    def observation_space(self):
        return self.vec_env.observation_spaces

    @property
    def action_space(self):
        return self.action_space

    def reset(self) -> Tuple[Sequence[np.ndarray], Sequence[Dict]]:
        results = self.vec_env.reset()
        observations, info = zip(*results)
        return observations, info

    def step(
        self, actions
    ) -> Tuple[Sequence[np.ndarray], Sequence[float], Sequence[bool], Sequence[Dict]]:
        results = self.vec_env.step(actions)
        observations, rewards, dones, infos = zip(*results)
        return observations, rewards, dones, infos

    def get_metrics(self) -> List[Dict]:
        metrics = self.vec_env.get_metrics()
        return metrics

    def get_rewards(self) -> List[float]:
        commands = ["get_reward"] * self.num_envs
        results = self.vec_env.call(commands)
        return results

    def call_at(
        self,
        index: int,
        function_name: str,
        function_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        r"""Calls a function or retrieves a property/member variable (which is passed by name)
        on the selected env and returns the result.

        :param index: which env to call the function on.
        :param function_name: the name of the function to call or property to retrieve on the env.
        :param function_args: optional function args.
        :return: result of calling the function.
        """
        return self.vec_env.call_at(index, function_name, function_args)

    def close(self):
        return self.vec_env.close()


def make_env_fn(cls: type[CustomEnv], config: DictConfig, rank: int) -> Any:
    """Creates a single custom RL env
    Args:
        cls: Custom RLEnv class
        config: config to create env
        rank: rank to set random seed

    Returns:
        Custon RLEnv instance
    """
    dataset = make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    config.habitat.simulator.scene = dataset.episodes[0].scene_id
    env = cls(config=config, dataset=dataset, rank=rank)
    env.seed(config.habitat.seed + rank)
    return env


def construct_vec_env(
    make_env_fn: Callable[[DictConfig, int], Any], config: DictConfig
) -> VectorEnv:
    """Construct vector envs of custom RL environment

    Args:
        make_env_fn: partial function that with cls initialized
        config: config to create the environment

    Raises:
        Exception:
            - config does not contain any scenes
            - num_process is greater than num_scenes
            - num_process exceeds limit of gpus

    Returns:
        VectorEnv
    """
    env_config_list = []
    rank_list = range(config.general.num_processes)

    # dataset = make_dataset(
    #     id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    # )
    if "*" in config.habitat.dataset.content_scenes:
        scenes = ObjectNavDatasetV1.get_scenes_to_load(config.habitat.dataset)
    else:
        scenes = config.habitat.dataset.content_scenes

    if not scenes:
        raise Exception("content_scenes is EMPTY!")

    # assert at least one scene for one process
    assert (
        len(scenes) >= config.general.num_processes
    ), "Not enough number of scenes, try reduce num_processes"

    # assert there are enough GPUs for all the processes
    assert (
        config.general.num_processes
        <= (torch.cuda.device_count() - 1) * config.general.num_processes_per_gpu
        + config.general.num_processes_on_first_gpu
    ), "Not enough gpus to accommodate all the processes, try decrease num_processes or increase num_processes_per_gpu"

    scene_split_sizes = split_scenes(
        num_scenes=len(scenes), num_processes=config.general.num_processes
    )

    for i in range(config.general.num_processes):
        new_config = config.copy()
        new_config.habitat.dataset.content_scenes = scenes[
            sum(scene_split_sizes[:i]) : sum(scene_split_sizes[: i + 1])
        ]

        if i < new_config.general.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = (
                i - new_config.general.num_processes_on_first_gpu
            ) // new_config.general.num_processes_per_gpu
        new_config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id
        env_config_list.append(new_config)
    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(tuple(zip(env_config_list, rank_list))),
    )
    return envs


def split_scenes(num_scenes: int, num_processes: int) -> List[int]:
    """create a list of num_scenes for each process

    Args:
        num_scenes
        num_processes:

    Returns:
        a list of num_scenes per each process
    """
    scenes_per_process = num_scenes // num_processes
    scenes_mod = num_scenes % num_processes
    scene_split_sizes = [scenes_per_process] * num_processes
    if scenes_mod > 0:
        for i in range(scenes_mod):
            scene_split_sizes[i] += 1
    return scene_split_sizes
