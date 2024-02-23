from objnav_env import ObjNavEnv
from omegaconf import DictConfig

from habitat import VectorEnv
from habitat.datasets.registration import make_dataset


def make_objnavenv(args, config: DictConfig, rank: int) -> ObjNavEnv:
    dataset = make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    config.defrost()
    config.simulator.scene = dataset.episodes[0].scene_id
    config.freeze()
    env = ObjNavEnv(config=config, dataset=dataset, rank=rank)
    env.seed(rank)
    return env

def construct_vec_objnavenv(config: DictConfig) -> VectorEnv:

