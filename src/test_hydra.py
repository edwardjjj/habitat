import hydra
from habitat.config.default import register_configs
from habitat.datasets.registration import make_dataset
from omegaconf import OmegaConf

from conf.default_structured_configs import FullConfig
from envs.habitat.objnav_env import ObjNavEnv

# register habitat default configs before hydra.main
register_configs()


@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: FullConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    dataset = make_dataset(
        id_dataset=cfg.habitat.dataset.type, config=cfg.habitat.dataset
    )
    # env = ObjNavEnv(full_config=cfg, rank=69, dataset=)


if __name__ == "__main__":
    main()
