import habitat

config = habitat.get_config(
    config_path="pointnav.yaml", configs_dir="/home/edward/Projects/test/habitat"
)
env = habitat.Env(config=config)
