import habitat

config = habitat.get_config(
    config_path="/home/edward/Projects/test/habitat/pointnav.yaml"
)
env = habitat.Env(config=config)
