import habitat

config = habitat.get_config(
    config_path="benchmark/nav/pointnav/pointnav_test.yaml",
    overrides=[
        "habitat.environment.max_episode_steps=10",
        "habitat.environment.iterator_options.shuffle=False",
    ],
)
env = habitat.Env(config=config)
