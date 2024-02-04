from typing import Any, Dict, Tuple
from pathlib import Path
import quaternion

import numpy as np
from habitat.core.env import RLEnv

target_category_to_id = [0, 3, 2, 4, 5, 1]
target_id_to_category = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]

mp3d_category_id = {
    "void": 1,
    "chair": 2,
    "sofa": 3,
    "plant": 4,
    "bed": 5,
    "toilet": 6,
    "tv_monitor": 7,
    "table": 8,
    "refrigerator": 9,
    "sink": 10,
    "stairs": 16,
    "fireplace": 12,
}


class ObjNavEnv(RLEnv):
    rank: int
    episode_num: int
    curr_dis_to_goal: float
    prev_dis_to_goal: float
    timestep: int
    stopped: bool
    path_length: float
    last_pose: Tuple[float, float, float]
    info: Dict[str, Any]

    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        self.rank = rank

        super().__init__(config_env, dataset)

        # Initializations
        self.episode_num = 0

        #  # Scene info
        #  self.last_scene_path = None
        #  self.scene_path = None
        #  self.scene_name = None
        #
        #  Episode Dataset info
        # self.eps_data = None
        # self.eps_data_idx = None
        # self.gt_planner = None
        # self.object_boundary = None
        # self.goal_idx = None
        # self.goal_name = None
        # self.map_obj_origin = None
        # self.starting_loc = None
        # self.starting_distance = None
        #
        # Episode tracking info
        self.curr_distance = 0
        self.prev_distance = 0
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.last_sim_location = (0, 0, 0)
        self.info = {}

        # create category dict
        fileName = Path("data/matterport_category_mappings.tsv")
        lines = []
        self.hm3d_semantic_mapping = {}

        with open(fileName) as file:
            file.readline()
            for line in file:
                lines.append(line.split("    "))

        for line in lines:
            self.hm3d_semantic_mapping[line[2]] = line[-1].strip()

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to a new episode.

        Returns:
            obs : RGBD observations (4 x H x W)
            info : contains timestep, pose, goal category and
                        evaluation metric info
        """

        self.episode_num += 1

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        # self.trajectory_states = []

        # if new_scene:
        obs = super().reset()
        self.scene = self.habitat_env.sim.semantic_annotations()
        # start_height = self._env.current_episode.start_position[1]
        # goal_height = self.scene.objects[self._env.current_episode.info['closest_goal_object_id']].aabb.center[1]

        # floor_height = []
        # floor_size = []
        # for obj in self.scene.objects:
        #     if obj.category.name() in self.hm3d_semantic_mapping and \
        #         self.hm3d_semantic_mapping[obj.category.name()] == 'floor':
        #         floor_height.append(abs(obj.aabb.center[1] - start_height))
        #         floor_size.append(obj.aabb.sizes[0]*obj.aabb.sizes[2])

        # if not args.eval:
        #     while all(h > 0.1 or s < 12 for (h,s) in zip(floor_height, floor_size)) or abs(start_height-goal_height) > 1.2:
        #         obs = super().reset()

        #         self.scene = self._env.sim.semantic_annotations()
        #         start_height = self._env.current_episode.start_position[1]
        #         goal_height = self.scene.objects[self._env.current_episode.info['closest_goal_object_id']].aabb.center[1]

        #         floor_height = []
        #         floor_size = []
        #         for obj in self.scene.objects:
        #             if obj.category.name() in self.hm3d_semantic_mapping and \
        #                 self.hm3d_semantic_mapping[obj.category.name()] == 'floor':
        #                 floor_height.append(abs(obj.aabb.center[1] - start_height))
        #                 floor_size.append(obj.aabb.sizes[0]*obj.aabb.sizes[2])

        self.prev_distance = self.habitat_env.get_metrics()["distance_to_goal"]
        self.starting_distance = self.habitat_env.get_metrics()["distance_to_goal"]

        rgb = obs["rgb"]
        depth = obs["depth"]
        semantic = self._preprocess_semantic(obs["semantic"])

        state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)
        self.last_sim_location = self.get_agent_pose()

        # Set info
        self.info["time"] = self.timestep
        self.info["sensor_pose"] = [0.0, 0.0, 0.0]
        self.info["goal_cat_id"] = target_category_to_id[obs["objectgoal"][0]]
        self.info["goal_name"] = target_id_to_category[obs["objectgoal"][0]]

        return state, self.info

    def step(
        self, dict_action: Dict[str, int]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                            evaluation metric info
        """

        action = dict_action["action"]
        if action == 0:
            self.stopped = True
            # Not sending stop to simulator, resetting manually
            action = 3

        obs, rew, done, _ = super().step(action)

        # Get pose change
        dx, dy, do = self.get_pose_change()
        self.info["sensor_pose"] = [dx, dy, do]
        self.path_length += get_l2_distance(0, dx, 0, dy)

        if done:
            spl, success, dist = self.get_metrics()
            self.info["distance_to_goal"] = dist
            self.info["spl"] = spl
            self.info["success"] = success

        rgb = obs["rgb"].astype(np.uint8)
        depth = obs["depth"]
        semantic = self._preprocess_semantic(obs["semantic"])
        state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)

        self.timestep += 1
        self.info["time"] = self.timestep

        return state, rew, done, self.info

    def _preprocess_semantic(self, semantic):
        # list of unique semantic labels
        sem_category = list(set(semantic.ravel()))
        for item in sem_category:
            if self.scene.objects[item].category.name() in self.hm3d_semantic_mapping:
                hm3d_category_name = self.hm3d_semantic_mapping[
                    self.scene.objects[item].category.name()
                ]
            else:
                hm3d_category_name = self.scene.objects[item].category.name()

            if hm3d_category_name in mp3d_category_id:
                semantic[semantic == item] = mp3d_category_id[hm3d_category_name] - 1
            else:
                semantic[semantic == item] = 0

        semantic = np.expand_dims(semantic.astype(np.uint8), 2)
        return semantic.astype(np.uint8)

    def get_reward_range(self) -> Tuple[float, float]:
        """This function is not used, Habitat-RLEnv requires this function"""
        return 0.0, 1.0

    def get_reward(self, observations: Any) -> float:
        """Return a dense reward based on distance to goal

        Args:
            observations: not used

        Returns:
            dense reward
        """
        self.curr_distance = self._env.get_metrics()["distance_to_goal"]

        reward = (self.prev_distance - self.curr_distance) * self.args.reward_coeff

        self.prev_distance = self.curr_distance
        return reward

    def get_metrics(self) -> Tuple[float, bool, float]:
        """This function computes evaluation metrics for the Object Goal task

        Returns:
            spl (float): Success weighted by Path Length
            success (bool):  Success: True, Failure: Flase
            dist (float): Distance to Success (DTS),  distance of the agent
                        from the success threshold boundary in meters.
        """
        dist = self.habitat_env.get_metrics()["distance_to_goal"]
        if dist < 0.1:
            success = True
        else:
            success = False
        spl = min(success * self.starting_distance / self.path_length, 1)
        return spl, success, dist

    def get_done(self, observations: Any) -> bool:
        if self.info["time"] >= self.args.max_episode_length - 1:
            done = True
        elif self.stopped:
            done = True
        else:
            done = False
        return done

    def get_info(self, observations: Any) -> Dict:
        return self.info

    def get_agent_pose(self) -> Tuple[float, float, float]:
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = self.habitat_env.sim.get_agent_state(agent_id=0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self) -> Tuple[float, float, float]:
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_agent_pose()
        dx, dy, do = get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_pose = curr_sim_pose
        return dx, dy, do


def get_l2_distance(x1: float, x2: float, y1: float, y2: float) -> float:
    """
    Computes the L2 distance between two points.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_rel_pose_change(
    pos2: Tuple[float, float, float], pos1: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2

    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1

    return dx, dy, do
