from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence, cast

import detectron2.data.transforms as T
import numpy as np
import torch
import torch.nn as nn
from detectron2 import model_zoo
from detectron2.config import get_cfg

if TYPE_CHECKING:
    from detectron2.config import CfgNode
    from detectron2.structures import Instances
    from omegaconf import DictConfig

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from torch import Tensor
from torch.nn import functional as F

import utils.depth_utils as du

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}


class Semantic_Mapping(nn.Module):
    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        # self.n_channels = 3
        self.vision_range = args.vision_range
        # self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(200 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.0
        self.shift_loc = np.asarray(
            [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        )
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov
        )

        # self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = (
            torch.zeros(
                args.num_processes,
                1 + self.num_sem_categories,
                vr,
                vr,
                self.max_height - self.min_height,
            )
            .float()
            .to(self.device)
        )
        self.feat = (
            torch.ones(
                args.num_processes,
                1 + self.num_sem_categories,
                self.screen_h // self.du_scale * self.screen_w // self.du_scale,
            )
            .float()
            .to(self.device)
        )

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.stair_mask_radius = 30
        self.stair_mask = self.get_mask(self.stair_mask_radius).to(self.device)

    def forward(
        self,
        obs: Tensor,
        pose_obs: Tensor,
        maps_last: Tensor,
        poses_last: Tensor,
        eve_angle: np.ndarray,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """foward pass on semantic map module

        Args:
            obs: [Batch x Channel x H x W] 0 - 3: rgbd, 4 - 19: semantic categories
            pose_obs: [Batch x (x, y, rotation)] agent current pose
            maps_last: [Batch x Chanel x M x M] 0: obstacle, 1: explored,
                        2 - 3: agent loc, 4 - 19: semantic categories
            poses_last: [Batch x (x, y, rotation)] agent pose from last step
            eve_angle: [Batch x elevation angle] camera elevation angle

        Returns:
            [TODO:return]
        """
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale
        )

        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, eve_angle, self.device
        )

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device
        )

        # normalize point_cloud
        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range

        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = XYZ_cm_std[..., :2] / xy_resolution
        XYZ_cm_std[..., :2] = (
            (XYZ_cm_std[..., :2] - vision_range // 2.0) / vision_range * 2.10
        )
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (
            (XYZ_cm_std[..., 2] - (max_h + min_h) // 2.0) / (max_h - min_h) * 2.0
        )
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(
            XYZ_cm_std.shape[0],
            XYZ_cm_std.shape[1],
            XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3],
        )

        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(obs[:, 4:, :, :]).view(
            bs, c - 4, h // self.du_scale * w // self.du_scale
        )

        # build voxel grids from point_cloud and splat semantic category labels onto voxel grids
        voxels = du.splat_feat_nd(
            self.init_grid * 0.0, self.feat, XYZ_cm_std
        ).transpose(2, 3)

        # ----------------------------------------------------------------------
        # Project voxel grids onto 2D grids to create Maps
        # Channel information: 0 obstacle, 1 explored, 2 - 3 agent loc, 4 - 19 sem categories
        # ----------------------------------------------------------------------
        min_z = int(25 / z_resolution - min_h)
        max_z = int((self.agent_height + 50) / z_resolution - min_h)
        mid_z = int(self.agent_height / z_resolution - min_h)

        agent_height_proj = voxel_to_grid(voxels[..., min_z:max_z])
        agent_height_stair_proj = voxel_to_grid(voxels[..., mid_z - 5 : mid_z])
        all_height_proj = voxel_to_grid(voxels)

        # Obstacle Map
        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)

        # Explored Map
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        # Stairs Map
        fp_stair_pred = agent_height_stair_proj[:, 0:1, :, :]
        fp_stair_pred = fp_stair_pred / self.map_pred_threshold
        fp_stair_pred = torch.clamp(fp_stair_pred, min=0.0, max=1.0)

        # pose_pred = poses_last

        agent_view = torch.zeros(
            bs,
            c,
            self.map_size_cm // self.resolution,
            self.map_size_cm // self.resolution,
        ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold, min=0.0, max=1.0
        )

        agent_view_stair = agent_view.clone().detach()
        agent_view_stair[:, 0:1, y1:y2, x1:x2] = fp_stair_pred

        relative_pose_change = pose_obs

        current_poses = get_new_pose_batch(poses_last, relative_pose_change)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = -(
            st_pose[:, :2] * 100.0 / self.resolution
            - self.map_size_cm // (self.resolution * 2)
        ) / (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90.0 - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.shape, self.device)

        # rotate and transform the map to egocentric view

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        # translated[:, 18:19, :, :] = -self.max_pool(-translated[:, 18:19, :, :])

        # ----------------------------------------------------------------------
        # Create a binary mask when explored map value is large and obstacle map
        # value is small, so that when the camera elevation angle is 0, the masked
        # value on the obstacle map is set to 0
        # ----------------------------------------------------------------------
        diff_ob_ex = translated[:, 1:2, :, :] - self.max_pool(translated[:, 0:1, :, :])

        diff_ob_ex[diff_ob_ex > 0.8] = 1.0
        diff_ob_ex[diff_ob_ex != 1.0] = 0.0

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        # when elevation angle is 0, make the masked value on obstacle map 0
        for i in range(eve_angle.shape[0]):
            if eve_angle[i] == 0:
                map_pred[i, 0:1, :, :][diff_ob_ex[i] == 1.0] = 0.0

        # stairs view
        rot_mat_stair, trans_mat_stair = get_grid(
            st_pose, agent_view_stair.shape, self.device
        )

        # rotate and transform the stair map to ergocentric view

        rotated_stair = F.grid_sample(
            agent_view_stair, rot_mat_stair, align_corners=True
        )
        translated_stair = F.grid_sample(
            rotated_stair, trans_mat_stair, align_corners=True
        )

        # ----------------------------------------------------------------------
        # create stairs map for environment tweaking, the effect is questionable
        # ----------------------------------------------------------------------

        stair_mask = torch.zeros(
            self.map_size_cm // self.resolution, self.map_size_cm // self.resolution
        ).to(self.device)

        s_y = int(current_poses[0][1] * 100 / 5)
        s_x = int(current_poses[0][0] * 100 / 5)
        limit_up = self.map_size_cm // self.resolution - self.stair_mask_radius - 1
        limit_be = self.stair_mask_radius
        if s_y > limit_up:
            s_y = limit_up
        if s_y < self.stair_mask_radius:
            s_y = self.stair_mask_radius
        if s_x > limit_up:
            s_x = limit_up
        if s_x < self.stair_mask_radius:
            s_x = self.stair_mask_radius
        stair_mask[
            int(s_y - self.stair_mask_radius) : int(s_y + self.stair_mask_radius),
            int(s_x - self.stair_mask_radius) : int(s_x + self.stair_mask_radius),
        ] = self.stair_mask

        translated_stair[0, 0:1, :, :] *= stair_mask
        translated_stair[0, 1:2, :, :] *= stair_mask

        # translated_stair[:, 13:14, :, :] = -self.max_pool(-translated_stair[:, 13:14, :, :])

        diff_ob_ex = translated_stair[:, 1:2, :, :] - translated_stair[:, 0:1, :, :]

        diff_ob_ex[diff_ob_ex > 0.8] = 1.0
        diff_ob_ex[diff_ob_ex != 1.0] = 0.0

        maps3 = torch.cat((maps_last.unsqueeze(1), translated_stair.unsqueeze(1)), 1)

        map_pred_stair, _ = torch.max(maps3, 1)

        for i in range(eve_angle.shape[0]):
            if eve_angle[i] == 0:
                map_pred_stair[i, 0:1, :, :][diff_ob_ex[i] == 1.0] = 0.0

        return translated, map_pred, map_pred_stair, current_poses

    def get_mask(self, step_size):
        size = int(step_size) * 2
        mask = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                if ((i + 0.5) - (size // 2)) ** 2 + (
                    (j + 0.5) - (size // 2)
                ) ** 2 <= step_size**2:
                    mask[i, j] = 1
        return mask


def get_grid(pose: Tensor, grid_size: torch.Size, device: torch.device):
    """
    Input:
        pose: (bs, 3) (x, y, theta) theta is in degrees
        grid_size: 4-tuple (bs, _, grid_h, grid_w)
        device: torch.device (cpu or gpu)
    Output:
        rot_grid: (bs, grid_h, grid_w, 2) rotation matrix
        trans_grid: (bs, grid_h, grid_w, 2) transformation matrix

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    # convert t from degrees to radians
    t = t * np.pi / 180.0
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack(
        [cos_t, -sin_t, torch.zeros(cos_t.shape).float().to(device)], 1
    )
    theta12 = torch.stack(
        [sin_t, cos_t, torch.zeros(cos_t.shape).float().to(device)], 1
    )
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack(
        [torch.ones(x.shape).to(device), -torch.zeros(x.shape).to(device), x], 1
    )
    theta22 = torch.stack(
        [torch.zeros(x.shape).to(device), torch.ones(x.shape).to(device), y], 1
    )
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, cast(list[int], grid_size))
    trans_grid = F.affine_grid(theta2, cast(list[int], grid_size))

    return rot_grid, trans_grid


def voxel_to_grid(voxels: Tensor) -> Tensor:
    """Create 2d projection of the voxels grid

    Args:
        voxels: (...X x Y x Z) the voxels grid

    Returns:
        (... X x Y) the 2d projection
    """
    return voxels.sum(-1)


def get_new_pose_batch(pose, rel_pose_change):
    pose[:, 1] += rel_pose_change[:, 0] * torch.sin(
        pose[:, 2] / (180 / np.pi)
    ) + rel_pose_change[:, 1] * torch.cos(pose[:, 2] / (180 / np.pi))
    pose[:, 0] += rel_pose_change[:, 0] * torch.cos(
        pose[:, 2] / (180 / np.pi)
    ) - rel_pose_change[:, 1] * torch.sin(pose[:, 2] / (180 / np.pi))
    pose[:, 2] += rel_pose_change[:, 2] * (180 / np.pi)

    pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
    pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

    return pose


class SemanticPredMaskRCNN:
    predictor: BatchPredictor
    config: "DictConfig"
    model_config: "CfgNode"
    resizer: T.ResizeShortestEdge
    input_format: str

    """Semantic segmentation class using MaskRCNN under the hood

       This class expects input as a list of BGR images of type ndarray
       with the shape [H x W x C]


    """

    def __init__(self, config: "DictConfig"):
        model_config = SemanticPredMaskRCNN.setup_sem_model_config(config)
        self.predictor = BatchPredictor(model_config)
        self.config = config
        self.model_config = model_config
        self.resizer = T.ResizeShortestEdge(
            [model_config.INPUT.MIN_SIZE_TEST, model_config.INPUT.MIN_SIZE_TEST],
            model_config.INPUT.MAX_SIZE_TEST,
        )
        self.input_format = self.model_config.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(
        self, image_list: Sequence[np.ndarray], resize: bool = False
    ) -> Sequence[np.ndarray]:
        prediction_list = self.predictor(self._preprocess_image(image_list, resize))
        return self.get_sem_label(prediction_list)

    def get_raw_result(
        self, image_list: Sequence[np.ndarray], resize: bool = False
    ) -> Sequence[Dict[str, "Instances"]]:
        return self.predictor(self._preprocess_image(image_list, resize))

    def get_sem_label(
        self, prediction_list: Sequence[Dict[str, "Instances"]]
    ) -> Sequence[np.ndarray]:
        sem_label_list = []
        for prediction in prediction_list:
            height, width = prediction["instances"].image_size
            sem_label = np.zeros((height, width, 16))

            pred_classes = prediction["instances"].pred_classes.cpu().numpy()
            pred_masks = prediction["instances"].pred_masks.cpu().numpy()
            if len(pred_classes) > 0:
                for j, coco_class_label in enumerate(pred_classes):
                    if coco_class_label in coco_categories_mapping.keys():
                        sem_label_index = coco_categories_mapping[coco_class_label]
                        sem_label[pred_masks[j], sem_label_index] = 1
            sem_label_list.append(sem_label)

        return sem_label_list

    @staticmethod
    def setup_sem_model_config(config: "DictConfig") -> "CfgNode":
        model_config = get_cfg()
        file_path = config.path.maskrcnn_config
        model_config.merge_from_file(model_zoo.get_config_file(file_path))
        model_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            config.sem_map.sem_pred_prob_threshold
        )
        model_config.MODEL.RETINANET.SCORE_THRESH_TEST = (
            config.sem_map.sem_pred_prob_threshold
        )
        model_config.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            config.sem_map.sem_pred_prob_threshold
        )
        model_config.MODEL.DEVICE = "cuda:{}".format(config.general.segmentation_gpu_id)
        model_config.MODEL.WEIGHTS = config.path.maskrcnn_weights
        return model_config

    def _preprocess_image(
        self, image_list: Sequence[np.ndarray], resize: bool
    ) -> List[Dict]:
        data_list = []
        for image in image_list:
            if resize:
                image = self._resize_shortest_edge(image)
            if self.input_format == "RGB":
                image = image[:, :, ::-1]
            height, width = image.shape[:2]
            image = image.astype(np.float32).transpose(2, 0, 1)
            image = torch.as_tensor(image)
            data = {"image": image, "height": height, "width": width}
            data_list.append(data)
        return data_list

    def _resize_shortest_edge(self, image: np.ndarray) -> np.ndarray:
        return self.resizer.get_transform(image).apply_image(image)


class BatchPredictor:
    def __init__(self, config: "CfgNode") -> None:
        self.config = config.clone()
        self.model = build_model(self.config)
        self.model.eval()
        self.metadata = MetadataCatalog.get(self.config.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.config.MODEL.WEIGHTS)

    def __call__(self, data_list: List[Dict]) -> List[Dict[str, "Instances"]]:
        with torch.no_grad():
            predictions = self.model(data_list)

        return predictions
