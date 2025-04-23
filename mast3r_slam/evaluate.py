import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement

#EDIT Raiza
import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()

import json
import random
import matplotlib.pyplot as plt

from os import listdir, makedirs
from os.path import join

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog


#EDIT Raiza End


def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)

def segment(im):

    simplf_classes = ['person', 'car', 'motorcycle',  'bus', 'train', 'truck', 'traffic light','stop sign']
    
    cfg = get_cfg()   # get a fresh new config
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Set confidence threshold

    cfg.merge_from_file(model_zoo.get_config_file("Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml"))
    cfg.MODEL.WEIGHTS = "detectron2://Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x/139797668/model_final_be35db.pkl"

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    panoptic_seg, segments_info = outputs["panoptic_seg"]

    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    id_to_class = {k['id']: meta.thing_classes[k['category_id']] 
                   for k in segments_info if k['isthing']}

    mask = panoptic_seg.cpu().numpy() if torch.is_tensor(panoptic_seg) else panoptic_seg
    filtered_mask = np.zeros_like(mask)
    id_to_class_map = {}

    for seg in segments_info:
        class_name = meta.thing_classes[seg['category_id']] if seg['isthing'] else meta.stuff_classes[seg['category_id']]
        if class_name in simplf_classes:
            print(f"Class name {class_name}: with id: {seg['id']}")
            filtered_mask[mask == seg['id']] = seg['id']  # retain only selected ids
            id_to_class_map[seg['id']] = class_name  # Store mapping

    #np.savetxt("../segmask.txt", filtered_mask, delimiter=" ", fmt="%.4f")  # Saves with 4 decim

    image_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    v = Visualizer(
    image_rgb, 
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
    scale=1.2, 
    instance_mode=ColorMode.IMAGE_BW  # Ensures masks are overlaid properly
)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    ax[0].imshow(image_rgb)
    ax[1].imshow(out.get_image())

    ax[0].set_title('Original RGB Frame')
    ax[1].set_title('Segmented Frame')

    ax[0].axis("off")
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig("Fig_1.png")
    plt.show()

    return filtered_mask, id_to_class_map

def extract_segmented_points_torch(seg_mask_torch, keyframe, id_to_class_mapping):
    device = "cuda:0"
    
    # Get pixel locations of segmented regions
    obj_pixels = torch.nonzero(seg_mask_torch > 0)  # Shape: (N, 2)

    # Convert pixels to homogeneous form
    obj_pixels_homog = torch.cat([
        obj_pixels, torch.ones((obj_pixels.shape[0], 1), device=device)
    ], dim=1).T  # Shape: (3, N)

    # Normalize pixel coordinates using intrinsics

    K_estimated = torch.tensor( [[2.69926114e+03, 0.00000000e+00, 1.52650052e+03],
    [0.00000000e+00, 2.70273810e+03, 1.97478860e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    , dtype=torch.float32, device=device)

    if keyframe.K != None:
        K_estimated = keyframe.K

    obj_pixels_norm = torch.linalg.solve(K_estimated, obj_pixels_homog)  # K^-1 * [x, y, 1]

    # Find corresponding 3D points
    obj_points_3D = []
    for i in range(obj_pixels.shape[0]):
        pixel = obj_pixels_norm[:, i]
        distances = torch.norm(keyframe.X_canon - pixel, dim=1)  # Compute distances
        closest_idx = torch.argmin(distances)
        obj_points_3D.append(keyframe.X_canon[closest_idx])

    return torch.stack(obj_points_3D)  # Shape: (N, 3)

def compute_distance_torch(obj_points_3D):
    """Compute object distance using PyTorch tensors."""
    obj_center = torch.mean(obj_points_3D, dim=0)  # Get centroid
    distance = torch.norm(obj_center)  # Euclidean distance

    return distance.item()  # Convert to float for readability

def computeSegmentationAndObjectDistance(keyframes, c_conf_threshold):
    pointclouds = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )

        img = keyframe.img
        numpy_img = img.cpu().numpy()
        numpy_img = np.transpose(numpy_img, (1, 2, 0))  # Convert to (H, W, C)
        numpy_img = (numpy_img * 255).astype(np.uint8)  # Convert normalized float32 to uint8

        seg_mask, id_to_class_mapping  = segment(numpy_img)
        device = "cuda:0"
        # Convert segmentation mask to PyTorch tensor
        seg_mask_torch = torch.tensor(seg_mask, device=device, dtype=torch.float32)

        # Extract 3D points corresponding to the segmentation mask
        obj_points_3D = extract_segmented_points_torch(seg_mask_torch, keyframe, id_to_class_mapping)

        # Compute distance from camera
        distance = compute_distance_torch(obj_points_3D)

        print(f"Segmented object distance: {distance:.3f} meters")
        
        
        pointclouds.append(pW[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)
