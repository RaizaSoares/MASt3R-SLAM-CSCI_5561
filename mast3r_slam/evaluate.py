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

import detectron2

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
    simplf_classes = ['person', 'car', 'motorcycle', 'bus', 'train', 'truck', 
                      'traffic light', 'stop sign', 'building']

    cfg = get_cfg()   # get a fresh new config
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # Set confidence threshold
    cfg.merge_from_file(model_zoo.get_config_file("Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml"))
    cfg.MODEL.WEIGHTS = "detectron2://Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x/139797668/model_final_be35db.pkl"

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    panoptic_seg, segments_info = outputs["panoptic_seg"]

    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    mask = panoptic_seg.cpu().numpy() if torch.is_tensor(panoptic_seg) else panoptic_seg
    filtered_mask = np.zeros_like(mask)
    id_to_class_map = {}

    image_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for seg in segments_info:
        class_name = meta.thing_classes[seg['category_id']] if seg['isthing'] else meta.stuff_classes[seg['category_id']]
        if class_name in simplf_classes:
            filtered_mask[mask == seg['id']] = seg['id']
            id_to_class_map[seg['id']] = f"{class_name} (ID: {seg['id']})"  # Append segmentation ID

    # Initialize Detectron2 Visualizer
    v = Visualizer(
        image_rgb, 
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
        scale=1.2, 
        instance_mode=ColorMode.IMAGE_BW  # Ensures masks are overlaid properly
    )

    mask2_filtered = np.where(np.isin(panoptic_seg, list(id_to_class_map.keys())), panoptic_seg, 0)
    mask3_indices = [i for i, seg in enumerate(segments_info) if seg['id'] in id_to_class_map]

    # Ensure Tensor format for compatibility with Detectron2
    mask2_tensor = torch.tensor(mask2_filtered, dtype=torch.int64, device=panoptic_seg.device)

    # Pass to visualization function
    out = v.draw_panoptic_seg_predictions(mask2_tensor, [segments_info[i] for i in mask3_indices])
  
    # Overlay segmentation IDs directly on image
    segmented_image = out.get_image()

    for seg in segments_info:
            if seg['id'] not in id_to_class_map.keys():
                continue
            # Get mask indices for current object
            mask_indices = np.argwhere(filtered_mask == seg['id'])  # Find pixels belonging to the object
            center_x, center_y = mask_indices[:, 1].mean().astype(int), mask_indices[:, 0].mean().astype(int)

            # Draw segmentation ID on image. TODO: This does not seem to work well :(
            # cv2.putText(segmented_image, f"ID: {seg['id']}", (center_x, center_y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

    # Display results with IDs on objects
    fig, ax = plt.subplots(figsize=(16, 9))  # No need for an array of axes
    ax.imshow(segmented_image)  # Display only the segmented frame

    ax.set_title('Segmented Frame')
    ax.axis("off")

    plt.tight_layout()
    plt.show()


    return filtered_mask, id_to_class_map

def extract_segmented_points_torch(seg_mask_torch, keyframe, id_to_class_map):
    """Extract 3D points for each segmented object using PyTorch."""
    #device = keyframe.K.device if keyframe.K is not None else "cuda:0"

    # Initialize a dictionary to store segmented object 3D points
    obj_points_dict = {class_name: [] for class_name in id_to_class_map.values()}

    # # Normalize pixel coordinates using intrinsic matrix
    # K_estimated = torch.tensor([
    #     [2.69926114e+03, 0.00000000e+00, 1.52650052e+03],
    #     [0.00000000e+00, 2.70273810e+03, 1.97478860e+03],
    #     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    # ], dtype=torch.float32, device=device)

    # if keyframe.K is not None:
    #     K_estimated = keyframe.K  # Use provided intrinsic matrix if available

    # If we assume the keyframe.X_canon already contains 3D world coordinates, meaning no projection step is required, we might not need K. But this assumption may be incorrect

    # Loop over each segmentation ID and extract its corresponding 3D points
    for segmented_id in id_to_class_map.keys():
        class_name = id_to_class_map[segmented_id]

        # Create a mask for the current object ID
        obj_mask = (seg_mask_torch == segmented_id)

        # Find all 3D points corresponding to the mask
        valid_3D_points = keyframe.X_canon[obj_mask.flatten()]  # Select only relevant points

        if valid_3D_points.shape[0] > 0:
            valid_3D_points_world = keyframe.T_WC.act(valid_3D_points)  # Convert to world space
            obj_points_dict[class_name] = valid_3D_points_world  # Assign transformed 3D points

    return obj_points_dict  # Dictionary mapping class names to their respective 3D points


def compute_distance_torch3D(obj_points_dict, keyframe):

    #obj_points_dict are already in camera coordinates
    camera_center = keyframe.T_WC.translation()
    #print("camera_center shape:", camera_center.shape)
    camera_center_3D = camera_center[0, :3]  # Remove batch and take first three elements
    object_info = {}  # Store distances and angles

    #print(f"Camera center {camera_center_3D}")

    for class_name, obj_points_3D in obj_points_dict.items():
        if obj_points_3D.shape[0] > 0:
            obj_center = torch.mean(obj_points_3D, dim=0)  # Compute centroid
            distance = torch.norm(obj_center - camera_center_3D)  # Distance from camera

            object_direction3d = obj_center - camera_center_3D  # Extract X, Y position
            #object_direction = object_direction3d[:2] 
            # TODO: Extract more meaningful info from angles wrt camera 
            # angle_radians = torch.atan2(object_direction[1], object_direction[0])  # Compute angle in radians
            # angle_degrees = torch.rad2deg(angle_radians)  # Convert to degrees

            if object_direction3d[0] < 0:
                direction = f"to your left."
            else:
                direction = f"to your right."

            # Store results
            object_info[class_name] = {
                "distance_meters": distance.item(),
                "direction": direction
            }
    
    print("\n*****************Distance Estimations******************")
    for class_name, info in object_info.items():
        print(f"{class_name}: {info['distance_meters']:.3f} meters, {info['direction']}")
    print("\n")
    return object_info 

def computeSegmentationAndObjectDistance(keyframes, seg_processed_frame_ids, c_conf_threshold):
    #pointclouds = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        # pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        # valid = (
        #     keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
        #     > c_conf_threshold
        # )

        if keyframe.frame_id not in seg_processed_frame_ids: #new frame
            img = keyframe.img
            numpy_img = img.cpu().numpy()
            numpy_img = np.transpose(numpy_img, (1, 2, 0))  # Convert to (H, W, C)
            numpy_img = (numpy_img * 255).astype(np.uint8)  # Convert normalized float32 to uint8

            seg_mask, id_to_class_mapping  = segment(numpy_img)
            device = "cuda:0"
            # Convert segmentation mask to PyTorch tensor
            seg_mask_torch = torch.tensor(seg_mask, device=device, dtype=torch.float32)

            # Extract 3D points corresponding to the segmentation mask
            obj_points_dict = extract_segmented_points_torch(seg_mask_torch, keyframe, id_to_class_mapping)

            # Compute distance from camera
            compute_distance_torch3D(obj_points_dict, keyframe=keyframe)
            seg_processed_frame_ids.append(keyframe.frame_id)

        
        #pointclouds.append(pW[valid])
    #pointclouds = np.concatenate(pointclouds, axis=0)
    return seg_processed_frame_ids


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
