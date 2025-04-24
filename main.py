import cv2
import numpy as np
import os
import argparse
import requests
import logging
from src.nt_schema import VISION_TIMESTAMP
import trimesh
import json
import sys
import re
import signal
import datetime
from networktables import NetworkTables
from urllib.parse import urlparse  
from src.estimater import *
from src.datareader import *
from src.file_processing import FileLoader
from src.masking import MaskingTool
from src.transform import PoseTransformer
from src.nt_schema import (
    ROOT_TABLE, COMMANDS_TABLE, STATUS_TABLE, VISION_TABLE, DIAGNOSTICS_TABLE,
    FOUNDATION_POSE, ARM_TARGET_POSITION, ARM_COMMAND_READY,
    ARM_CURRENT_POSITION, ARM_STATE, ARM_ERROR,
    ARM_COMMAND_RECEIVED, ARM_COMMAND_EXECUTED,
    COMMAND_TIMESTAMP, VISION_TIMESTAMP,
    NT_UPDATE_FREQUENCY, EXPECTED_LATENCY_MS
)

output_dir = "output_dir"
debug_dir = os.path.join(output_dir, "debug")
base_url = "https://6d25-162-218-227-129.ngrok-free.app"

def run_foundation_pose(rgb_img, depth_img, mask_img, mesh, K):
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=2,
        glctx=glctx
    )

    mask_bool = mask_img.astype(bool)
    pose = est.register(
        K=K,
        rgb=rgb_img,
        depth=depth_img,
        ob_mask=mask_bool,
        iteration=5
    )
    
    # Save pose result
    pose_path = os.path.join(output_dir, f"{timestamp}.txt")
    np.savetxt(pose_path, pose.reshape(4, 4))
    
    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=rgb_img, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(rgb_img, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
    vis_path = os.path.join(output_dir, f"{timestamp}.png")
    cv2.imwrite(vis_path, vis)
    
    # Show visualization
    cv2.imshow("Pose Estimation Result", vis)
    print("✅ Pose estimation complete. Closing in 5 seconds.")
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
    return center_pose

def publish_pose(pose_6d):
    NetworkTables.initialize(server=base_url)

    # Wait for NT connection
    while not NetworkTables.isConnected():
        print("Waiting for NetworkTables connection...")
        time.sleep(1)
    
    print(f"Connected to NetworkTables server")

    pose_str = f"{pose_6d[0]:.3f},{pose_6d[1]:.3f},{pose_6d[2]:.3f},{pose_6d[3]:.3f},{pose_6d[4]:.3f},{pose_6d[5]:.3f}"
    
    vision_table = NetworkTables.getTable(VISION_TABLE)
    
    # Publish to NT using constants from schema
    vision_table.putString(FOUNDATION_POSE.split('/')[-1], pose_str)

    # Update timestamp for latency tracking
    timestamp_key = VISION_TIMESTAMP.split('/')[-1]
    vision_table.putNumber(timestamp_key, time.time())

    print(f"✅ Published pose: {pose_str}")

def main():
    print("📁 Creating output directories...")
    # create output and debug directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    print("🔄 Initializing file loader...")
    file_loading = FileLoader(base_url)

    #rgb_img = file_loading.fetch_rgb_image()
    #depth_img = file_loading.fetch_depth()

    print("📷 Loading RGB image...")
    rgb_img = file_loading.load_rgb_image_from_file()
    print("📊 Loading depth data...")
    depth_img = file_loading.load_depth_from_file()
    
    print("🎭 Creating object mask...")
    mask_path = os.path.join(output_dir, "mask.png")
    mask_img = MaskingTool().run(rgb_img, mask_path)
    
    print("📦 Loading 3D model...")
    mesh = trimesh.load("data/models/bent_test_1.obj")
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) > 0:
            mesh = list(mesh.geometry.values())[0]
    
    print("📏 Loading camera intrinsics...")
    K = file_loading.load_camera_intrinsics("data/intrinsics/cam_K.txt", "LEFT_CAM_FHD1200")

    print(rgb_img.shape)
    print(depth_img.shape)
    print(mask_img.shape)
    print(mesh)
    print(K)

    print("🔍 Running foundation pose estimation...")
    center_pose = run_foundation_pose(rgb_img, depth_img, mask_img, mesh, K)
    
    print("🔄 Transforming pose to 6D format...")
    pose_6d = PoseTransformer().transform_pose(center_pose)
    print(f"✅ Pose 6D: {pose_6d}")

    #publish_pose(pose_6d)

if __name__ == "__main__":
    main()