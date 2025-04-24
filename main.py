#!/usr/bin/env python3
"""
FoundationPose Pipeline
-----------------------
A modular pipeline for 3D pose estimation using FoundationPose.
"""
import argparse
import sys
import os
import cv2
import signal

from src.pipeline.base import Pipeline
from src.pipeline.data import PipelineData

from src.modules.input.http_client import HttpImageFetcher
from src.modules.input.file_loader import FileImageLoader

from src.modules.detection.ground_dino import GroundingDinoDetector

from src.modules.segmentation.sam import SAM2Segmenter
from src.modules.segmentation.manual_masking import ManualMasker

from src.modules.pose.foundation_pose import FoundationPoseEstimator
from src.modules.pose.transform import PoseTransformer

from src.modules.output.network_tables import NetworkTablesPublisher
from src.modules.output.file_saver import ResultSaver

from src.utils.visualization import create_pipeline_visualization

# Global flag for exit request
exit_requested = False

# Signal handler for graceful exit
def signal_handler(sig, frame):
    global exit_requested
    print("\n⚠️ Ctrl+C detected! Exiting gracefully...")
    exit_requested = True
    cv2.destroyAllWindows()
    
# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='FoundationPose Pipeline')
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--url', default="http://localhost:8080", 
                        help='Base URL of the HTTP server (e.g., http://localhost:8080)')
    input_group.add_argument('--use-file', action='store_true',
                        help='Use local files instead of HTTP')
    input_group.add_argument('--rgb-file', default=None,
                        help='Path to RGB image file (when using --use-file)')
    input_group.add_argument('--depth-file', default=None,
                        help='Path to depth image file (when using --use-file)')
    input_group.add_argument('--depth-npy-file', default=None,
                        help='Path to depth NPY file (when using --use-file)')
    
    # Detection/segmentation options
    detect_group = parser.add_argument_group('Detection/Segmentation Options')
    detect_group.add_argument('--manual-mask', action='store_true',
                        help='Use manual masking instead of automatic detection/segmentation')
    detect_group.add_argument('--prompt', default="bent sheet metal",
                        help='Text prompt for Grounding DINO detection (default: "bent sheet metal")')
    detect_group.add_argument('--confidence', type=float, default=0.25,
                        help='Confidence threshold for detection (default: 0.25)')
    detect_group.add_argument('--box-threshold', type=float, default=0.2,
                        help='Box threshold for detection (default: 0.2)')
    
    # Pose estimation options
    pose_group = parser.add_argument_group('Pose Estimation Options')
    pose_group.add_argument('--mesh-file', default="data/bent_test_1.obj", 
                        help='Path to the CAD model file (.obj)')
    pose_group.add_argument('--cam-intrinsics', default="data/cam_K.txt",
                        help='Path to camera intrinsics file (cam_K.txt)')
    pose_group.add_argument('--camera-section', default="LEFT_CAM_FHD1200",
                        help='Camera section to use from intrinsics file (default: LEFT_CAM_FHD1200)')
    pose_group.add_argument('--est-refine-iter', type=int, default=5,
                        help='Number of estimation refinement iterations')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-dir', default="output",
                        help='Output directory to save images and results')
    output_group.add_argument('--publish-nt', action='store_true',
                        help='Publish pose to NetworkTables')
    output_group.add_argument('--nt-server', default=None,
                        help='NetworkTables server address (default: use the same as HTTP URL)')
    output_group.add_argument('--vision-table', default="Vision",
                        help='NetworkTables vision table name (default: Vision)')
    output_group.add_argument('--nt-entry', default="FoundationPose",
                        help='NetworkTables entry name (default: FoundationPose)')
    
    # Debug options
    debug_group = parser.add_argument_group('Debug Options')
    debug_group.add_argument('--debug', type=int, default=2,
                        help='Debug level (0-3)')
    debug_group.add_argument('--visualize', action='store_true',
                        help='Show visualizations during processing')
    debug_group.add_argument('--stop-on-error', action='store_true',
                        help='Stop pipeline when a processor fails')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    debug_dir = os.path.join(args.output_dir, "debug")
    
    # Create pipeline
    pipeline = Pipeline("FoundationPose Pipeline")
    
    # Create data container
    data = PipelineData()
    data.output_dir = args.output_dir
    
    # Add input processor
    if args.use_file:
        if not args.rgb_file:
            print("❌ Error: --rgb-file is required when using --use-file")
            return 1
            
        pipeline.add_processor(FileImageLoader(
            rgb_path=args.rgb_file,
            depth_path=args.depth_file,
            depth_npy_path=args.depth_npy_path
        ))
    else:
        pipeline.add_processor(HttpImageFetcher(
            base_url=args.url,
            cache_dir=os.path.join(args.output_dir, "cache")
        ))
    
    # Add detection/segmentation processors
    if args.manual_mask:
        pipeline.add_processor(ManualMasker(
            debug_dir=debug_dir
        ))
    else:
        pipeline.add_processor(GroundingDinoDetector(
            text_prompt=args.prompt,
            confidence_threshold=args.confidence,
            box_threshold=args.box_threshold,
            debug_dir=debug_dir
        ))
        
        pipeline.add_processor(SAM2Segmenter(
            debug_dir=debug_dir
        ))
    
    # Add pose estimation processors
    pipeline.add_processor(FoundationPoseEstimator(
        mesh_file=args.mesh_file,
        camera_intrinsics_file=args.cam_intrinsics,
        camera_section=args.camera_section,
        est_refine_iter=args.est_refine_iter,
        debug_level=args.debug,
        debug_dir=debug_dir
    ))

    pipeline.add_processor(PoseTransformer(
        to_inches=True,
        to_degrees=True
    ))
    
    # Add output processors
    pipeline.add_processor(ResultSaver(
        output_dir=args.output_dir,
        save_images=True,
        save_pose=True,
        save_visualizations=True
    ))
    
    if args.publish_nt:
        pipeline.add_processor(NetworkTablesPublisher(
            server_url=args.nt_server,
            vision_table=args.vision_table,
            entry_name=args.nt_entry
        ))
    
    # Run the pipeline
    try:
        print(f"🚀 Running FoundationPose Pipeline...")
        result = pipeline.run(data, stop_on_error=args.stop_on_error, visualize=args.visualize)
        
        # Check for errors
        if result.errors:
            print("⚠️ Errors occurred during processing:")
            for error in result.errors:
                print(f"  - {error['module']}: {error['message']}")
        
        # Show final result
        if result.pose_6d:
            x, y, z, roll, pitch, yaw = result.pose_6d
            print(f"\n✅ Final 6D Pose:")
            print(f"  Position (inches): x={x:.4f}, y={y:.4f}, z={z:.4f}")
            print(f"  Rotation (degrees): roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")
            
            # Format for NetworkTables
            pose_str = ",".join(map(str, result.pose_6d))
            print(f"  NetworkTables format: {pose_str}")
        
        # Create and display combined visualization
        vis = create_pipeline_visualization(result)
        if vis is not None:
            cv2.imshow("FoundationPose Pipeline Results", vis)
            cv2.imwrite(os.path.join(args.output_dir, "pipeline_result.png"), vis)
            print("Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("✅ FoundationPose Pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user. Exiting...")
        cv2.destroyAllWindows()
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        return 1

if __name__ == "__main__":
    sys.exit(main())