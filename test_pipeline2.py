import cv2
import os
from src.pipeline.base import Pipeline
from src.pipeline.data import PipelineData
from src.modules.input.file_loader import FileImageLoader
from src.modules.segmentation.manual_masking import ManualMasker
from src.modules.pose.foundation_pose import FoundationPoseEstimator
from src.modules.pose.transform import PoseTransformer
from src.modules.output.file_saver import ResultSaver
from src.utils.visualization import create_pipeline_visualization

# Create the pipeline
pipeline = Pipeline("Manual Masking Pipeline")

# Define paths
rgb_file = "data/samples/batch1/rgb/000000.png"
depth_file = "data/samples/batch1/depth/000000.npy"  # or .png
model_file = "data/models/bent_test_1.obj"
intrinsics_file = "data/cam_K.txt"
output_dir = "output/my_test"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Add processors
pipeline.add_processor(FileImageLoader(rgb_file, depth_file))
pipeline.add_processor(ManualMasker())
pipeline.add_processor(FoundationPoseEstimator(model_file, intrinsics_file))
pipeline.add_processor(PoseTransformer())
pipeline.add_processor(ResultSaver(output_dir))

# Run the pipeline
data = PipelineData()
result = pipeline.run(data, visualize=True)  # This shows visualizations at each step

# Print result
if result.pose_6d:
    x, y, z, roll, pitch, yaw = result.pose_6d
    print(f"\n✅ Final 6D Pose:")
    print(f"  Position (inches): x={x:.4f}, y={y:.4f}, z={z:.4f}")
    print(f"  Rotation (degrees): roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")

# Create and show a final combined visualization
final_vis = create_pipeline_visualization(result)
if final_vis is not None:
    cv2.imshow("Pipeline Results", final_vis)
    cv2.imwrite(os.path.join(output_dir, "pipeline_result.png"), final_vis)
    print("\nPipeline visualization saved to output directory.")
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()