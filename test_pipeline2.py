import os
import logging
import cv2
from src.pipeline.base import Pipeline
from src.pipeline.data import PipelineData
from src.modules.input.file_loader import FileImageLoader
from src.modules.segmentation.manual_masking import ManualMasker
from src.modules.pose.foundation_pose import FoundationPoseEstimator
from src.modules.pose.transform import PoseTransformer
from src.modules.output.file_saver import ResultSaver
from src.utils.visualization import create_pipeline_visualization

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestPipeline")

# Get the script directory for resolving paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the pipeline
pipeline = Pipeline("Manual Masking Pipeline")

# Define paths with validation
rgb_file = os.path.join(script_dir, "data/samples/batch1/rgb/000000.png")
if not os.path.exists(rgb_file):
    logger.error(f"ERROR: RGB file not found at {rgb_file}")
    raise FileNotFoundError(f"RGB file {rgb_file} not found")

depth_file = os.path.join(script_dir, "data/samples/batch1/depth/000000.npy")
if not os.path.exists(depth_file):
    depth_file_png = os.path.join(script_dir, "data/samples/batch1/depth/000000.png")
    if os.path.exists(depth_file_png):
        logger.info(f"Using PNG depth file instead: {depth_file_png}")
        depth_file = depth_file_png
    else:
        logger.error(f"ERROR: Depth file not found at {depth_file} or {depth_file_png}")
        raise FileNotFoundError(f"Depth file not found")

model_file = os.path.join(script_dir, "data/models/bent_test_1.obj")
if not os.path.exists(model_file):
    logger.error(f"ERROR: Model file not found at {model_file}")
    raise FileNotFoundError(f"Model file {model_file} not found")

intrinsics_file = os.path.join(script_dir, "data/intrinsics/cam_K.txt")
if not os.path.exists(intrinsics_file):
    logger.error(f"ERROR: Intrinsics file not found at {intrinsics_file}")
    raise FileNotFoundError(f"Intrinsics file {intrinsics_file} not found")

output_dir = os.path.join(script_dir, "output_dir/my_test")
os.makedirs(output_dir, exist_ok=True)

logger.info(f"Using RGB file: {rgb_file}")
logger.info(f"Using depth file: {depth_file}")
logger.info(f"Using model file: {model_file}")
logger.info(f"Using intrinsics file: {intrinsics_file}")
logger.info(f"Using output directory: {output_dir}")

# Add processors
pipeline.add_processor(FileImageLoader(rgb_file, None, depth_file))
pipeline.add_processor(ManualMasker())
pipeline.add_processor(FoundationPoseEstimator(model_file, intrinsics_file))
pipeline.add_processor(PoseTransformer())
pipeline.add_processor(ResultSaver(output_dir))

# Run the pipeline
logger.info("Starting pipeline execution")
data = PipelineData()
result = pipeline.run(data, visualize=True)  # This shows visualizations at each step

# Debug check for depth image
if result.depth_image is None:
    logger.error("❌ Depth image was not loaded correctly")
    logger.error("Check that the depth file exists and is in the correct format")
else:
    logger.info(f"✅ Depth image loaded successfully, shape: {result.depth_image.shape}")
    logger.debug(f"Min depth: {result.depth_image.min()}, Max depth: {result.depth_image.max()}")
    
# Print result
if result.pose_6d:
    x, y, z, roll, pitch, yaw = result.pose_6d
    logger.info(f"✅ Final 6D Pose:")
    logger.info(f"  Position (inches): x={x:.4f}, y={y:.4f}, z={z:.4f}")
    logger.info(f"  Rotation (degrees): roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")

# Create and show a final combined visualization
final_vis = create_pipeline_visualization(result)
if final_vis is not None:
    cv2.imshow("Pipeline Results", final_vis)
    cv2.imwrite(os.path.join(output_dir, "pipeline_result.png"), final_vis)
    logger.info("Pipeline visualization saved to output directory.")
    cv2.destroyAllWindows()