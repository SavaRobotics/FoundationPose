import os
import sys

# Add the project root to Python path 
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    
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

# Create the pipeline
pipeline = Pipeline("Manual Masking Pipeline")

# Define paths with validation - use absolute paths to avoid issues
rgb_file = os.path.join(script_dir, "data/samples/batch1/rgb/000000.png")
depth_file = os.path.join(script_dir, "data/samples/batch1/depth/000000.npy")
model_file = os.path.join(script_dir, "data/models/bent_test_1.obj")
intrinsics_file = os.path.join(script_dir, "data/intrinsics/cam_K.txt")
output_dir = os.path.join(script_dir, "output_dir/my_test")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Add processors
pipeline.add_processor(FileImageLoader(rgb_file, None, depth_file))
pipeline.add_processor(ManualMasker())
# If you want to temporarily skip the FoundationPoseEstimator for testing
# pipeline.add_processor(PoseTransformer())
# pipeline.add_processor(ResultSaver(output_dir))

# If you want to include the FoundationPoseEstimator with additional error handling
try:
    # Verify files exist before passing them
    for path, name in [(model_file, "Model"), (intrinsics_file, "Intrinsics")]:
        if not os.path.exists(path):
            logger.error(f"❌ {name} file not found: {path}")
            raise FileNotFoundError(f"{name} file {path} not found")
            
    # Try importing a key module to check if dependencies are installed
    try:
        from src.utils.estimater import FoundationPose
        logger.info("✅ FoundationPose modules are available")
    except ImportError as e:
        logger.error(f"❌ FoundationPose modules not available: {e}")
        logger.info("Installing required dependencies might help resolve this issue.")
        # You could raise an exception here to stop execution
        
    # Add the processor if everything is ok
    pipeline.add_processor(FoundationPoseEstimator(model_file, intrinsics_file))
    pipeline.add_processor(PoseTransformer())
    pipeline.add_processor(ResultSaver(output_dir))
except Exception as e:
    logger.error(f"❌ Error setting up FoundationPoseEstimator: {e}")
    # Optionally continue with a simplified pipeline
    pipeline.add_processor(ResultSaver(output_dir))

# Run the pipeline
logger.info("Starting pipeline execution")
data = PipelineData()
result = pipeline.run(data, visualize=True)

# Debug check for depth image
if result.depth_image is None:
    logger.error("❌ Depth image was not loaded correctly")
    logger.error("Check that the depth file exists and is in the correct format")
else:
    logger.info(f"✅ Depth image loaded successfully, shape: {result.depth_image.shape}")
    logger.debug(f"Min depth: {result.depth_image.min()}, Max depth: {result.depth_image.max()}")
    
# Print result
if hasattr(result, 'pose_6d') and result.pose_6d:
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