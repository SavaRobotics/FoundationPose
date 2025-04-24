# Foundation Pose Pipeline API Reference

## Core Components

### Pipeline Class

```python
from src.pipeline.base import Pipeline

# Create a pipeline
pipeline = Pipeline(name="My Pipeline")

# Add processors
pipeline.add_processor(processor)  # Returns self for chaining

# Run the pipeline
result = pipeline.run(data, stop_on_error=False, visualize=False)
```

Parameters:
- `name`: Optional name for the pipeline

### PipelineData Class

```python
from src.pipeline.data import PipelineData

# Create data container
data = PipelineData()

# Key attributes:
# data.rgb_image         - RGB image (numpy array)
# data.depth_image       - Depth image (numpy array)
# data.mask              - Binary mask (numpy array)
# data.detection_boxes   - List of bounding boxes [x1, y1, x2, y2, score]
# data.detection_labels  - List of detection labels 
# data.pose_matrix       - 4x4 pose matrix
# data.pose_6d           - [x, y, z, roll, pitch, yaw]
# data.output_dir        - Output directory path
# data.debug_images      - Dict of debug visualizations

# Add errors
data.add_error(module_name, error_message)
```

## Input Modules

### HTTP Image Fetcher

```python
from src.modules.input.http_client import HttpImageFetcher

# Fetch RGB and depth images from HTTP server
http_fetcher = HttpImageFetcher(
    base_url="http://camera-server:8080",
    timeout=5,                           # Request timeout in seconds
    cache_dir="data/cache"               # Optional directory to cache images
)
```

### File Image Loader

```python
from src.modules.input.file_loader import FileImageLoader

# Load images from files
file_loader = FileImageLoader(
    rgb_path="data/samples/rgb.png",
    depth_path="data/samples/depth.png",  # Optional
    depth_npy_path="data/samples/depth.npy"  # Optional, preferred over depth_path
)
```

## Detection Modules

### Grounding DINO Detector

```python
from src.modules.detection.ground_dino import GroundingDinoDetector

# Detect objects with text prompts
dino_detector = GroundingDinoDetector(
    text_prompt="bent sheet metal",
    confidence_threshold=0.25,           # Text conditioning threshold
    box_threshold=0.2,                   # Detection threshold
    debug_dir="output/debug"             # Optional directory for debug output
)
```

## Segmentation Modules

### SAM2 Segmenter

```python
from src.modules.segmentation.sam import SAM2Segmenter

# Segment objects using SAM2
sam_segmenter = SAM2Segmenter(
    debug_dir="output/debug"             # Optional directory for debug output
)
```

### Manual Masker

```python
from src.modules.segmentation.manual_masking import ManualMasker

# Create masks interactively
manual_masker = ManualMasker(
    brush_size=10,                       # Initial brush size
    debug_dir="output/debug"             # Optional directory for debug output
)
```

## Pose Estimation Modules

### FoundationPose Estimator

```python
from src.modules.pose.foundation_pose import FoundationPoseEstimator

# Estimate 3D pose using FoundationPose
pose_estimator = FoundationPoseEstimator(
    mesh_file="data/models/part.obj",    # CAD model file
    camera_intrinsics_file="data/cam_K.txt",  # Camera calibration
    camera_section="LEFT_CAM_FHD1200",   # Section in the intrinsics file
    est_refine_iter=5,                   # Number of refinement iterations
    debug_level=2,                       # Debug level (0-3)
    debug_dir="output/debug"             # Optional directory for debug output
)
```

### Pose Transformer

```python
from src.modules.pose.transform import PoseTransformer

# Transform pose matrix to 6D pose
pose_transformer = PoseTransformer(
    to_inches=True,                      # Convert meters to inches
    to_degrees=True                      # Convert radians to degrees
)
```

## Output Modules

### Result Saver

```python
from src.modules.output.file_saver import ResultSaver

# Save results to files
result_saver = ResultSaver(
    output_dir="output",                 # Output directory
    save_images=True,                    # Save RGB, depth, and mask
    save_pose=True,                      # Save pose matrices
    save_visualizations=True             # Save visualizations
)
```

### Network Tables Publisher

```python
from src.modules.output.network_tables import NetworkTablesPublisher

# Publish pose to NetworkTables
nt_publisher = NetworkTablesPublisher(
    server_url="roborio-team-frc.local", # NT server address
    vision_table="Vision",               # NT table name
    entry_name="FoundationPose",         # NT entry name
    timeout=10                           # Connection timeout
)
```

## Utility Functions

### Visualization

```python
from src.utils.visualization import create_pipeline_visualization

# Create a combined visualization of all pipeline stages
vis = create_pipeline_visualization(result)

# Show visualization
import cv2
cv2.imshow("Pipeline Results", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Example Pipelines

### Simple Manual Masking Pipeline

```python
from src.pipeline.base import Pipeline
from src.pipeline.data import PipelineData
from src.modules.input.file_loader import FileImageLoader
from src.modules.segmentation.manual_masking import ManualMasker
from src.modules.pose.foundation_pose import FoundationPoseEstimator
from src.modules.pose.transform import PoseTransformer
from src.modules.output.file_saver import ResultSaver

# Create pipeline
pipeline = Pipeline()

# Add processors
pipeline.add_processor(FileImageLoader("path/to/rgb.png", "path/to/depth.npy"))
pipeline.add_processor(ManualMasker())
pipeline.add_processor(FoundationPoseEstimator("path/to/model.obj", "path/to/camera.txt"))
pipeline.add_processor(PoseTransformer())
pipeline.add_processor(ResultSaver("output/dir"))

# Run pipeline
data = PipelineData()
result = pipeline.run(data, visualize=True)

# Use results
if result.pose_6d:
    print(f"Final pose: {result.pose_6d}")
```

### Automatic Detection Pipeline

```python
from src.pipeline.base import Pipeline
from src.pipeline.data import PipelineData
from src.modules.input.http_client import HttpImageFetcher
from src.modules.detection.ground_dino import GroundingDinoDetector
from src.modules.segmentation.sam import SAM2Segmenter
from src.modules.pose.foundation_pose import FoundationPoseEstimator
from src.modules.pose.transform import PoseTransformer
from src.modules.output.network_tables import NetworkTablesPublisher

# Create pipeline
pipeline = Pipeline()

# Add processors
pipeline.add_processor(HttpImageFetcher("http://camera:8080"))
pipeline.add_processor(GroundingDinoDetector("sheet metal part"))
pipeline.add_processor(SAM2Segmenter())
pipeline.add_processor(FoundationPoseEstimator("model.obj", "camera.txt"))
pipeline.add_processor(PoseTransformer())
pipeline.add_processor(NetworkTablesPublisher())

# Run pipeline
data = PipelineData()
result = pipeline.run(data)
```