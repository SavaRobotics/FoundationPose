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
    server_url="https://6d25-162-218-227-129.ngrok-free.app"  # NT server address (optional, defaults to NT_SERVER env var or 'localhost')
)

# Direct publishing method
pose_6d = [1.0, 2.0, 3.0, 45.0, 0.0, 90.0]  # x, y, z, roll, pitch, yaw
nt_publisher.publish_pose(pose_6d)
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

## Logging System

The Foundation Pose pipeline includes a comprehensive logging system to provide detailed information about each stage of processing.

### Basic Logging Configuration

```python
import logging

# Configure logging in your application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Create a logger for your module
logger = logging.getLogger("YourModuleName")
```

### Pipeline Logging

The Pipeline class automatically configures logging for all processors. Each component's logs will include:
- Start and end of processing
- Processing time
- Errors and warnings
- Status of each step

### Creating Custom Processors with Logging

To create a custom processor with integrated logging, extend the BaseProcessor class:

```python
from src.pipeline.processor import BaseProcessor

class YourCustomProcessor(BaseProcessor):
    def __init__(self, your_param, debug_dir=None):
        # Always call the parent constructor first
        super().__init__(debug_dir)
        self.your_param = your_param
        
    def process(self, data):
        # Log the start of processing
        self.log_step_start("your custom process")
        
        # Your processing logic
        # ...
        
        # Log any errors
        if error_condition:
            self.log_error("Something went wrong", data)
            return data
            
        # Log successful completion
        self.log_step_complete("your custom process")
        return data
```

### Logging Methods

The BaseProcessor class provides these logging methods:

- `self.logger.info(message)` - Log general information
- `self.logger.warning(message)` - Log warnings
- `self.logger.error(message)` - Log errors
- `self.logger.debug(message)` - Log detailed debugging information
- `self.log_step_start(step_name)` - Log the start of a processing step
- `self.log_step_complete(step_name)` - Log the completion of a processing step
- `self.log_error(message, data)` - Log an error and add it to the data object

These logs follow a consistent format with timestamps, module names, and log levels, making it easy to track the pipeline's progress.

## Creating New Middleware Components

To accelerate development of new middleware components, you can use the included template.

### Processor Template

The Foundation Pose pipeline includes a template for creating new processors. To use it:

1. Copy the template file `src/pipeline/template.py`
2. Rename the class and file to match your component
3. Implement the required functionality
4. Add your processor to the pipeline

```python
# Example of extending the template
from src.pipeline.template import YourProcessorName

# Copy the template and rename
class MyCustomProcessor(YourProcessorName):
    def __init__(self, my_param, debug_dir=None):
        super().__init__(my_param, None, debug_dir)
        
    def process(self, data):
        # Custom implementation
        # ...
        return data
```

### Best Practices for New Components

When creating new middleware components:

1. **Use the BaseProcessor class**: All processors should extend `BaseProcessor` to inherit logging functionality.
2. **Log each step**: Use `self.log_step_start()` and `self.log_step_complete()` to track progress.
3. **Handle errors properly**: Use `self.log_error()` to record and report errors.
4. **Create visualizations**: Implement the `visualize()` method for debugging.
5. **Use debug directories**: Support the `debug_dir` parameter for saving intermediate results.
6. **Follow the existing patterns**: Maintain consistency with other components.

For more examples, examine the implementation of existing processors in the `src/modules/` directory.

## Camera Intrinsics

The FoundationPose pipeline supports two formats for camera intrinsics:

### Matrix Format

The simplest format is a 3x3 matrix:

```
fx 0  cx
0  fy cy
0  0  1
```

For example:
```
735.327 0 978.117
0 735.104 576.618
0 0 1
```

### Parameter-Based Format

For multi-camera setups like the ZED X Mini, the system supports a parameter-based format with camera sections:

```
[LEFT_CAM_FHD1200]
fx=735.327
fy=735.104
cx=978.117
cy=576.618
k1=-0.0166323
k2=-0.0273565
p1=7.04243e-05
p2=2.09564e-05
k3=0.00695552

[RIGHT_CAM_FHD1200]
fx=736.339
fy=735.835
cx=976.348
cy=576.689
k1=-0.0158105
k2=-0.0276155
p1=-6.59245e-05
p2=3.80654e-06
k3=0.006958
```

When using this format, specify the camera section name to use:

```python
pose_estimator = FoundationPoseEstimator(
    mesh_file="model.obj",
    camera_intrinsics_file="data/intrinsics/cam_K.txt",
    camera_section="LEFT_CAM_FHD1200"  # Specify which camera section to use
)
```

### Testing Camera Intrinsics Loading

To verify if your camera intrinsics file can be loaded properly, use the testing utility:

```bash
# Test with default camera section (LEFT_CAM_FHD1200)
python -m src.utils.tests.test_cam_intrinsics data/intrinsics/cam_K.txt

# Test with specific camera section
python -m src.utils.tests.test_cam_intrinsics data/intrinsics/cam_K.txt LEFT_CAM_SVGA
```

This tool will show you all available camera sections and confirm that the file can be loaded correctly.