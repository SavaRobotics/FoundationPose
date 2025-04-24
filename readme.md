# FoundationPose Pipeline

A modular pipeline for 3D pose estimation using FoundationPose with support for various detection, segmentation, and output methods.

## Features

- Modular pipeline architecture with swappable components
- Support for both HTTP and file-based input
- Multiple detection and segmentation options:
  - Grounding DINO for object detection
  - SAM2 for instance segmentation
  - Manual mask creation
- Pose estimation using FoundationPose
- Output to multiple destinations:
  - Local files
  - NetworkTables for integration with robotics systems
- Extensive visualization and debugging capabilities

## Installation

### Environment Setup Options

#### Option 1: Docker (Recommended)
```bash
cd docker/
docker build -t foundationpose .
bash docker/run_container.sh

# Inside the container, build extensions
bash build_all.sh

# Later you can execute into the container without re-build
docker exec -it foundationpose bash
```

#### Option 2: Conda (Experimental)
```bash
# Create conda environment
conda create -n foundationpose python=3.9
conda activate foundationpose

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

# Install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Install additional dependencies
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```

### Required Dependencies

```bash
# Install SAM2 for segmentation
pip install ultralytics==8.2.70
```

## Usage

### Basic Usage (HTTP Input)
```bash
python main.py --url http://camera-server:8080 --mesh-file data/models/part.obj --cam-intrinsics data/camera_intrinsics.txt
```

### Using Local Files
```bash
python main.py --use-file --rgb-file data/samples/rgb.png --depth-file data/samples/depth.png --mesh-file data/models/part.obj --cam-intrinsics data/camera_intrinsics.txt
```

### Using Manual Masking
```bash
python main.py --url http://camera-server:8080 --manual-mask --mesh-file data/models/part.obj --cam-intrinsics data/camera_intrinsics.txt
```

### Publishing to NetworkTables
```bash
python main.py --url http://camera-server:8080 --publish-nt --nt-server roborio-team-frc.local
```

## Command-Line Options

### Input Options
- `--url`: Base URL of the HTTP server (default: http://localhost:8080)
- `--use-file`: Use local files instead of HTTP
- `--rgb-file`: Path to RGB image file
- `--depth-file`: Path to depth image file
- `--depth-npy-file`: Path to depth NPY file (for better precision)

### Detection/Segmentation Options
- `--manual-mask`: Use manual masking instead of automatic detection/segmentation
- `--prompt`: Text prompt for Grounding DINO detection (default: "bent sheet metal")
- `--confidence`: Confidence threshold for detection (default: 0.25)
- `--box-threshold`: Box threshold for detection (default: 0.2)

### Pose Estimation Options
- `--mesh-file`: Path to the CAD model file (.obj)
- `--cam-intrinsics`: Path to camera intrinsics file
- `--camera-section`: Camera section to use from intrinsics file (default: LEFT_CAM_FHD1200)
- `--est-refine-iter`: Number of estimation refinement iterations (default: 5)

### Output Options
- `--output-dir`: Output directory for results (default: "output")
- `--publish-nt`: Publish pose to NetworkTables
- `--nt-server`: NetworkTables server address
- `--vision-table`: NetworkTables vision table name (default: "Vision")
- `--nt-entry`: NetworkTables entry name (default: "FoundationPose")

### Debug Options
- `--debug`: Debug level (0-3)
- `--visualize`: Show visualizations during processing
- `--stop-on-error`: Stop pipeline when a processor fails

## Creating Custom Pipelines

You can create custom pipelines by creating a new script and importing the modules you need:

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
pipeline = Pipeline("Custom Pipeline")

# Add processors in the order you want them to execute
pipeline.add_processor(HttpImageFetcher("http://example.com"))
pipeline.add_processor(GroundingDinoDetector("object name"))
pipeline.add_processor(SAM2Segmenter())
pipeline.add_processor(FoundationPoseEstimator("model.obj", "camera.txt"))
pipeline.add_processor(PoseTransformer())
pipeline.add_processor(NetworkTablesPublisher())

# Create data container
data = PipelineData()

# Run pipeline
result = pipeline.run(data)

# Use results
print(result.pose_6d)
```

## Development

### Adding a New Processor

To create a new processor, subclass the `Processor` base class:

```python
from src.pipeline.base import Processor

class MyCustomProcessor(Processor):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def process(self, data):
        # Process the data
        # ...
        return data
    
    def visualize(self, data):
        # Visualize the results (optional)
        # ...
```

Then add it to your pipeline:

```python
pipeline.add_processor(MyCustomProcessor(param1, param2))
```