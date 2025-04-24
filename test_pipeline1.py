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

# TODO: Add processors in the order you want them to execute
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