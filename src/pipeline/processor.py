import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseProcessor(ABC):
    """Base class for all pipeline processors with integrated logging."""
    
    def __init__(self, debug_dir: Optional[str] = None):
        """Initialize the processor with debug directory and logger.
        
        Args:
            debug_dir: Directory for debug outputs
        """
        self.debug_dir = debug_dir
        
        # Set up logger with processor's class name
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def process(self, data):
        """Process the input data and return updated data.
        
        This method must be implemented by all subclasses.
        
        Args:
            data: The PipelineData object
            
        Returns:
            Updated PipelineData object
        """
        pass
        
    def visualize(self, data):
        """Visualize the results of processing.
        
        Default implementation does nothing. Subclasses can override.
        
        Args:
            data: The PipelineData object after processing
        """
        pass
    
    def log_step_start(self, step_name: str):
        """Log the start of a processing step.
        
        Args:
            step_name: Name of the step being started
        """
        self.logger.info(f"⏳ Starting {step_name}...")
    
    def log_step_complete(self, step_name: str):
        """Log the completion of a processing step.
        
        Args:
            step_name: Name of the completed step
        """
        self.logger.info(f"✅ Completed {step_name}")
    
    def log_error(self, message: str, data=None):
        """Log an error and add it to the data object if provided.
        
        Args:
            message: Error message
            data: Optional PipelineData object to add error to
        """
        self.logger.error(f"❌ {message}")
        if data:
            data.add_error(self.__class__.__name__, message) 