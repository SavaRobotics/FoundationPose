import logging
import sys
import time
from typing import List, Optional, Any, Dict
from .processor import BaseProcessor

class Processor(BaseProcessor):
    """Legacy interface for backward compatibility."""
    pass

class Pipeline:
    """Pipeline for processing data through a series of processors."""
    
    def __init__(self, name="Pipeline"):
        self.name = name
        self.processors = []
        self._setup_logging()
        self.logger = logging.getLogger(f"Pipeline.{name}")
        
    def _setup_logging(self):
        """Set up logging for the pipeline and all processors."""
        # Create root logger
        root_logger = logging.getLogger()
        
        # Remove any existing handlers to avoid duplicate logs
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        # Create console handler with formatting
        console = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s')
        console.setFormatter(formatter)
        
        # Add handler to root logger
        root_logger.addHandler(console)
        root_logger.setLevel(logging.INFO)
    
    def add_processor(self, processor):
        """Add a processor to the pipeline."""
        self.processors.append(processor)
        return self
    
    def run(self, data, stop_on_error=False, visualize=False):
        """Run the pipeline on the given data."""
        start_time = time.time()
        self.logger.info(f"🚀 Starting {self.name} with {len(self.processors)} processors")
        
        # Process data through each processor in sequence
        for i, processor in enumerate(self.processors):
            processor_name = processor.__class__.__name__
            step_start = time.time()
            
            self.logger.info(f"⏳ [{i+1}/{len(self.processors)}] Running {processor_name}...")
            
            try:
                # Process the data
                data = processor.process(data)
                
                # Check for errors
                if data.has_errors():
                    errors = data.get_errors()
                    for error in errors:
                        self.logger.error(f"⚠️ Error in {error['module']}: {error['message']}")
                    
                    if stop_on_error:
                        self.logger.error(f"❌ Pipeline stopped due to errors in {processor_name}")
                        break
                
                # Optional visualization
                if visualize:
                    processor.visualize(data)
                
                step_time = time.time() - step_start
                self.logger.info(f"✅ Completed {processor_name} in {step_time:.2f}s")
                
            except Exception as e:
                import traceback
                self.logger.error(f"❌ Error in {processor_name}: {str(e)}")
                self.logger.debug(traceback.format_exc())
                
                if stop_on_error:
                    self.logger.error(f"❌ Pipeline stopped due to exception in {processor_name}")
                    break
        
        total_time = time.time() - start_time
        self.logger.info(f"🏁 Pipeline completed in {total_time:.2f}s")
        
        return data