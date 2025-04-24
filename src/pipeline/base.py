class Processor:
    """Base interface for all pipeline processors."""
    
    def process(self, data):
        """Process the data and return modified data.
        
        Args:
            data: PipelineData object containing all pipeline data
            
        Returns:
            Modified PipelineData object
        """
        raise NotImplementedError
        
    def visualize(self, data):
        """Optional visualization for debugging.
        
        Args:
            data: PipelineData object to visualize
        """
        pass

class Pipeline:
    """Manages the flow of data through processors."""
    
    def __init__(self, name="Pipeline"):
        self.processors = []
        self.name = name
        
    def add_processor(self, processor):
        """Add a processor to the pipeline.
        
        Args:
            processor: Processor object to add
            
        Returns:
            self (for method chaining)
        """
        self.processors.append(processor)
        return self  # For method chaining
        
    def run(self, data, stop_on_error=False, visualize=False):
        """Execute the pipeline on input data.
        
        Args:
            data: PipelineData object to process
            stop_on_error: If True, pipeline will stop when a processor fails
            visualize: If True, processors will visualize their output
            
        Returns:
            Processed PipelineData object
        """
        print(f"Running pipeline: {self.name}")
        for idx, processor in enumerate(self.processors):
            processor_name = processor.__class__.__name__
            print(f"Step {idx+1}: Running {processor_name}...")
            
            try:
                data = processor.process(data)
                if visualize:
                    processor.visualize(data)
            except Exception as e:
                data.add_error(processor_name, str(e))
                print(f"Error in {processor_name}: {str(e)}")
                if stop_on_error:
                    print("Pipeline stopped due to error")
                    break
        
        return data