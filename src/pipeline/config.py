import yaml
import os

class PipelineConfig:
    """Configuration manager for pipeline components."""
    
    def __init__(self, config_file=None):
        self.config = {}
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, file_path):
        """Load configuration from a YAML file."""
        try:
            with open(file_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            self.config = {}
    
    def get(self, key, default=None):
        """Get a configuration value by key.
        
        Args:
            key: Key to look up, can use dot notation for nested keys (e.g., 'http.timeout')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """Set a configuration value.
        
        Args:
            key: Key to set, can use dot notation for nested keys
            value: Value to set
        """
        keys = key.split('.')
        last_key = keys.pop()
        
        # Navigate to the correct nested dictionary
        current = self.config
        for k in keys:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[last_key] = value
    
    def save(self, file_path):
        """Save configuration to a file.
        
        Args:
            file_path: Path to save the configuration
        """
        try:
            with open(file_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving config file: {str(e)}")