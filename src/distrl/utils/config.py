import yaml
import os

def load_config(config_path="config.yaml"):
    """
    Loads configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Config:
    """
    Static class to hold configuration.
    """
    _config = None

    @classmethod
    def get(cls, key=None, default=None):
        if cls._config is None:
            cls._config = load_config()
        
        if key is None:
            return cls._config
        
        return cls._config.get(key, default)
