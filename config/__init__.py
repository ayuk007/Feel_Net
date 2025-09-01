# config/__init__.py
import yaml
import os
from pathlib import Path

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        # Get the directory of this file and look for config.yaml
        config_dir = Path(__file__).parent
        config_path = config_dir / "config.yaml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Create directories if they don't exist
    os.makedirs(config['paths']['checkpoints_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    os.makedirs(config['paths']['tensorboard_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['data']['raw_data_path']), exist_ok=True)
    
    return config

class Config:
    """Configuration class for easy access to config parameters."""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __contains__(self, key):
        return hasattr(self, key)