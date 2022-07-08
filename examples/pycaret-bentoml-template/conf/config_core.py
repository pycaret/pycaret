from pathlib import Path
from omegaconf import OmegaConf


def fetch_yaml_data(config_path: Path = Path('./conf/config.yaml')) -> dict:
    """
    This function reads and imports the data from the main
    config YAML file so taht it can be used by any function.
    """
    if config_path:
        return OmegaConf.load(config_path)
    
    raise OSError(f"Didn't find config file at the specified directory! ({config_path})")


config = fetch_yaml_data()