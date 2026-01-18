# src/config.py

# libraries
import os
import toml
from types import MappingProxyType


# project path
project_path: str = os.path.dirname(os.path.abspath(__file__))

# config path
config_path: str = os.path.join(project_path, "config.toml")


# recursively load items to make them immutable
def _load_config_recursively(data: dict):
    """
    Helper function to make nested dictionaries immutable (read-only).
    Example: config['paths'] is read-only, but
    config['paths']['videos'] would still be mutable. Thus, we recursively load it to protect it.
    """
    if isinstance(data, dict):
        return MappingProxyType({k: _load_config_recursively(v) for k, v in data.items()})
    return data


# load the raw config file
with open(config_path, "r") as config_file:
    _raw_config: dict = toml.load(config_file)

# load mapping proxy type config
config: MappingProxyType = _load_config_recursively(_raw_config)
