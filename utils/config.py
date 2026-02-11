"""
Configuration utilities
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values

    Args:
        config: Original configuration
        updates: Updates to apply

    Returns:
        Updated configuration
    """
    import copy
    config = copy.deepcopy(config)

    def _update_recursive(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                d[k] = _update_recursive(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return _update_recursive(config, updates)
