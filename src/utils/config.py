import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_dir: str = "config") -> dict[str, Any]:
    """Loads base.yaml, strategies.yaml, and models.yaml and merges them."""
    # Find project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    config_folder = os.path.join(project_root, config_dir)

    config = {}

    # 1. Load Base
    with open(os.path.join(config_folder, "base.yaml")) as f:
        config.update(yaml.safe_load(f))

    # 2. Load Strategies
    with open(os.path.join(config_folder, "strategies.yaml")) as f:
        config["strategy_config"] = yaml.safe_load(f)

    # 3. Load Models
    with open(os.path.join(config_folder, "models.yaml")) as f:
        config["model_config"] = yaml.safe_load(f)

    # Inject project root for absolute paths
    config["paths"]["project_root"] = project_root

    return config
