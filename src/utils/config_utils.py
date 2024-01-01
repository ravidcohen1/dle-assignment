import sys
from pathlib import Path

import yaml

DEFAULT_CONFIGS = Path(__file__).parent.parent / "config.yaml"


def load_config():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIGS
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
