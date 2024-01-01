import sys

import yaml


def load_config():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "src/config.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
