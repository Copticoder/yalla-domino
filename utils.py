import json


def parse_config(config_json_file):
    """
    Parse annotation config
    """
    with open(config_json_file, 'r') as f:
        configs = json.load(f)
    return configs
