import json


def parse_config(config_json_file):
    """
    解析标注config
    """
    with open(config_json_file, 'r') as f:
        configs = json.load(f)
    return configs
