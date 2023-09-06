import os
import yaml
from collections.abc import Mapping
class Config(object):
    DEBUG = False
    MODELS_FOLDER = os.environ.get("MODELS_FOLDER")

class ProductionConfig(Config):
    DEBUG = False

class DevelopmentConfig(Config):
    ENV = "development"
    DEVELOPMENT = True

def read_config_file(config):
    config_file = yaml.safe_load(config)

    with open('./config_default.yaml') as file:
        default_config_file = yaml.load(file, Loader=yaml.FullLoader)
    
    for key, value in config_file.items():
        update_dict(key, value, default_config_file)

    print(default_config_file)

    return default_config_file

def update_dict(key, value, d):
    if isinstance(value, Mapping):
        for k, v in value.items():
            update_dict(k, v, d[key])
    else:
        d[key] = value
