import os
import yaml

# read configurations from config file
def get_config(filename=None):
    """
    Creates configuration variables for model setup
    Args:
        filename    - name/path of yaml file 
                    (defaults to None)
    Returns:
        dictionary  - containing model configurations
    """
    if filename:
        yaml_file = filename
    else:
        yaml_file = 'config.yaml'

    with open(yaml_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config