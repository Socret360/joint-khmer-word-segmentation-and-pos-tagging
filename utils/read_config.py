import os
import json
from typing import Dict


def read_config(config_filepath) -> Dict[str, str]:
    """ Read characters from `filepath` into a python list.

    Args
    ---
    config_filepath: A string representing path to the json config file.

    Returns
    ---
    A python dict representing the json config file.

    Raises
    ---
    - `config_filepath` does not exist.
    - `batch_size` must be larger than 0.
    - `learning_rate` must be larger than 0.

    """
    assert os.path.exists(config_filepath), "config_filepath does not exist."
    with open(config_filepath, "r") as config_file:
        config = json.load(config_file)
    assert config["training"]["batch_size"] > 0, "batch_size must be larger than 0"
    assert config["training"]["learning_rate"] > 0, "learning_rate must be larger than 0"

    return config
