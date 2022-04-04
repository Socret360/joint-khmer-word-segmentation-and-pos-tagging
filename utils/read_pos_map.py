import os
from typing import List


def read_pos_map(pos_map_filepath) -> List[str]:
    """ Read POS tags from `pos_map_filepath` into a python list.

    Args
    ---
    pos_map_filepath: A string representing path to the pos map file.

    Returns
    ---
    A python list.

    Raises
    ---
    `pos_map_filepath` does not exist.

    """
    assert os.path.exists, "pos_map_filepath does not exist."
    with open(pos_map_filepath, "r") as file:
        pos_map = [i.strip("\n").split("\t")[0] for i in file.readlines()]
        return pos_map
