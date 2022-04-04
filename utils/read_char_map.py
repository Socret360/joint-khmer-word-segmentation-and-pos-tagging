import os
from typing import List


def read_char_map(char_map_filepath) -> List[str]:
    """ Read characters from `char_map_filepath` into a python list.

    Args
    ---
    char_map_filepath: A string representing path to the characters map file.

    Returns
    ---
    A python list with an extra "UNK" char for unkown character.

    Raises
    ---
    `char_map_filepath` does not exist.

    """
    assert os.path.exists, "char_map_filepath does not exist."

    with open(char_map_filepath, "r") as file:
        char_map = [i.strip("\n").split("\t")[0] for i in file.readlines()]
        char_map += ["UNK"]
        return char_map
