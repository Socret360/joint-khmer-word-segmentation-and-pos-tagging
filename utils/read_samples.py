import os
from typing import List


def read_samples(split_filepath) -> List[str]:
    """ Read samples from `split_filepath` into a python list.

    Args
    ---
    split_filepath: A string representing path to the split_filepath file.

    Returns
    ---
    A python list.

    Raises
    ---
    `split_filepath` does not exist.
    """
    assert os.path.exists(split_filepath), "split_filepath does not exist."

    with open(split_filepath, "r") as dataset_file:
        samples = [i.strip("\n").strip("") for i in dataset_file.readlines()]
        return samples
