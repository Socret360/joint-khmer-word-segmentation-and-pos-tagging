import argparse


def str2bool(v) -> bool:
    """ Converts command line boolean statements to python boolean.

    Args
    ---
    - v: A string representing the command line boolean statements.

    Returns
    ---
    A python boolean.

    Raises
    ---
    - `v` does not match any command line boolean statements.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
