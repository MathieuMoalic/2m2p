import os
import configparser
import itertools

import numpy as np
from appdirs import user_config_dir


def get_config() -> configparser.ConfigParser:
    config_dir = user_config_dir("llyr")
    if os.path.isfile("llyr.ini"):
        config_path = "llyr.ini"
    elif os.path.isfile(f"{config_dir}/llyr.ini"):
        config_path = f"{config_dir}/llyr.ini"
    config = configparser.ConfigParser()
    config.read(config_path)
    return config["llyr"]


def get_shape(arr: np.ndarray) -> dict:
    arr = np.ma.masked_greater_equal(arr, 0)
    mask = arr.mask  # pylint: disable=no-member
    x_max = 0
    xi_max = 0
    for i, row in enumerate(mask):
        groups = [list(group) for _, group in itertools.groupby(row)]
        if len(groups) > 3:
            return
        elif len(groups) == 3:
            if len(groups[1]) > x_max:
                x_max = len(groups[1])
                xi_max = i
    y_max = 0
    yi_max = 0
    for i, col in enumerate(mask.T):
        groups = [list(group) for _, group in itertools.groupby(col)]
        if len(groups) > 3:
            return
        elif len(groups) == 3:
            if len(groups[1]) > y_max:
                y_max = len(groups[1])
                yi_max = i
    if y_max == 0:
        return

    return {"xi_max": xi_max, "x_max": x_max, "yi_max": yi_max, "y_max": y_max}
