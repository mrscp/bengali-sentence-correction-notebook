from common.path import ProjectPath
import numpy as np

project_path = ProjectPath()


def save_np_array(array, location):
    location = project_path.format_location(location)
    np.savetxt(location, array, fmt='%s')


def load_np_array(location, dtype=None):
    location = project_path.format_location(location)
    if dtype:
        return np.loadtxt(location, dtype=dtype)

    return np.loadtxt(location)
