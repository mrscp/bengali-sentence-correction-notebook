from common.path import ProjectPath
import numpy as np

project = ProjectPath()


def format_location(location):
    return project.format_location(location)


def project_path():
    return project.get_project_path()


def save_np_array(array, location):
    location = format_location(location)
    np.savetxt(location, array, fmt='%s')


def load_np_array(location, dtype=None):
    location = format_location(location)
    if dtype:
        return np.loadtxt(location, dtype=dtype)

    return np.loadtxt(location)


def append_np_array(array, location):
    with open(format_location(location), "ab") as file:
        np.savetxt(file, array, fmt='%s')
