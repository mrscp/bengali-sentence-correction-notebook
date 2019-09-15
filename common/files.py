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


def list_to_file(data, location):
    location = format_location(location)
    with open(location, "w") as file:
        for line in data:
            file.write(" ".join(str(item) for item in line) + "\n")


def load_int_list(location):
    location = format_location(location)
    lists = []
    with open(location) as input_file:
        for line in input_file:
            line = line.strip().split(" ")
            line = [int(token) for token in line]
            lists.append(line)

    return lists
