from common.path import ProjectPath
import numpy as np

project_path = ProjectPath()


def save_np_array(array, location):
    location = project_path.format_location(location)
    np.savetxt(location, array, fmt='%s')
