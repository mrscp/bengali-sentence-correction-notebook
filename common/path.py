from os.path import join
from common.config import Config


class ProjectPath:
    def __init__(self):
        self.__config = Config()

    def get_project_path(self):
        return self.__config.project_dir

    def format_location(self, location):
        return join(self.__config.project_dir, join(*location.split("/")))
