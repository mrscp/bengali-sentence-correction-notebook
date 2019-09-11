from configparser import ConfigParser
from os.path import join
import os
import sys


class Config(ConfigParser):
    def __init__(self, project_dir=None):
        super().__init__()

        if project_dir:
            self.__project_dir = join("/", *project_dir.split("/"))
        else:
            self.__project_dir = os.path.dirname(sys.modules['__main__'].__file__)

        self.__config_location = join(self.__project_dir, "config.ini")
        self.read(self.__config_location)

    def save(self):
        with open(self.__config_location, "w") as file:
            self.write(file)
