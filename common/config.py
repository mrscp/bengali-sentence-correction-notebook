from configparser import ConfigParser
from os.path import join
import os
import sys


class Config(ConfigParser):
    def __init__(self, project_dir=None):
        super().__init__()
        # project_dir = 'content/gdrive/My Drive/ColabData/bengali-sentence-correction-notebook'

        if project_dir:
            self.project_dir = join("/", *project_dir.split("/"))
        else:
            self.project_dir = os.path.dirname(sys.modules['__main__'].__file__)

        self.__config_location = join(self.project_dir, "config.ini")
        self.read(self.__config_location)

    def save(self):
        with open(self.__config_location, "w") as file:
            self.write(file)
