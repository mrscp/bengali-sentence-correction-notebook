from common.config import Config
from common.path import ProjectPath
from os.path import join


class RawData:
    def __init__(self):
        self.__config = Config()
        self.__path = ProjectPath()

    def get_data(self):
        data_location = self.__path.format_location(
            join(
                self.__config["PROCESS_DATA"]["DATA_LOCATION"],
                self.__config["PROCESS_DATA"]["FILENAME"]
            )
        )
        i = 0
        vocabulary = []
        lines = []
        with open(data_location, "r") as file:
            for line in file:
                if i > int(self.__config["PROCESS_DATA"]["LIMIT"]):
                    break
                wss = line.strip()
                ws = wss.split(" ")

                vocabulary.extend(ws)
                lines.append(ws)

                i += 1
                if i % int(self.__config["PROCESS_DATA"]["REPORT_POINT"]) == 0:
                    print("{} lines processed".format(i))

        return lines, vocabulary

