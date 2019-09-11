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

        print(data_location)
        # with open(, "r") as file:
        #     for line in file:
        #         if i > config["max_line_read"]:
        #             break
        #         wss = line.strip()
        #         ws = wss.split(" ")
        #
        #         vocabulary.extend(ws)
        #         lines.append(ws)
        #
        #         i += 1
        #         if i % report_point == 0:
        #             print("{} lines processed".format(i))