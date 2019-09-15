from common.config import Config
from processors.utils import ProcessData
from dl.seq2seq import SequenceToSequence, WordRelation
from common.files import load_np_array
import numpy as np
from common.path import ProjectPath


class Main:
    def __init__(self):
        self.__config = Config()
        self.__project_path = ProjectPath()

        model_path = self.__project_path.format_location(
                    "{}/{}".format(
                        self.__config["TRAIN"]["MODEL_LOCATION"],
                        "{}.h5".format(self.__config["GENERAL"]["MODEL"])
                    )
                )

        if self.__config["GENERAL"]["MODE"] == "process-data":
            ProcessData()

        if self.__config["GENERAL"]["MODE"] == "train":
            if self.__config["GENERAL"]["MODEL"] == "SEQ2SEQ":
                model = SequenceToSequence(
                    int(self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"]),
                    int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"])
                )
            else:
                model = WordRelation()

            model.train()
            model.save_weights(model_path)

        if self.__config["GENERAL"]["MODE"] == "test":
            if self.__config["GENERAL"]["MODEL"] == "SEQ2SEQ":
                model = SequenceToSequence(
                    int(self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"]),
                    int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"])
                )
            else:
                model = WordRelation()

            model.test()


if __name__ == '__main__':
    Main()
