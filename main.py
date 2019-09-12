from common.config import Config
from processors.utils import ProcessData
from dl.seq2seq import SequenceToSequence
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
                        self.__config["TRAIN"]["WEIGHTS_FILE_NAME"]
                    )
                )

        if self.__config["GENERAL"]["MODE"] == "process-data":
            ProcessData()

        if self.__config["GENERAL"]["MODE"] == "train":
            x = load_np_array("{}/correct.txt".format(self.__config["TRAIN"]["DATA_LOCATION"]))
            y = load_np_array("{}/incorrect.txt".format(self.__config["TRAIN"]["DATA_LOCATION"]))
            target = load_np_array("{}/target.txt".format(self.__config["TRAIN"]["DATA_LOCATION"]))

            target = target.reshape(
                -1,
                int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"]),
                int(self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"])
            )

            model = SequenceToSequence(
                int(self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"]),
                int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"])
            )
            model.fit((x, y), target, batch_size=16, epochs=int(self.__config["TRAIN"]["EPOCHS"]))

            model.save_weights(model_path)

        if self.__config["GENERAL"]["MODE"] == "test":
            x = load_np_array("{}/correct.txt".format(self.__config["TEST"]["DATA_LOCATION"]))
            y = load_np_array("{}/incorrect.txt".format(self.__config["TEST"]["DATA_LOCATION"]))
            target = load_np_array("{}/target.txt".format(self.__config["TEST"]["DATA_LOCATION"]))
            model = SequenceToSequence(
                int(self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"]),
                int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"])
            )

            model.load_weights(model_path)

            result = model.predict((x, x))
            for r, t, _y in zip(result, target, y):
                print(np.argmax(r, axis=1))
                print(_y)
                # print(np.argmax(t, axis=1))
                print()


if __name__ == '__main__':
    Main()
