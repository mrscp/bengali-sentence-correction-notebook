from processors.data import RawData
from processors.utils import Normalize, Perturbed
from common.config import Config


class Main:
    def __init__(self):
        self.__config = Config()

        if self.__config["GENERAL"]["MODE"] == "process-data":
            raw_data = RawData()
            lines, vocabulary = raw_data.get_data()

            normalize = Normalize()
            data, dictionary, reversed_dictionary, word_frequency = normalize.build_dictionary(
                lines,
                vocabulary,
                int(self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"])
            )

            print("vocabulary size {}/{}".format(len(dictionary), self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"]))

            perturbed = Perturbed()
            perturbed.build_dataset(data)
            print(perturbed.get_data())


if __name__ == '__main__':
    Main()
