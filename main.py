from processors.data import RawData
from processors.utils import Normalize, Perturbed
from common.config import Config
from common.files import save_np_array


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
            print(dictionary)

            perturbed = Perturbed()
            perturbed.build_dataset(data)
            train_x, train_y, test_x, test_y = perturbed.get_data()
            print(train_x[:5])
            save_np_array(train_x, "{}/correct.txt".format(self.__config["TRAIN"]["DATA_LOCATION"]))
            save_np_array(train_y, "{}/incorrect.txt".format(self.__config["TRAIN"]["DATA_LOCATION"]))
            save_np_array(test_x, "{}/correct.txt".format(self.__config["TEST"]["DATA_LOCATION"]))
            save_np_array(test_y, "{}/incorrect.txt".format(self.__config["TEST"]["DATA_LOCATION"]))


if __name__ == '__main__':
    Main()
