import collections
import numpy as np
import scipy.stats as stat
from common.config import Config
from random import random, randint
from common.files import save_np_array, append_np_array, list_to_file
from processors.data import RawData
from tensorflow.keras.utils import to_categorical
import time
from datetime import timedelta


class Normalize:
    def __init__(self):
        self.__config = Config()

    def build_dictionary(self, lines, vocabulary, vocabulary_size):
        word_frequency = [['UNK', -1]]
        word_frequency.extend(collections.Counter(vocabulary).most_common())
        c = np.transpose(word_frequency)
        c = np.array(c[1], dtype=np.int32)
        print("\nUnique words", len(word_frequency))
        print("Count mode", stat.mode(c))
        print("Count average", np.average(c))

        dictionary = dict()
        dictionary["UNK"] = len(dictionary)
        dictionary["END"] = len(dictionary)
        for word, c in word_frequency:
            if (
                    int(
                        self.__config["PROCESS_DATA"]["MIN_WORD_FREQ"]
                    ) <= c <= int(
                        self.__config["PROCESS_DATA"]["MAX_WORD_FREQ"]
                    )
            ) and len(dictionary) < vocabulary_size:
                dictionary[word] = len(dictionary)

        data = list()
        unk_count = 0
        count_trainable = 0
        for line in lines:
            line_data = list()
            for word in line:
                index = dictionary.get(word, 0)
                if index == 0:  # dictionary['UNK']
                    unk_count += 1
                line_data.append(index)

            length = len(line_data)
            unk = line_data.count(0)
            num = line_data.count(2)

            if (unk + num) / length <= float(self.__config["PROCESS_DATA"]["UNK_CONSIDER"]) \
                    and int(self.__config["PROCESS_DATA"]["MIN_LINE_LENGTH"]) <= length \
                    < int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"]):
                data.append(line_data)
                count_trainable += 1

        print("trainable lines {}".format(count_trainable))

        word_frequency[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        return data, dictionary, reversed_dictionary, word_frequency


class Perturbed:
    def __init__(self):
        self.__config = Config()

        self.__X = []
        self.__y = []

    def add_instance(self, x, y):
        if int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"]) - len(x) > 0:
            x = np.pad(
                np.array(x),
                [(0, int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"]) - len(x))],
                mode='constant',
                constant_values=1
            )
        else:
            x = np.array(x)

        if (int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"]) - len(y)) > 0:
            y = np.pad(
                np.array(y),
                [(0, int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"]) - len(y))],
                mode='constant',
                constant_values=1
            )
        else:
            y = np.array(y)

        for _ in range(randint(
                int(self.__config["PERTURBATION"]["MIN_REPETITION"]),
                int(self.__config["PERTURBATION"]["MAX_REPETITION"])
        )):
            self.__X.append(x)
            self.__y.append(y)

    def swap_words(self, sentence):
        length = len(sentence)
        incorrect = list(sentence)
        count = 0

        while count < int(self.__config["PERTURBATION"]["MAX_SWAP_WORDS"]) and count < length:
            i = randint(0, length - 2)
            incorrect[i], incorrect[i + 1] = incorrect[i + 1], incorrect[i]
            count += 1

        return incorrect

    def remove_words(self, sentence):
        length = len(sentence)
        incorrect = list(sentence)
        count = 0
        while count < int(self.__config["PERTURBATION"]["MAX_MISSING_WORDS"]) and count < length:
            i = randint(0, length - 2)
            del incorrect[i]
            count += 1

        return incorrect

    def build_dataset(self, data):
        count_missing = 0
        count_swap = 0
        count = 0
        count_normal = 0

        for line in data:
            perturbed = False

            if random() <= float(self.__config["PERTURBATION"]["SWAP_WORDS_RATE"]):
                swap_line = self.swap_words(line)
                self.add_instance(line, swap_line)

                count_swap += 1
                perturbed = True

            if random() <= float(self.__config["PERTURBATION"]["SWAP_WORDS_RATE"]):
                missing_line = self.remove_words(line)
                self.add_instance(line, missing_line)

                count_missing += 1
                perturbed = True

            if not perturbed:
                self.add_instance(line, line)
                count_normal += 1

            count += 1
            if count % int(self.__config["PROCESS_DATA"]["REPORT_POINT"]) == 0:
                print("done {}, swap {}, missing {}, normal {}, total {}".format(
                    count,
                    count_swap,
                    count_missing,
                    count_normal,
                    (count_swap + count_missing + count_normal)
                ))

    def get_data(self):
        length = len(self.__X)
        start_index = length - int((length * int(self.__config["DATASET"]["TEST_SPLIT"])) / 100)
        test_x = self.__X[start_index:]
        x = self.__X[:start_index]
        test_y = self.__y[start_index:]
        y = self.__y[:start_index]

        return np.array(x), np.array(y), np.array(test_x), np.array(test_y)


class ProcessData:
    def one_hot(self, data):
        one_hot = to_categorical(data, num_classes=int(self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"]))
        one_hot = one_hot.reshape(
            -1,
            int(
                self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"]
            ) * int(
                self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"]
            )
        )

        return one_hot

    def one_hot_batch_save(self, data, location):
        batch_size = int(self.__config["PROCESS_DATA"]["BATCH_SIZE"])
        chunks = (len(data) - 1) // batch_size + 1
        for i in range(chunks):
            batch = data[i * batch_size:(i + 1) * batch_size]
            train_target = self.one_hot(batch)
            append_np_array(train_target, location)
            print("done {} batch(es)".format(i + 1))

    def __init__(self):
        self.__config = Config()

        start_time = time.time()

        raw_data = RawData()
        lines, vocabulary = raw_data.get_data()

        normalize = Normalize()
        data, dictionary, reversed_dictionary, word_frequency = normalize.build_dictionary(
            lines,
            vocabulary,
            int(self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"])
        )

        print("vocabulary size {}/{}".format(len(dictionary), self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"]))
        print(word_frequency)
        print(dictionary)

        if self.__config["GENERAL"]["MODEL"] == "WordRelation":
            list_to_file(data, "{}/vector.txt".format(self.__config["TRAIN"]["DATA_LOCATION"]))

        if self.__config["GENERAL"]["MODE"] == "SEQ@SEQ":
            perturbed = Perturbed()
            perturbed.build_dataset(data)
            train_x, train_y, test_x, test_y = perturbed.get_data()

            print("elapsed time {}".format(timedelta(seconds=time.time()-start_time)))
            print("one hot train_y..")

            self.one_hot_batch_save(train_y, "{}/target.txt".format(self.__config["TRAIN"]["DATA_LOCATION"]))

            print("elapsed time {}".format(timedelta(seconds=time.time() - start_time)))
            print("one hot test_y..")

            self.one_hot_batch_save(train_y, "{}/target.txt".format(self.__config["TEST"]["DATA_LOCATION"]))

            print("elapsed time {}".format(timedelta(seconds=time.time() - start_time)))
            print("saving files..")
            save_np_array(train_x, "{}/correct.txt".format(self.__config["TRAIN"]["DATA_LOCATION"]))
            save_np_array(train_y, "{}/incorrect.txt".format(self.__config["TRAIN"]["DATA_LOCATION"]))

            save_np_array(test_x, "{}/correct.txt".format(self.__config["TEST"]["DATA_LOCATION"]))
            save_np_array(test_y, "{}/incorrect.txt".format(self.__config["TEST"]["DATA_LOCATION"]))

        print("elapsed time {}".format(timedelta(seconds=time.time() - start_time)))
        print("done")
