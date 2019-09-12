import collections
import numpy as np
import scipy.stats as stat
from common.config import Config
from random import random, randint


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
        for line in lines:
            line_data = list()
            for word in line:
                index = dictionary.get(word, 0)
                if index == 0:  # dictionary['UNK']
                    unk_count += 1
                line_data.append(index)
            data.append(line_data)
        word_frequency[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        return data, dictionary, reversed_dictionary, word_frequency


class Perturbed:
    def __init__(self):
        self.__config = Config()

        self.__X = []
        self.__y = []

    def add_instance(self, x, y):
        for _ in range(randint(
                int(self.__config["PERTURBATION"]["MIN_REPETITION"]),
                int(self.__config["PERTURBATION"]["MAX_REPETITION"])
        )):
            self.__X.append(" ".join(x) + "\n")
            self.__y.append(" ".join(y) + "\n")

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
        count_lines = 0
        count_normal = 0

        for line in data:
            line = [str(word) for word in line]
            length = len(line)
            unk = line.count("0")
            num = line.count("1")
            perturbed = False

            if (unk+num)/length <= float(self.__config["PROCESS_DATA"]["UNK_CONSIDER"]) \
                    and int(self.__config["PROCESS_DATA"]["MIN_LINE_LENGTH"]) <= length\
                    < int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"]):

                if random() >= float(self.__config["PERTURBATION"]["SWAP_WORDS_RATE"]):
                    swap_line = self.swap_words(line)
                    self.add_instance(line, swap_line)

                    count_swap += 1
                    perturbed = True

                if random() >= float(self.__config["PERTURBATION"]["SWAP_WORDS_RATE"]):
                    missing_line = self.swap_words(line)
                    self.add_instance(line, missing_line)

                    count_missing += 1
                    perturbed = True

                if not perturbed:
                    self.add_instance(line, line)
                    count_normal += 1

                count_lines += 1

            count += 1
            if count % int(self.__config["PROCESS_DATA"]["REPORT_POINT"]) == 0:
                print("done {}, lines {}, swap {}, missing {}, normal {}, total {}".format(
                    count,
                    count_lines,
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

        print(len(x), len(test_x))
