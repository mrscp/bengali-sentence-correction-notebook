from common.config import Config
from common.path import ProjectPath
from os.path import join
from common.files import load_int_list
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


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


class WordRelationDataset:
    def __init__(self):
        self.__config = Config()
        self.__vector = load_int_list("{}/vector.txt".format(self.__config["TRAIN"]["DATA_LOCATION"]))

    def next_batch(self):
        radius = 1
        data_length = len(self.__vector)
        vocabulary_size = int(self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"])
        batch_size = int(self.__config["TRAIN"]["BATCH_SIZE"])
        max_seq_len = int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"])
        max_context_len = max_seq_len - (4 * radius + 1)

        padding = np.zeros(int(radius+1), dtype="int32").tolist()

        indexes = np.random.randint(0, data_length, batch_size)
        prev_contexts = []
        prev_words = []
        next_words = []
        next_contexts = []

        class_words = []

        lines = []

        for index in indexes:
            line = self.__vector[index]
            line = padding + [int(token) for token in line] + padding
            line_length = len(line)
            lines.append(line)

            class_index = np.random.randint(radius + 1, line_length - (radius + 1))
            #     class_words.append(to_categorical([line[class_index]], vocabulary_size).transpose())
            class_words.append(to_categorical(line[class_index], vocabulary_size))

            prev_contexts.append(
                pad_sequences(
                    [line[:class_index - radius]],
                    maxlen=max_context_len,
                    dtype='int32',
                    padding='pre',
                    truncating='pre',
                    value=0.
                )
            )

            prev_words.append(
                pad_sequences(
                    [line[class_index - radius:class_index]],
                    maxlen=radius,
                    dtype='int32',
                    padding='post',
                    truncating='post',
                    value=0.
                )
            )

            next_words.append(
                pad_sequences(
                    [line[class_index + 1:class_index + radius + 1]],
                    maxlen=radius,
                    dtype='int32',
                    padding='post',
                    truncating='post',
                    value=0.
                )
            )

            next_contexts.append(
                pad_sequences(
                    [line[class_index + radius + 1:]],
                    maxlen=max_context_len,
                    dtype='int32',
                    padding='post',
                    truncating='post',
                    value=0.
                )
            )

        prev_contexts = np.array(prev_contexts)
        prev_words = np.array(prev_words)
        next_words = np.array(next_words)
        next_contexts = np.array(next_contexts)
        class_words = np.array(class_words)

        return lines, prev_contexts, prev_words, next_words, next_contexts, class_words
