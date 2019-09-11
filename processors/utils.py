import collections
import numpy as np
import scipy.stats as stat


class Normalization:
    def __init__(self):
        print("yes")

    def build_data_set(self, lines, vocabulary, vocabulary_size):
        """Process raw inputs into a bangla-text1."""
        count = [['UNK', -1]]
        count.extend(collections.Counter(vocabulary).most_common())
        # print(count)
        c = np.transpose(count)
        c = np.array(c[1], dtype=np.int32)
        print("\nUnique words", len(count))
        print("Count mode", stat.mode(c))
        print("Count average", np.average(c))

        dictionary = dict()
        dictionary["UNK"] = len(dictionary)
        dictionary["END"] = len(dictionary)
        for word, c in count:
            if (config["min_count"] <= c <= config["max_count"]) and len(dictionary) < vocabulary_size:
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
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        return data, count, dictionary, reversed_dictionary
