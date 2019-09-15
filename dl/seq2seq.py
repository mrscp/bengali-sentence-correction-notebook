from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, TimeDistributed, Dense, concatenate, Flatten, Dropout
from common.files import load_np_array
from processors.data import WordRelationDataset
from common.config import Config
import numpy as np
from common.path import ProjectPath


class SequenceToSequence(Model):
    def __init__(self, vocab_size, max_length):
        embedding_layer = Embedding(input_dim=vocab_size,
                                    output_dim=10,
                                    input_length=max_length)

        encoder_inputs = Input(shape=(max_length,), dtype='int32', )
        encoder_embedding = embedding_layer(encoder_inputs)
        encoder_lstm = LSTM(512, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

        decoder_inputs = Input(shape=(max_length,), dtype='int32', )
        decoder_embedding = embedding_layer(decoder_inputs)
        decoder_lstm = LSTM(512, return_state=True, return_sequences=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

        # dense_layer = Dense(vocab_size, activation='relu')(decoder_outputs)

        outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder_outputs)
        super(SequenceToSequence, self).__init__([encoder_inputs, decoder_inputs], outputs)

        self.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["mape"])

    def train(self):
        x = load_np_array("{}/correct.txt".format(self.__config["TRAIN"]["DATA_LOCATION"]))
        y = load_np_array("{}/incorrect.txt".format(self.__config["TRAIN"]["DATA_LOCATION"]))
        target = load_np_array("{}/target.txt".format(self.__config["TRAIN"]["DATA_LOCATION"]))

        target = target.reshape(
            -1,
            int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"]),
            int(self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"])
        )

        self.fit((x, y), target, batch_size=int(self.__config["TRAIN"]["BATCH_SIZE"]), epochs=int(self.__config["TRAIN"]["EPOCHS"]))

    def test(self):
        model_path = self.__project_path.format_location(
            "{}/{}".format(
                self.__config["TRAIN"]["MODEL_LOCATION"],
                "{}.h5".format(self.__config["GENERAL"]["MODEL"])
            )
        )

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
            print()


class WordRelation(Model):
    def __init__(self):
        self.__config = Config()
        self.__data = WordRelationDataset()
        self.__project_path = ProjectPath()

        radius = 1
        self.__vocabulary_size = int(self.__config["PROCESS_DATA"]["VOCABULARY_SIZE"])
        max_seq_len = int(self.__config["PROCESS_DATA"]["MAX_LINE_LENGTH"])
        max_context_len = max_seq_len - (4 * radius + 1)
        embedding_output_size = 100

        prev_context_input = Input((None, max_context_len))
        prev_context_output = Dense(embedding_output_size, activation="relu")(prev_context_input)

        prev_word_input = Input((None, radius))
        prev_word_output = Dense(embedding_output_size, activation="relu")(prev_word_input)

        next_word_input = Input((None, radius))
        next_word_output = Dense(embedding_output_size, activation="relu")(next_word_input)

        next_context_input = Input((None, max_context_len))
        next_context_output = Dense(embedding_output_size, activation="relu")(next_context_input)

        sentence_image = concatenate([prev_context_output, prev_word_output, next_word_output, next_context_output])
        dropout = Dropout(0.25)(sentence_image)
        output = Dense(self.__vocabulary_size, activation="softmax")(dropout)

        super(WordRelation, self).__init__([prev_context_input, prev_word_input, next_word_input, next_context_input], output)
        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.summary())

    def train(self):
        lines, prev_contexts, prev_words, next_words, next_contexts, class_words = self.__data.next_batch()

        import time

        total_time = 0
        epochs = 10
        batch_per_epoch = 4

        for epoch in range(epochs):
            epoch_time_start = time.time()

            print("Epoch: {}".format(epoch + 1))
            print("\tTraining ...")
            #   print("\tTraining ", end="", flush=True)

            batch_losses = []
            batch_accuracies = []
            for batch in range(batch_per_epoch):
                lines, prev_contexts, prev_words, next_words, next_contexts, class_words = self.__data.next_batch()
                loss, accuracy = self.train_on_batch(
                    [prev_contexts, prev_words, next_words, next_contexts],
                    class_words.reshape(-1, 1, self.__vocabulary_size)
                )

                batch_losses.append(loss)
                batch_accuracies.append(accuracy)

                if batch % ((batch_per_epoch * 10) / 100) == 0:
                    print("\t\tBatch {}: loss - {:.4f}, acc - {:.4f}".format(batch, round(np.average(batch_losses), 4),
                                                                             round(np.average(batch_accuracies), 4)))
            #       print(".",end="",flush=True)

            print("")
            print("\tEpoch Loss: {}".format(np.average(batch_losses)))
            print("\tEpoch Accu: {}".format(np.average(batch_accuracies)))

            epoch_time = time.time() - epoch_time_start
            total_time += epoch_time
            batch_time = epoch_time / batch_per_epoch

            print("\tBatch train time: {:.3f} s".format(round(batch_time, 3)))
            print("\tEpoch train time: {}".format(time.strftime("%H:%M:%S", time.gmtime(epoch_time))))
            print("\tTotal train time: {}".format(time.strftime("%H:%M:%S", time.gmtime(total_time))))

    def test(self):
        lines, prev_contexts, prev_words, next_words, next_contexts, class_words = self.__data.next_batch()
        model_path = self.__project_path.format_location(
            "{}/{}".format(
                self.__config["TRAIN"]["MODEL_LOCATION"],
                "model_weights.h5"
            )
        )

        self.load_weights(model_path)

        result = self.predict([prev_contexts, prev_words, next_words, next_contexts])
        result = np.argmax(result, axis=2)
        class_words = np.argmax(class_words, axis=1)

        for r, c in zip(result, class_words):
            print(r, c)
