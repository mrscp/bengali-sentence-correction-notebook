from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, TimeDistributed, Dense


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

