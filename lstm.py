from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
import numpy as np
import re


class LSTM_model:
    def __init__(self, path):
        self.load_source_file(path)
        self.clean_raw_text()
        self.make_vocabulary()
        self.make_encoding_dictionary()
        self.make_decoding_dictionary()

    def load_source_file(self, path):
        self.raw_text = open(path).read()

    def clean_raw_text(self):
        self.text = self.raw_text[self.raw_text.find(
            'THE FIRST BOOK'):self.raw_text.rfind('APPENDIX')]
        self.text = re.sub(r'THE\s\w+\sBOOK\n', '', self.text)
        self.text = re.sub(r'\n[XVI]+\.\s', '', self.text)
        self.text = self.text.lower()
        self.text = re.sub(r'[^\,\.\!\?\'\;\:\-a-z]+', ' ', self.text)

    def make_vocabulary(self):
        self.vocabulary = sorted(list(set(self.text)))
        self.vocabulary_size = len(self.vocabulary)

    def make_encoding_dictionary(self):
        self.char_to_index = {c: i for i, c in enumerate(self.vocabulary)}

    def make_decoding_dictionary(self):
        self.index_to_char = {i: c for i, c in enumerate(self.vocabulary)}

    def prepare_data(self, sentence_length, step):
        self.sentence_length = sentence_length
        self.step = step

        sentences = []
        next_chars = []

        for i in range(0, len(self.text) - sentence_length, step):
            sentences.append(self.text[i:i+sentence_length])
            next_chars.append(self.text[i+sentence_length])

        x_shape = len(sentences), sentence_length, self.vocabulary_size
        y_shape = len(sentences), self.vocabulary_size

        self.x = np.zeros(shape=x_shape, dtype=np.bool)
        self.y = np.zeros(shape=y_shape, dtype=np.bool)

        for i, sentence in enumerate(sentences):
            for j, char in enumerate(sentence):
                self.x[i, j, self.char_to_index[char]] = 1
            self.y[i, self.char_to_index[next_chars[i]]] = 1

    def make_model(self, size, learning_rate=0.01, optimizer_fun=RMSprop, loss='categorical_crossentropy'):
        self.model = Sequential()
        self.model.add(LSTM(size, input_shape=(
            self.sentence_length, self.vocabulary_size)))
        self.model.add(Dense(self.vocabulary_size))
        self.model.add(Activation('softmax'))

        self.optimizer = optimizer_fun(learning_rate)

        self.model.compile(loss=loss, optimizer=self.optimizer)

    def fit(self, batch_size=128, epochs=50, callbacks=[]):
        self.model.fit(self.x, self.y, batch_size=batch_size,
                       epochs=epochs, callbacks=callbacks)

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def predict(self, prediction_length, iterations, temperature=0.5):
        start_indices = np.random.choice(
            len(self.text) - self.sentence_length, iterations)
        for i in start_indices:
            print('-' * prediction_length)
            input_chars = self.text[i:i+self.sentence_length]
            print('Input:', input_chars)
            output_chars = ''
            for _ in range(prediction_length):
                x_predict = np.zeros(
                    shape=(1, self.sentence_length, self.vocabulary_size))
                for k, char in enumerate(input_chars):
                    x_predict[0, k, self.char_to_index[char]] = 1

                probs = self.model.predict(x_predict, verbose=0)[0]
                probs = np.asarray(probs).astype('float64')
                probs = np.clip(probs, a_min=1e-32, a_max=None)
                not_probs = np.exp(np.log(probs) / temperature)
                adjusted_probs = not_probs / np.sum(not_probs)
                predicted_index = np.argmax(
                    np.random.multinomial(1, adjusted_probs))
                predicted_char = self.index_to_char[predicted_index]
                output_chars += predicted_char
                input_chars = input_chars[1:] + predicted_char
            print('Output:', output_chars)


if __name__ == "__main__":
    import sys
    path = sys.argv[1]

    marcus = LSTM_model(path)
    marcus.prepare_data(sentence_length=50, step=5)
    marcus.make_model(size=128)
    marcus.fit(batch_size=512, epochs=1)
    marcus.save_weights('./new_weights.hdf5')
    marcus.predict(prediction_length=200, iterations=1, temperature=0.3)
