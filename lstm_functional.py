from functools import reduce, partial
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import re


def compose(*args):
    return lambda x: reduce(lambda r, f: f(r), reversed(args), x)


def inspect(x):
    return print(x) or x


def read_file(path):
    return open(path).read()


def clip(text, start_pattern, end_pattern):
    return text[text.find(start_pattern):text.rfind(end_pattern)]


def remove_pattern(text, pattern, replacement=''):
    return re.sub(pattern, replacement, text)


def convert_to_lowercase(text):
    return text.lower()


def save_weights(model, path):
    model.save_weights(path)
    return model


def load_weights(model, path):
    model.load_weights(path)
    return model


def make_encoding_dictionary(vocabulary):
    return {c: i for i, c in enumerate(vocabulary)}


def make_decoding_dictionary(vocabulary):
    return {i: c for i, c in enumerate(vocabulary)}


def prepare_data(text, vocabulary_size, char_to_index, sentence_length, step):
    sentences = []
    next_chars = []
    for i in range(0, len(text)-sentence_length, step):
        sentences.append(text[i:i+sentence_length])
        next_chars.append(text[i+sentence_length])

    x = np.zeros(
        shape=(len(sentences), sentence_length, vocabulary_size),
        dtype=np.bool)
    y = np.zeros(
        shape=(len(sentences), vocabulary_size),
        dtype=np.bool)

    for i, sentence in enumerate(sentences):
        y[i, char_to_index[next_chars[i]]] = 1
        for j, char in enumerate(sentence):
            x[i, j, char_to_index[char]] = 1

    return x, y


def fit_model(model, x, y, batch_size, epochs, callbacks):
    model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    return model


def make_model(
        vocabulary_size,
        learning_rate=0.01,
        hidden_layers=128,
        sentence_length=50,
        optimizer_fun=RMSprop,
        loss='categorical_crossentropy'):
    model = Sequential()
    model.add(
        LSTM(hidden_layers, input_shape=(sentence_length, vocabulary_size)))
    model.add(Dense(vocabulary_size))
    model.add(Activation('softmax'))

    optimizer = optimizer_fun(learning_rate)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def predict(model, text, sentece_length, prediction_length, vocabulary_size, index_to_char, char_to_index, iterations, temperature):
    start_indices = np.random.choice(len(text) - sentece_length, iterations)
    for i in start_indices:
        print('-' * prediction_length)
        input_chars = text[i:i+sentence_length]
        print('Input:', input_chars)
        output_chars = ''
        for j in range(prediction_length):
            x_predict = np.zeros(shape=(1, sentence_length, vocabulary_size))
            for k, char in enumerate(input_chars):
                x_predict[0, k, char_to_index[char]] = 1

            probs = model.predict(x_predict, verbose=0)[0]
            probs = np.asarray(probs).astype('float64')
            probs = np.clip(probs, a_min=1e-32, a_max=None)
            not_probs = np.exp(np.log(probs) / temperature)
            adjusted_probs = not_probs / np.sum(not_probs)
            predicted_index = np.argmax(
                np.random.multinomial(1, adjusted_probs))
            predicted_char = index_to_char[predicted_index]
            output_chars += predicted_char
            input_chars = input_chars[1:] + predicted_char

        print('Output:', output_chars)


path = './meditations.txt'
start_from = 'THE FIRST BOOK'
end_at = 'APPENDIX'
pattern1 = r'THE\s\w+\sBOOK\n'
pattern2 = r'\n[XVI]+\.\s'
pattern3 = r'[^\,\.\!\?\'\;\:\-a-z]+'

load_text = compose(
    partial(remove_pattern, pattern=pattern3, replacement=' '),
    convert_to_lowercase,
    partial(remove_pattern, pattern=pattern2),
    partial(remove_pattern, pattern=pattern1),
    partial(clip, start_pattern=start_from, end_pattern=end_at),
    read_file,
)

make_vocabulary = compose(
    sorted,
    list,
    set
)

text = load_text(path)
vocabulary = make_vocabulary(text)
vocabulary_size = len(vocabulary)
sentence_length = 80
char_to_index = make_encoding_dictionary(vocabulary)
index_to_char = make_decoding_dictionary(vocabulary)
x, y = prepare_data(
    text,
    vocabulary_size,
    make_encoding_dictionary(vocabulary),
    sentence_length=sentence_length,
    step=3)

make_predictions = compose(
    partial(
        predict,
        text=text,
        sentece_length=sentence_length,
        prediction_length=50,
        vocabulary_size=vocabulary_size,
        index_to_char=index_to_char,
        char_to_index=char_to_index,
        iterations=10,
        temperature=0.5),
    partial(save_weights, path='./weights1.hdf5'),
    partial(
        fit_model,
        x=x,
        y=y,
        batch_size=512,
        epochs=20,
        callbacks=[]),
    partial(
        make_model,
        sentence_length=sentence_length
    )
)

predictions = make_predictions(vocabulary_size)
