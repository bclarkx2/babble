#! /usr/bin/env python3

###############################################################################
# Imports                                                                     #
###############################################################################

# LIB
import argparse
import numpy as np
from keras.layers import Conv1D, LeakyReLU, Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer

# LOCAL

np.random.seed(1234)


###############################################################################
# Constants                                                                   #
###############################################################################

SIMPLE_SENTENCES = "data/simple/sentences"
SIMPLE_NEGATIVES = "data/simple/negatives"
SIMPLE_WORDS = "data/simple/words"


###############################################################################
# Classes                                                                     #
###############################################################################


class GAN(object):

    SENTENCE_LENGTH = 3

    def __init__(self):

        self.tkn = tokenizer(SIMPLE_WORDS)
        self.corpus = corpus(SIMPLE_SENTENCES)
        self.num_words = len(self.corpus)

        self.pos_sentences = sequences(self.tkn, SIMPLE_SENTENCES)
        self.neg_sentences = sequences(self.tkn, SIMPLE_NEGATIVES)

        self.disc = self._disc()
        self.gen = None

        self.disc_flow = self._disc_flow()

    def _disc(self):

        self.disc = Sequential()

        depth = 64
        dropout = 0.4

        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64

        # input_shape = (GAN.SENTENCE_LENGTH, 1)
        input_shape = self.pos_sentences.shape[1:]
        # padding='same'
        # strides=1
        self.disc.add(Conv1D(2, 2, input_shape=input_shape))
        # self.disc.add(LeakyReLU(alpha=0.2))
        # self.disc.add(Dropout(dropout))

        # self.disc.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        # self.disc.add(LeakyReLU(alpha=0.2))
        # self.disc.add(Dropout(dropout))
        #
        # self.disc.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        # self.disc.add(LeakyReLU(alpha=0.2))
        # self.disc.add(Dropout(dropout))
        #
        # self.disc.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        # self.disc.add(LeakyReLU(alpha=0.2))
        # self.disc.add(Dropout(dropout))

        # Out: 1-dim probability
        self.disc.add(Flatten())
        self.disc.add(Dense(1))
        self.disc.add(Activation('sigmoid'))
        self.disc.summary()
        return self.disc

    def _disc_flow(self):
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.disc_flow = Sequential()
        self.disc_flow.add(self._disc())
        self.disc_flow.compile(loss='binary_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])
        return self.disc_flow

    def train_disc_flow(self, train_itr=100, batch_size=10):

        for itr in range(train_itr):

            pos_examples = self.pos_sentences[self.train_indices(batch_size)]
            neg_examples = self.neg_sentences[self.train_indices(batch_size)]

            examples = np.concatenate((pos_examples, neg_examples))

            pos_labels = np.ones((batch_size,))
            neg_labels = np.zeros((batch_size,))

            labels = np.concatenate((pos_labels, neg_labels))

            disc_loss = self.disc_flow.train_on_batch(examples, labels)

            print("Iter: {:<5d}; Discriminator loss: {}".format(itr, disc_loss))

    def train_indices(self, batch_size):
        return np.random.randint(0, self.num_words, size=batch_size)


###############################################################################
# Helper functions                                                            #
###############################################################################

def read_lines(filename):
    with open(filename, 'r') as sentence_file:
        return sentence_file.read().splitlines()


def sequences(tkn, filename):

    sentences = read_lines(filename)

    seqs = tkn.texts_to_sequences(sentences)
    seqs = np.array(seqs, dtype='float32')
    seqs = np.expand_dims(seqs, axis=2)

    return seqs


def corpus(filename):

    tkn = tokenizer(filename)

    return tkn.word_index


def tokenizer(filename):

    sentences = read_lines(filename)

    tkn = Tokenizer(num_words=None,
                    filters="!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
                    lower=True,
                    split=" ",
                    char_level=False)

    tkn.fit_on_texts(sentences)

    return tkn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg")
    return parser.parse_args()


###############################################################################
# Main script                                                                 #
###############################################################################

def main():

    args = get_args()

    gan = GAN()

    # print(gan.pos_sentences)
    # print(gan.neg_sentences)

    gan.train_disc_flow(train_itr=10000)


if __name__ == '__main__':
    main()
