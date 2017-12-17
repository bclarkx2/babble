#! /usr/bin/env python3

###############################################################################
# Imports                                                                     #
###############################################################################

# LIB
import argparse

import numpy as np
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer

# LOCAL
from discriminator import basic_discriminator
from generator import basic_generator

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
    NOISE_SIZE = 100

    def __init__(self, disc, gen):

        self.tkn = tokenizer(SIMPLE_WORDS)
        self.corpus = corpus(SIMPLE_SENTENCES)
        self.reverse_corpus = reverse_corpus(self.corpus)
        self.num_words = len(self.corpus)

        self.pos_sentences = sequences(self.tkn, SIMPLE_SENTENCES)
        self.neg_sentences = sequences(self.tkn, SIMPLE_NEGATIVES)

        self.disc = disc
        self.gen = gen

        self.disc_flow = self._disc_flow()
        self.adv_flow = self._adv_flow()

    def _disc_flow(self):
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.disc_flow = Sequential()
        self.disc_flow.add(self.disc)
        self.disc_flow.compile(loss='binary_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])
        return self.disc_flow

    def _adv_flow(self):
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.adv_flow = Sequential()
        self.adv_flow.add(self.gen)
        self.adv_flow.add(self.disc)
        self.adv_flow.compile(loss='binary_crossentropy',
                              optimizer=optimizer,
                              metrics=['accuracy'])
        return self.adv_flow

    def train_disc(self, train_itr=100, batch_size=10):

        for itr in range(train_itr):

            pos_examples = self.pos_sentences[self.train_indices(batch_size)]
            neg_examples = self.neg_sentences[self.train_indices(batch_size)]

            examples = np.concatenate((pos_examples, neg_examples))

            pos_labels = np.ones((batch_size,))
            neg_labels = np.zeros((batch_size,))

            labels = np.concatenate((pos_labels, neg_labels))

            disc_loss = self.disc_flow.train_on_batch(examples, labels)

            print("Iter: {:<5d}; Discriminator loss: {}".format(itr, disc_loss))

    def train_adv(self, train_itr=100, batch_size=10, test_interval=100):

        for itr in range(train_itr):

            # generate some fake sentences from generator
            noise = GAN.noise(batch_size)
            neg_examples = self.gen.predict(noise)
            neg_labels = np.zeros((batch_size,))

            # grab some positive examples from list of valid sentences
            pos_examples = self.pos_sentences[self.train_indices(batch_size)]
            pos_labels = np.ones((batch_size,))

            # combine them to make a well rounded batch
            examples = np.concatenate((pos_examples, neg_examples))
            labels = np.concatenate((pos_labels, neg_labels))

            # train the discriminator to get better at ignoring fake sentences
            disc_loss = self.disc_flow.train_on_batch(examples, labels)

            # adversarial
            adv_noise = GAN.noise(batch_size)
            adv_labels = np.ones([batch_size, 1])
            adv_loss = self.adv_flow.train_on_batch(adv_noise, adv_labels)

            if itr % test_interval == 0:
                print("=" * 50)
                report = "Iter: {:<5d}; disc: ({:<5f} {:<5f}); adv: ({:<5f} {:<5f})"
                report = report.format(itr, *disc_loss, *adv_loss)
                print(report)
                print(self.generate())
                print("=" * 50)

    def generate(self):
        """Generate a single sentence from the model"""
        test_noise = GAN.noise(1)
        generated_output = self.gen.predict(test_noise)[0]
        generated_list = [x for lst in generated_output for x in lst]
        translated = self.translate(generated_list)
        return translated

    def translate(self, index_list):
        index_scalar = self.num_words - 1
        scaled = [int(idx * index_scalar) + 1 for idx in index_list]
        translated_lst = [self.reverse_corpus[idx] for idx in scaled]
        translated_string = " ".join(translated_lst)
        return translated_string

    @staticmethod
    def noise(batch_size):
        return np.random.uniform(-1.0, 1.0, size=[batch_size, GAN.NOISE_SIZE])

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


def reverse_corpus(corpus_dict):
    return {val: key for key, val in corpus_dict.items()}


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

    disc = basic_discriminator(shape=(3,1))
    gen = basic_generator()
    gan = GAN(disc, gen)

    # print(gan.pos_sentences)
    # print(gan.neg_sentences)

    # gan.train_disc(train_itr=2000)

    gan.train_adv(train_itr=101)


if __name__ == '__main__':
    main()
