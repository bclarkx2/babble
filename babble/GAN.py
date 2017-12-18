#! /usr/bin/env python3

###############################################################################
# Imports                                                                     #
###############################################################################

# LIB
import argparse

import keras
import numpy as np
import tensorflow as tf
from copy import deepcopy
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input
from keras.optimizers import RMSprop

# LOCAL
from keras.utils import plot_model

from discriminator import no_conv_disc
from generator import basic_generator, only_dense_gen
from grammar import SimpleGrammar

np.random.seed(1234)


###############################################################################
# Constants                                                                   #
###############################################################################

EPSILON = K.epsilon()


###############################################################################
# Classes                                                                     #
###############################################################################

class GAN(object):

    SENTENCE_LENGTH = 3
    NOISE_SIZE = 100

    def __init__(self, disc, gen, grammar):

        self.grammar = grammar

        self.pos_sentences = self.grammar.sentences()
        self.neg_sentences = self.grammar.negatives()
        self.all_sentences = np.concatenate((self.pos_sentences,
                                             self.neg_sentences))
        self.all_labels = np.concatenate((
            np.ones((len(self.pos_sentences)),),
            np.zeros(len(self.neg_sentences)),))

        self.disc = disc
        self.gen = gen

        self.disc_flow = self._disc_flow()
        self.gen_flow = self._gen_flow()
        self.adv_flow = self._adv_flow()

    @property
    def num_words(self):
        return self.grammar.num_words

    def translate(self, index_list):
        return self.grammar.to_sentence(index_list)

    def _disc_flow(self):
        optimizer = RMSprop(lr=0.002, decay=6e-8)
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

    def _gen_flow(self):
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.gen_flow = Sequential()
        self.gen_flow.add(self.gen)
        self.gen_flow.add(self.disc)
        self.disc.trainable = False
        self.gen_flow.compile(loss="binary_crossentropy",
                              optimizer=optimizer,
                              metrics=['accuracy'])
        return self.gen_flow

    def train_gen(self, train_itr=100, batch_size=10):

        for itr in range(train_itr):

            noise = GAN.noise(batch_size)
            labels = np.ones([batch_size, 1])
            loss = self.gen_flow.train_on_batch(noise, labels)

            print("Itr: {}; loss: {}".format(itr, loss))

    def train_disc_all(self, train_itr=100, batch_size=32):
        """Train only the discriminative model

        This version uses the fit method of keras.models.Sequential
        """

        pos_examples = self.pos_sentences
        neg_examples = self.neg_sentences

        examples = np.concatenate((pos_examples, neg_examples))

        pos_labels = np.ones((len(pos_examples),))
        neg_labels = np.zeros((len(neg_examples),))

        labels = np.concatenate((pos_labels, neg_labels))

        self.disc_flow.fit(x=examples,
                           y=labels,
                           batch_size=None,
                           epochs=train_itr,
                           class_weight={0: 0.5, 1: 0.5},
                           validation_split=0.1,
                           verbose=1)

    def train_disc_batch(self,
                         train_itr=100,
                         batch_size=10,
                         printerval=100,
                         full_eval=False):
        """Train only the discriminative model

        This version uses randomized batches for training
        """
        for itr in range(train_itr):

            pos_examples = self.pos_sentences[self.train_indices(batch_size)]
            neg_examples = self.neg_sentences[self.train_indices(batch_size)]

            examples, labels = label(pos_examples, neg_examples)

            disc_loss = self.disc_flow.train_on_batch(examples, labels)

            if itr % printerval == 0:
                if full_eval:  # potentially look at all examples
                    disc_loss = self.disc_flow.evaluate(self.all_sentences,
                                                        self.all_labels,
                                                        verbose=0)

                print("Iter: {:<5d}; Discriminator loss: {}".format(itr, disc_loss))

    def save_oracle(self):
        self.train_disc_all(train_itr=300,
                            batch_size=100)
        self.disc_flow.save("disc_flow")
        self.adv_flow.save("adv_flow")

    def train_adv_oracle(self,
                         disc_itr=200,
                         disc_batch_size=100,
                         adv_itr=1000,
                         adv_batch_size=100,
                         printerval=100,
                         fresh_oracle=False):

        # first, train the discriminator fully
        if fresh_oracle:
            self.train_disc_all(train_itr=disc_itr,
                                batch_size=disc_batch_size)
            oracle = self.disc_flow.layers[0]
        else:
            old_disc_flow = keras.models.load_model("disc_flow")
            oracle = old_disc_flow.layers[0]
        oracle.trainable = False

        # build an adversarial model using oracle
        optimizer = RMSprop(lr=0.001)  # decay=3e-8)
        oracle_flow = Sequential()
        oracle_flow.add(self.gen)
        oracle_flow.add(oracle)  # add fixed, saved oracle
        oracle_flow.compile(loss='binary_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # train on a bunch of noise
        # assert that noise should always lead to valid sentence
        # eventually, it should actually make valid sentences from noise
        for itr in range(adv_itr):

            noise = GAN.noise(adv_batch_size)
            labels = np.ones([adv_batch_size, 1])

            loss = oracle_flow.train_on_batch(noise, labels)

            if itr % printerval == 0:
                print("Itr: {}; loss: {}".format(itr, loss))
                print(self.generate())

    def train_adv(self, train_itr=100, batch_size=10, test_interval=100):

        for itr in range(train_itr):

            # generate some fake sentences from generator
            noise = GAN.noise(batch_size)
            neg_examples = self.gen.predict(noise)

            # grab some positive examples from list of valid sentences
            pos_examples = self.pos_sentences[self.train_indices(batch_size)]

            # combine them to make a well rounded batch
            examples, labels = label(pos_examples, neg_examples)

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

    @staticmethod
    def noise(batch_size):
        return np.random.uniform(-1.0, 1.0, size=[batch_size, GAN.NOISE_SIZE])

    def train_indices(self, batch_size):
        return np.random.randint(0, self.num_words, size=batch_size)


###############################################################################
# Helper functions                                                            #
###############################################################################

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def label(pos_examples, neg_examples):
    examples = np.concatenate((pos_examples, neg_examples))

    pos_labels = np.ones((len(pos_examples),))
    neg_labels = np.zeros((len(neg_examples),))

    labels = np.concatenate((pos_labels, neg_labels))

    return examples, labels


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg")
    return parser.parse_args()


###############################################################################
# Main script                                                                 #
###############################################################################

def main():

    # args = get_args()

    disc = no_conv_disc(shape=(3, 1))
    gen = only_dense_gen(input_dim=100)
    gram = SimpleGrammar()
    gan = GAN(disc, gen, gram)

    # gan.train_disc_all(train_itr=200,
    #                      batch_size=100)
    #
    # final_stats = gan.disc_flow.evaluate(gan.all_sentences,
    #                                      gan.all_labels)
    # print("final stats: {}".format(final_stats))

    gan.train_adv_oracle(adv_itr=2000,
                         adv_batch_size=1000)

    # gan.save_oracle()


if __name__ == '__main__':
    main()
