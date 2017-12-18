#! /usr/bin/env python3

###############################################################################
# Imports                                                                     #
###############################################################################

# LIB
import argparse
import nltk
import os
import random
import numpy as np

from keras.preprocessing.text import Tokenizer


###############################################################################
# Classes                                                                     #
###############################################################################

class Grammar(object):

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.words = self.build_words()
        self.tokenizer = self.build_tokenizer()
        self.parser = self.build_parser()
        self.corpus = self.tokenizer.word_index
        self.reverse_corpus = {val: key for key, val in self.corpus.items()}
        self.sentence_file = os.path.join(self.base_dir, "sentences")
        self.negative_file = os.path.join(self.base_dir, "negatives")

    @property
    def num_words(self):
        return len(self.tokenizer.word_index)

    def build_words(self):
        raise NotImplementedError()

    def build_tokenizer(self):
        raise NotImplementedError()

    def build_parser(self):
        raise NotImplementedError()

    def sentences(self):
        raise NotImplementedError()

    def negatives(self):
        raise NotImplementedError()

    def write_sentences(self):
        raise NotImplementedError()

    def write_negatives(self):
        raise NotImplementedError()

    def parse(self, sentence):
        return self.parser.parse(sentence)

    def to_sentence(self, index_list):
        # index_scalar = self.num_words - 1
        # scaled = [int(idx * index_scalar) + 1 for idx in index_list]
        scaled = [int(idx) for idx in index_list]
        translated_lst = [self.reverse_corpus[idx] for idx in scaled]
        translated_string = " ".join(translated_lst)
        return translated_string


class SimpleGrammar(Grammar):

    BASE = "data/simple"

    def __init__(self):
        self.noun_file = os.path.join(SimpleGrammar.BASE, "nouns")
        self.verb_file = os.path.join(SimpleGrammar.BASE, "verbs")
        self.object_file = os.path.join(SimpleGrammar.BASE, "objects")
        self.nouns = read_file(self.noun_file)
        self.verbs = read_file(self.verb_file)
        self.objects = read_file(self.object_file)
        super(SimpleGrammar, self).__init__(SimpleGrammar.BASE)

    def build_words(self):
        words = []
        for word_type in [self.nouns, self.verbs, self.objects]:
            words.extend(word_type)
        return words

    def build_tokenizer(self):
        tkn = Tokenizer(num_words=None,
                        filters="!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
                        lower=True,
                        split=" ",
                        char_level=False)

        tkn.fit_on_texts(self.words)
        return tkn

    def build_parser(self):
        grammar_str = """
        S -> N V O
        N -> {}
        V -> {}
        O -> {}
        """.format(
            " | ".join(['"' + x + '"' for x in self.nouns]),
            " | ".join(['"' + x + '"' for x in self.verbs]),
            " | ".join(['"' + x + '"' for x in self.objects])
        )
        grammar = nltk.CFG.fromstring(grammar_str)
        parser = nltk.RecursiveDescentParser(grammar)
        return parser

    def sentences(self):
        if not os.path.isfile(self.sentence_file):
            self.write_sentences()
        return self._sequences(self.sentence_file)

    def negatives(self):
        if not os.path.isfile(self.negative_file):
            self.write_negatives()
        return self._sequences(self.negative_file)

    def _sequences(self, filename):
        sentence_list = read_file(filename)

        seqs = self.tokenizer.texts_to_sequences(sentence_list)
        seqs = np.array(seqs, dtype='float32')
        seqs = np.expand_dims(seqs, axis=2)

        return seqs

    def write_sentences(self, num=100):
        with open(self.sentence_file, "w") as sentence_file:
            for _ in range(num):
                sentence_file.write(self._sentence() + "\n")

    def _sentence(self):
        return " ".join([
            random.choice(self.nouns),
            random.choice(self.verbs),
            random.choice(self.objects)
        ])

    def _invalid_sentence(self):

        while True:
            sentence = [random.choice(self.words) for _ in range(3)]
            if not self.is_valid_sentence(sentence):
                return sentence

    def is_valid_sentence(self, sentence):
        try:
            trees = list(self.parse(sentence))
            return len(trees) == 1
        except ValueError:
            return False

    def write_negatives(self, num=100):
        with open(self.negative_file, "w") as neg_file:
            for _ in range(num):
                sentence = self._invalid_sentence()
                neg_file.write(" ".join(sentence) + "\n")


###############################################################################
# Helper functions                                                            #
###############################################################################

def read_file(filename):
    with open(filename) as lines_file:
        lines = lines_file.read().splitlines()
    return lines


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg")
    return parser.parse_args()


###############################################################################
# Main script                                                                 #
###############################################################################

def main():

    # args = get_args()

    bad_sentence = "Mary saw".split()
    good_sentence = "Mary saw apples".split()

    sg = SimpleGrammar()

    print("bad")
    for tree in sg.parse(bad_sentence):
        print(tree)

    print("good")
    for tree in sg.parse(good_sentence):
        print(tree)

    print("sentences")
    print(sg.sentences()[0:3])

    print("negatives")
    print(sg.negatives()[1:3])

    print("translate")
    print(sg.to_sentence([0.02, 0.55, 0.98]))


if __name__ == '__main__':
    main()
