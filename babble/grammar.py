#! /usr/bin/env python3

###############################################################################
# Imports                                                                     #
###############################################################################

import argparse
import nltk

from keras.preprocessing.text import Tokenizer


simple = nltk.data.load("file:grammars/simple.cfg")


###############################################################################
# Helper functions                                                            #
###############################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg")
    return parser.parse_args()


def simple_parser():
    return nltk.RecursiveDescentParser(simple)


###############################################################################
# Main script                                                                 #
###############################################################################

def main():

    # args = get_args()

    sentence = "Mary spoke".split()

    parser = nltk.RecursiveDescentParser(simple)

    for tree in parser.parse(sentence):
        print(tree)


if __name__ == '__main__':
    main()
