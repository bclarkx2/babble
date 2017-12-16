#! /usr/bin/env python3

###############################################################################
# Imports                                                                     #
###############################################################################

import argparse
import nltk


###############################################################################
# Constants                                                                   #
###############################################################################



###############################################################################
# Classes                                                                     #
###############################################################################

# simple_grammar = nltk.CFG.fromstring("""
#   S -> NP VP
#   VP -> V NP | V NP PP
#   PP -> P NP
#   V -> "saw" | "ate" | "walked"
#   NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
#   Det -> "a" | "an" | "the" | "my"
#   N -> "man" | "dog" | "cat" | "telescope" | "park"
#   P -> "in" | "on" | "by" | "with"
#   """)

simple = nltk.data.load("file:grammars/simple.cfg")


###############################################################################
# Helper functions                                                            #
###############################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg")
    return parser.parse_args()


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
