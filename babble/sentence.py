
import os
import random

from keras.preprocessing.text import Tokenizer

from grammar import simple_parser

SIMPLE = "data/simple"
SIMPLE_NOUNS = os.path.join(SIMPLE, "nouns")
SIMPLE_VERBS = os.path.join(SIMPLE, "verbs")
SIMPLE_OBJECTS = os.path.join(SIMPLE, "objects")

SIMPLE_WORDS = os.path.join(SIMPLE, "words")

SIMPLE_SENTENCES = os.path.join(SIMPLE, "sentences")
SIMPLE_NEGATIVES = os.path.join(SIMPLE, "negatives")

def read_words(words_filename):
    with open(words_filename) as words_file:
        words = words_file.read().splitlines()
    return words


def read_all():
    nouns = read_words(SIMPLE_NOUNS)
    verbs = read_words(SIMPLE_VERBS)
    objects = read_words(SIMPLE_OBJECTS)
    return nouns, verbs, objects


def sentence(nouns, verbs, objects):
    return " ".join([
        random.choice(nouns),
        random.choice(verbs),
        random.choice(objects)
    ])


def build_simple_sentences(iterations=10):

    nouns, verbs, objects = read_all()

    with open(SIMPLE_SENTENCES, "w") as sentence_file:
        for _ in range(iterations):
            sentence_file.write(sentence(nouns, verbs, objects) + "\n")


def write_words(f, words):
    for word in words:
        f.write(word + "\n")


def build_simple_words():

    with open(SIMPLE_WORDS, "w") as words_file:
        for words in read_all():
            write_words(words_file, words)


def build_simple_negatives(iterations=100):

    words = read_words(SIMPLE_WORDS)

    parser = simple_parser()

    with open(SIMPLE_NEGATIVES, "w") as neg_file:
        for _ in range(iterations):
            sent = invalid_sentence(parser, words)
            neg_file.write(" ".join(sent) + "\n")


def invalid_sentence(parser, words):

    while True:
        sent = [random.choice(words) for _ in range(3)]
        if not is_valid_sentence(parser, sent):
            return sent


def is_valid_sentence(parser, sent):
    try:
        trees = list(parser.parse(sent))
        return len(trees) == 1
    except ValueError:
        return False



def main():

    build_simple_sentences(100)
    # build_simple_words()
    # build_simple_negatives(iterations=100)

    # print(sentences)


if __name__ == '__main__':
    main()
