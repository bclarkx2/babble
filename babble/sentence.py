
import os
import random

from keras.preprocessing.text import Tokenizer

SIMPLE = "data/simple"
SIMPLE_NOUNS = os.path.join(SIMPLE, "nouns")
SIMPLE_VERBS = os.path.join(SIMPLE, "verbs")
SIMPLE_OBJECTS = os.path.join(SIMPLE, "objects")

SIMPLE_SENTENCES = os.path.join(SIMPLE, "sentences")


def read_words(words_filename):
    with open(words_filename) as words_file:
        words = words_file.read().splitlines()
    return words


def sentence(nouns, verbs, objects):
    return " ".join([
        random.choice(nouns),
        random.choice(verbs),
        random.choice(objects)
    ])


def build_simple_sentences(iterations=10):

    nouns = read_words(SIMPLE_NOUNS)
    verbs = read_words(SIMPLE_VERBS)
    objects = read_words(SIMPLE_OBJECTS)

    with open(SIMPLE_SENTENCES, "w") as sentence_file:
        for _ in range(iterations):
            sentence_file.write(sentence(nouns, verbs, objects) + "\n")


def main():

    build_simple_sentences(10)

    # with open("data/sentences", "r") as sentence_file:
    #     sentences = sentence_file.read().splitlines()
    #
    # tokenizer = Tokenizer(num_words=None,
    #                       filters="!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
    #                       lower=True,
    #                       split=" ",
    #                       char_level=False)
    #
    # tokenizer.fit_on_texts(sentences)


    # print(sentences)


if __name__ == '__main__':
    main()
