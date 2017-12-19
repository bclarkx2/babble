#! /usr/bin/env python3

###############################################################################
# Imports                                                                     #
###############################################################################

# LIB
import argparse

#LOCAL
from GAN import GAN
from discriminator import no_conv_disc
from generator import only_dense_gen
from grammar import SimpleGrammar


###############################################################################
# Operations                                                                  #
###############################################################################

def disc(gan):
    gan.train_disc_all(train_itr=250,
                       batch_size=100)

    final_stats = gan.disc_flow.evaluate(gan.all_sentences,
                                         gan.all_labels)
    print("final stats: {}".format(final_stats))


def oracle(gan):
    gan.train_adv_oracle(adv_itr=1000,
                         adv_batch_size=1000)


def full(gan):
    gan.train_adv(train_itr=100,
                  batch_size=10,
                  test_interval=10)


###############################################################################
# Helper functions                                                            #
###############################################################################

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--disc",
                       action="store_true")
    group.add_argument("--oracle",
                       action="store_true")
    group.add_argument("--full",
                       action="store_true")
    return parser.parse_args()


###############################################################################
# Main script                                                                 #
###############################################################################

def main():

    args = get_args()

    discriminator = no_conv_disc(shape=(3, 1))
    generator = only_dense_gen(input_dim=100)
    gram = SimpleGrammar()
    gan = GAN(discriminator, generator, gram)

    if args.disc:
        disc(gan)
    elif args.oracle:
        oracle(gan)
    elif args.full:
        full(gan)


if __name__ == '__main__':
    main()
