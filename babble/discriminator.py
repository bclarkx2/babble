
###############################################################################
# Imports                                                                     #
###############################################################################

from keras import Sequential
from keras.layers import Conv1D, Flatten, Dense, Activation, LeakyReLU, Dropout


###############################################################################
# Discriminators                                                              #
###############################################################################

def basic_discriminator(shape=(3,1)):

    disc = Sequential()

    depth = 64
    dropout = 0.4

    # In: 28 x 28 x 1, depth = 1
    # Out: 14 x 14 x 1, depth=64

    # input_shape = (GAN.SENTENCE_LENGTH, 1)
    # input_shape = self.pos_sentences.shape[1:]
    # padding='same'
    # strides=1
    disc.add(Conv1D(2, 2, input_shape=shape))
    # disc.add(LeakyReLU(alpha=0.2))
    # disc.add(Dropout(dropout))

    # disc.add(Conv2D(depth*2, 5, strides=2, padding='same'))
    # disc.add(LeakyReLU(alpha=0.2))
    # disc.add(Dropout(dropout))
    #
    # disc.add(Conv2D(depth*4, 5, strides=2, padding='same'))
    # disc.add(LeakyReLU(alpha=0.2))
    # disc.add(Dropout(dropout))

    # disc.add(Conv2D(depth*8, 5, strides=1, padding='same'))
    # disc.add(LeakyReLU(alpha=0.2))
    # disc.add(Dropout(dropout))

    # Out: 1-dim probability
    disc.add(Flatten())
    disc.add(Dense(1))
    disc.add(Activation('sigmoid'))
    disc.summary()
    return disc


def no_conv_disc(shape=(3,1)):

    disc = Sequential()

    disc.add(Dense(6, input_shape=shape))
    disc.add(LeakyReLU(alpha=0.2))

    # Out: 1-dim probability
    disc.add(Flatten())
    disc.add(Dense(1))
    disc.add(Activation('sigmoid'))
    disc.summary()
    return disc
