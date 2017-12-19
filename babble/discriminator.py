
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

    disc.add(Conv1D(2, 2, input_shape=shape))

    disc.add(Flatten())
    disc.add(Dense(1))
    disc.add(Activation('sigmoid'))
    disc.summary()
    return disc


def no_conv_disc(shape=(3,1)):

    disc = Sequential()

    disc.add(Dense(6, input_shape=shape))
    disc.add(LeakyReLU(alpha=0.2))

    disc.add(Flatten())
    disc.add(Dense(1))
    disc.add(Activation('sigmoid'))
    disc.summary()
    return disc
