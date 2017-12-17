
###############################################################################
# Imports                                                                     #
###############################################################################
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Reshape, Dropout, UpSampling2D, Conv2DTranspose, Flatten


###############################################################################
# Generators                                                                  #
###############################################################################

def basic_generator():

    gen = Sequential()

    dim = 3
    depth = 16
    dropout = 0.4

    gen.add(Dense(dim*dim*depth, input_dim=100))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(Activation('relu'))
    gen.add(Reshape((dim, dim, depth)))
    gen.add(Dropout(dropout))

    gen.add(UpSampling2D())
    gen.add(Conv2DTranspose(int(depth/2), 4, padding='same'))

    gen.add(Flatten())
    gen.add(Dense(3))
    gen.add(Activation('sigmoid'))
    gen.add(Reshape((3, 1)))

    gen.summary()
    return gen
