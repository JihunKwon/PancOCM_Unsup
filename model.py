from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K


def build_cae_model(time, feature):
    """
    build convolutional autoencoder model
    """
    input = Input(shape=(time, feature))

    # encoder
    net = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(input)
    net = MaxPooling1D(pool_size=2, padding='same')(net)
    net = Conv1D(8, (3, 3), activation='relu', padding='same')(net)
    net = MaxPooling1D((2, 2), padding='same')(net)
    net = Conv1D(4, (3, 3), activation='relu', padding='same')(net)
    encoded = MaxPooling1D((2, 2), padding='same', name='enc')(net)

    # decoder
    net = Conv1D(4, (3, 3), activation='relu', padding='same')(encoded)
    net = UpSampling1D((2, 2))(net)
    net = Conv1D(8, (3, 3), activation='relu', padding='same')(net)
    net = UpSampling1D((2, 2))(net)
    net = Conv1D(16, (3, 3), activation='relu', padding='same')(net)
    net = UpSampling1D((2, 2))(net)
    decoded = Conv1D(3, (3, 3), activation='sigmoid', padding='same')(net)

    return Model(input, decoded)
