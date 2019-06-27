from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K


def build_cae_model(time, feature):
    """
    build convolutional autoencoder model
    """
    input = Input(shape=(time, feature))

    # encoder
    net = Conv1D(32, 5, activation='relu', padding='same')(input)
    net = MaxPooling1D(3, padding='same')(net)
    net = Conv1D(16, 3, activation='relu', padding='same')(net)
    net = MaxPooling1D(2, padding='same')(net)
    net = Conv1D(8, 3, activation='relu', padding='same')(net)
    encoded = MaxPooling1D(2, padding='same', name='enc')(net)

    # decoder
    net = Conv1D(8, 3, activation='relu', padding='same')(encoded)
    net = UpSampling1D(2)(net)
    net = Conv1D(16, 3, activation='relu', padding='same')(net)
    net = UpSampling1D(2)(net)
    net = Conv1D(32, 5, activation='relu', padding='same')(net)
    net = UpSampling1D(3)(net)
    decoded = Conv1D(3, 3, activation='sigmoid', padding='same')(net)

    '''
    input_img = Input(shape=(32, 32, 3))

    # encoder
    net = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    net = MaxPooling2D((2, 2), padding='same')(net)
    net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
    net = MaxPooling2D((2, 2), padding='same')(net)
    net = Conv2D(4, (3, 3), activation='relu', padding='same')(net)
    encoded = MaxPooling2D((2, 2), padding='same', name='enc')(net)

    # decoder
    net = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    net = UpSampling2D((2, 2))(net)
    net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
    net = UpSampling2D((2, 2))(net)
    net = Conv2D(16, (3, 3), activation='relu', padding='same')(net)
    net = UpSampling2D((2, 2))(net)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(net)
    '''

    return Model(input, decoded)

