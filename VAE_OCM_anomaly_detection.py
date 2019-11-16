# Anomaly detection with Variational Autoencoder
# Reference:
# [1] https://www.kaggle.com/sharaman/fraud-detection-with-variational-autoencoder
# [2] https://www.kaggle.com/kmader/vae-to-detect-anomalies-on-digits

# This kernel trains a Variational Autoencoder in Keras with Gaussian input and output.

# import Keras and other libralies:
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Pre_processing

from sklearn import preprocessing, metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model
#from keras import metrics

num_subject = 1
num_train = 3
num_test = 1
num_ocm = 3

ocm_channel = 1 # 'all', 0, 1, and 2. All is combination of three OCMs

# Read the dataset
for fidx in range(0, num_subject*2, 2):
    print('state1 importing...')
    ocm0_pre, ocm1_pre, ocm2_pre = Pre_processing.preprocessing(fidx)
    print('state2 importing...')
    ocm0_post, ocm1_post, ocm2_post = Pre_processing.preprocessing(fidx+1)

    # set all OCM shape same
    ocm1_pre = ocm1_pre[:, :np.size(ocm0_pre[1])]
    ocm2_pre = ocm2_pre[:, :np.size(ocm0_pre[1])]
    print('ocm0 shape pre', ocm0_pre.shape) # (350, 16934)

    ocm1_post = ocm1_post[:, :np.size(ocm0_post[1])]
    ocm2_post = ocm2_post[:, :np.size(ocm0_post[1])]
    print('ocm0 shape post', ocm0_post.shape) # (350, 16934)

    ## Use only bh=5 for state 2
    ocm0_post = ocm0_post[:, :num_test*int(np.size(ocm0_post[1])/5)]
    ocm1_post = ocm1_post[:, :num_test*int(np.size(ocm0_post[1])/5)]
    ocm2_post = ocm2_post[:, :num_test*int(np.size(ocm0_post[1])/5)]
    print('ocm0 shape post for bh5', ocm0_post.shape)

    # Split to train and test bh
    ocm0_train = ocm0_pre[:, :num_train*int(np.size(ocm0_pre[1])/5)]
    ocm1_train = ocm1_pre[:, :num_train*int(np.size(ocm1_pre[1])/5)]
    ocm2_train = ocm2_pre[:, :num_train*int(np.size(ocm2_pre[1])/5)]
    ocm0_test_pre = ocm0_pre[:, num_train*int(np.size(ocm0_pre[1])/5):-1]
    ocm1_test_pre = ocm1_pre[:, num_train*int(np.size(ocm1_pre[1])/5):-1]
    ocm2_test_pre = ocm2_pre[:, num_train*int(np.size(ocm2_pre[1])/5):-1]

    # Chose which data type you want
    if ocm_channel is 'all':
        # Combine three OCM
        ocm_train = np.concatenate([ocm0_train, ocm1_train, ocm2_train], 0)
        ocm_test_pre = np.concatenate([ocm0_test_pre, ocm1_test_pre, ocm2_test_pre], 0)
        ocm_test_post = np.concatenate([ocm0_post, ocm1_post, ocm2_post], 0)
    else:
        # Use individual OCM
        if ocm_channel is 0:
            ocm_train = ocm0_train
            ocm_test_pre = ocm0_test_pre
            ocm_test_post = ocm0_post
        elif ocm_channel is 1:
            ocm_train = ocm1_train
            ocm_test_pre = ocm1_test_pre
            ocm_test_post = ocm1_post
        elif ocm_channel is 2:
            ocm_train = ocm2_train
            ocm_test_pre = ocm2_test_pre
            ocm_test_post = ocm2_post

    print('ocm_train shape', ocm_train.shape) # (1050, 10110)
    print('ocm_test_pre shape', ocm_test_pre.shape) # (1050, 6743)
    print('ocm_test_post shape', ocm_test_post.shape) # (1050, 16484)
    print(type(ocm_train))

    x_train = ocm_train.T
    #np.random.shuffle(x_train) # shuffle traces so that validation set is taken from each bh=1, 2, and 3
    x_test = np.hstack([ocm_test_pre, ocm_test_post]).T
    y_train = np.zeros((np.size(ocm_train[1])))

    y_test_pre = np.zeros(np.size(ocm_test_pre[1])) # state 1
    y_test_post = np.ones(np.size(ocm_test_post[1])) # state 2
    y_test = np.concatenate([y_test_pre, y_test_post], axis=0)

    print('x_train shape', x_train.shape) # (10110, 1050)
    print('x_test shape', x_test.shape) # (23227, 1050)
    print('y_train shape', y_train.shape) # (10110,)
    print('y_test shape', y_test.shape) # (23227,)

    scaler = preprocessing.StandardScaler()
    x_train, x_test = scaler.fit_transform(x_train), scaler.fit_transform(x_test)

    # Build the model and print summary;
    original_shape = x_train.shape[1:]
    input_size = np.prod(original_shape)
    latent_dim = 2  # number of latent variables to learn
    hidden_size = 128  # size of the hidden layer in encoder and decoder
    final_size = 64

    x = Input(shape=(input_size,))
    t = BatchNormalization()(x)
    t = Dense(hidden_size, activation='relu', name='encoder_hidden_h')(t)
    t = BatchNormalization()(t)
    t = Dense(final_size, activation='relu', name='encoder_hidden_f')(t)
    t = BatchNormalization()(t)

    z_mean = Dense(latent_dim, name='z_mean')(t)
    z_log_var = Dense(latent_dim, name='z_log_var')(t)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, name='z_sampled')([z_mean, z_log_var])


    decoder_f = Dense(final_size, activation='relu', name='decoder_hidden_f')
    decoder_h = Dense(hidden_size, activation='relu', name='decoder_hidden_h')
    decoder_mean = Dense(input_size, activation='sigmoid', name='decoded_mean')

    f_decoded = decoder_f(z)
    h_decoded = decoder_h(f_decoded)
    x_decoded_mean = decoder_mean(h_decoded)

    # instantiate VAE model
    vae = Model(x, x_decoded_mean)

    # Compute VAE loss
    def rec_loss(y_true, y_pred):
        return K.sum(K.square(y_true - y_pred), axis=-1)

    def kl_loss(y_true, y_pred):
        return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    def vae_loss(x, decoded_mean):
        rec_loss = K.sum(K.square(x - decoded_mean), axis=-1)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean((rec_loss + kl_loss) / 2)

    vae.compile(optimizer=Adam(lr=1e-2), loss=vae_loss, metrics=[rec_loss, kl_loss])
    vae.summary()
    plot_model(vae, to_file='vae_ocm.png', show_shapes=True)

    # train model
    n_epochs = 30
    batch_size = 128

    early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=1e-5) #stop training if loss does not decrease with at least 0.00001
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, min_delta=1e-5, factor=0.2) #reduce learning rate (divide it by 5 = multiply it by 0.2) if loss does not decrease with at least 0.00001

    callbacks = [early_stopping, reduce_lr]

    #collect training data in history object
    history = vae.fit(x_train, x_train,
                      #validation_data=(x_val, x_val),
                      batch_size=batch_size, epochs=n_epochs,
                      callbacks=callbacks)

    encoder = Model(x, z_mean)
    # display a 2D plot of the classes in the latent space
    X_test_encoded = encoder.predict(x_test, batch_size=batch_size)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    plt.scatter(X_test_encoded[:, 0], X_test_encoded[:, 1], c=y_test, alpha=0.3, s=10)
    #plt.scatter(X_test_encoded[:, 0], X_test_encoded[:, 1], facecolors=y_test, edgecolors=y_test, s=10)
    plt.colorbar()
    fig.savefig('result_2d_fidx'+str(fidx)+'_ch'+str(ocm_channel)+'.png')


    x_data = x_train
    y_data = y_train
    encoder = Model(x, z_mean)
    with_labels = np.concatenate([x_data, np.reshape(y_data, (-1, 1))], axis=1) #concatenate x and y to be able to filter by class

    # Apply simple Linear Discriminant Analysis to learned features (the latent representations):
    with_labels_encoded = encoder.predict(with_labels[:, :-1], batch_size=128)

    X = with_labels_encoded
    y = with_labels[:, -1]

    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)

    pred = clf.predict(X)

#    print("AUC(ROC): " + str(metrics.roc_auc_score(y, pred)))
    print("Precision: " + str(metrics.precision_score(y, pred)))
    print("Recall: " + str(metrics.recall_score(y, pred)))
    print("F1 score: " + str(metrics.f1_score(y, pred)))

    tn, fp, fn, tp = metrics.confusion_matrix(y, pred).ravel()

    print("False positives: " + str(fp))
    print("True positives: " + str(tp))
    print("False negatives: " + str(fn))
    print("True negateives: " + str(tn))

