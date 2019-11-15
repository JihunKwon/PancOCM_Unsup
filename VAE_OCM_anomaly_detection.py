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

num_subject = 1
num_train = 3
num_test = 10 - num_train
num_ocm = 3

# Read the dataset
for fidx in range(0, num_subject*2):
    ocm0, ocm1, ocm2 = Pre_processing.preprocessing(fidx)

    # set all OCM shape same
    ocm1 = ocm1[:, :np.size(ocm0[1])]
    ocm2 = ocm2[:, :np.size(ocm0[1])]
    print('ocm0 shape', ocm0.shape) # (350, 16934)

    # Split to train and test bh
    ocm0_train = ocm0[:, :num_train*int(np.size(ocm0[1])/5)]
    ocm1_train = ocm1[:, :num_train*int(np.size(ocm1[1])/5)]
    ocm2_train = ocm2[:, :num_train*int(np.size(ocm2[1])/5)]
    ocm0_test = ocm0[:, num_train*int(np.size(ocm0[1])/5):-1]
    ocm1_test = ocm1[:, num_train*int(np.size(ocm1[1])/5):-1]
    ocm2_test = ocm2[:, num_train*int(np.size(ocm2[1])/5):-1]

    # Combine three OCM
    ocm_train = np.concatenate([ocm0_train, ocm1_train, ocm2_train], 0)
    print('ocm_train shape', ocm_train.shape)



csv = pd.read_csv('creditcard.csv')
test_split = 0.3 # portion of data used for testing
val_split = 0.2  # portion of training data used for validation

data = csv.astype('float32').copy()
data.sort_values('Time', inplace=True)
print('data shape', data.shape)

first_test = int(data.shape[0] * (1 - test_split))
first_val = int(first_test * (1 - val_split))

train = data.iloc[:first_val]
val = data.iloc[first_val:first_test]
test = data.iloc[first_test:]

x_train_df, x_val_df, x_test_df = train.iloc[:, 1:-2], val.iloc[:, 1:-2], test.iloc[:, 1:-2]
y_train_df, y_val_df, y_test_df = train.iloc[:, -1], val.iloc[:, -1], test.iloc[:, -1]

x_train, x_val, x_test = x_train_df.values, x_val_df.values, x_test_df.values
y_train, y_val, y_test = y_train_df.values, y_val_df.values, y_test_df.values

scaler = preprocessing.StandardScaler()

x_train, x_val, x_test = scaler.fit_transform(x_train), scaler.fit_transform(x_val), scaler.fit_transform(x_test)

# Build the model and print summary;
hidden_size = 16 #size of the hidden layer in encoder and decoder
latent_dim = 2 #number of latent variables to learn

input_dim = x_train.shape[1]

x = Input(shape=(input_dim,))
t = BatchNormalization()(x)
t = Dense(hidden_size, activation='tanh' , name='encoder_hidden')(t)
t = BatchNormalization()(t)

z_mean = Dense(latent_dim, name='z_mean')(t)
z_log_var = Dense(latent_dim, name='z_log_var')(t)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, name='z_sampled')([z_mean, z_log_var])
#t = BatchNormalization()(z)

t = Dense(hidden_size, activation='tanh', name='decoder_hidden')(z)
#t = BatchNormalization()(t)

decoded_mean = Dense(input_dim, activation=None, name='decoded_mean')(t)

vae = Model(x, decoded_mean)

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
plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

# train model
n_epochs = 30
batch_size = 128

early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=1e-5) #stop training if loss does not decrease with at least 0.00001
reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, min_delta=1e-5, factor=0.2) #reduce learning rate (divide it by 5 = multiply it by 0.2) if loss does not decrease with at least 0.00001

callbacks = [early_stopping, reduce_lr]

#collect training data in history object
history = vae.fit(x_train, x_train,
                  validation_data=(x_val, x_val),
                  batch_size=batch_size, epochs=n_epochs,
                  callbacks=callbacks)


fig = plt.figure(figsize=(14, 6))
ax = fig.gca()
ax.plot(history.history['rec_loss']);
ax.plot(history.history['val_rec_loss']);

x_data = x_train
y_data = y_train

encoder = Model(x, z_mean)
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

with_labels = np.concatenate([x_data, np.reshape(y_data, (-1, 1))], axis=1) #concatenate x and y to be able to filter by class

normal = with_labels[np.where(with_labels[:, -1] == 0)] #filter normal instances
anomalous = with_labels[np.where(with_labels[:, -1] == 1)] #filter anomalous instances

normal_encoded = encoder.predict(normal[:, :-1], batch_size=128)
anomalous_encoded = encoder.predict(anomalous[:, :-1], batch_size=128)

fig = plt.figure(figsize=(6, 6))
ax = fig.gca()

#semicolon at the end hides unnecessary output
ax.scatter(normal_encoded[:, 0], normal_encoded[:, 1]);
ax.scatter(anomalous_encoded[:, 0], anomalous_encoded[:, 1]);
fig.savefig('result.png')

# Apply simple Linear Discriminant Analysis to learned features (the latent representations):
with_labels_encoded = encoder.predict(with_labels[:, :-1], batch_size=128)

X = with_labels_encoded
y = with_labels[:, -1]

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

pred = clf.predict(X)

print("AUC(ROC): " + str(metrics.roc_auc_score(y, pred)))
print("Precision: " + str(metrics.precision_score(y, pred)))
print("Recall: " + str(metrics.recall_score(y, pred)))
print("F1 score: " + str(metrics.f1_score(y, pred)))

tn, fp, fn, tp = metrics.confusion_matrix(y, pred).ravel()

print("False positives: " + str(fp))
print("True positives: " + str(tp))
print("False negatives: " + str(fn))
print("True negateives: " + str(tn))