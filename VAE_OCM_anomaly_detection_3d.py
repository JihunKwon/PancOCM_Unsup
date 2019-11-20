# Anomaly detection with Variational Autoencoder
# Reference:
# [1] https://www.kaggle.com/kmader/vae-to-detect-anomalies-on-digits (unsupervised)
# [2] https://www.kaggle.com/sharaman/fraud-detection-with-variational-autoencoder (self-supervised)
# [3] https://techblog.nhn-techorus.com/archives/13499 (Japanese)

# This kernel trains a Variational Autoencoder in Keras with Gaussian input and output.

# import Keras and other libralies:
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import Pre_processing

from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve

import tensorflow as tf
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Lambda, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model
from keras import metrics


plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["font.size"] = 11

num_subject = 6
num_train = 3
num_ocm = 3
val_split = 0.2
sr_list = ['s1r1', 's1r1', 's1r2', 's1r2', 's2r1', 's2r1', 's2r2', 's2r2', 's3r1', 's3r1', 's3r2', 's3r2']

#ocm_channel = 3 # 0, 1, 2, and 3. 3 is combination of three OCMs

# Read the dataset
for fidx in range(0, num_subject*2, 2):
    Sub_run_name = sr_list[fidx]
    print('state1 importing...')
    ocm0_pre, ocm1_pre, ocm2_pre, filt_str = Pre_processing.preprocessing(fidx)
    print('state2 importing...')
    ocm0_post_, ocm1_post, ocm2_post, filt_str = Pre_processing.preprocessing(fidx+1)

    # Scaling max to 1
    ocm0_pre = ocm0_pre / np.max(ocm0_pre)
    ocm1_pre = ocm1_pre / np.max(ocm1_pre)
    ocm2_pre = ocm2_pre / np.max(ocm2_pre)
    ocm0_post_ = ocm0_post_ / np.max(ocm0_post_)
    ocm1_post = ocm1_post / np.max(ocm1_post)
    ocm2_post = ocm2_post / np.max(ocm2_post)

    # set all OCM shape same
    ocm1_pre = ocm1_pre[:, :np.size(ocm0_pre[1])]
    ocm2_pre = ocm2_pre[:, :np.size(ocm0_pre[1])]
    print('ocm0 shape pre', ocm0_pre.shape) # (350, 16934)
    print('ocm1 shape pre', ocm1_pre.shape)
    print('ocm2 shape pre', ocm2_pre.shape)

    ocm1_post = ocm1_post[:, :np.size(ocm0_post_[1])]
    ocm2_post = ocm2_post[:, :np.size(ocm0_post_[1])]
    print('ocm0 shape post', ocm0_post_.shape) # (350, 16484)
    print('ocm1 shape post', ocm1_post.shape) # (350, 16484)
    print('ocm2 shape post', ocm2_post.shape) # (350, 16484)

    ## Store all bh in state 2
    if ocm0_post_.shape[1] % 5 != 0: # if array size is not devided by 5 (# of bh), shorten it to reshape it later
        arr_temp = ocm0_post_.shape[1]
        while arr_temp % 5 != 0:
            arr_temp = arr_temp - 1
        ocm0_post_ = ocm0_post_[:, :arr_temp]
        ocm1_post = ocm1_post[:, :arr_temp]
        ocm2_post = ocm2_post[:, :arr_temp]

    print('ocm0 shape post_shortend', ocm0_post_.shape)  # (350, 16480)
    print('ocm1 shape post_shortend', ocm1_post.shape)  # (350, 16480)
    print('ocm2 shape post_shortend', ocm2_post.shape)  # (350, 16480)

    # Convert 2D data to 3D. 3rd dim is bh number
    ocm0_post_3d = np.zeros((ocm0_post_.shape[0], int(np.size(ocm0_post_[1])/5), 5))
    ocm1_post_3d = np.zeros((ocm0_post_.shape[0], int(np.size(ocm0_post_[1])/5), 5))
    ocm2_post_3d = np.zeros((ocm0_post_.shape[0], int(np.size(ocm0_post_[1])/5), 5))
    for bh in range(0, 5):
        ocm0_post_3d[:, :, bh] = ocm0_post_[:, bh*int(np.size(ocm0_post_[1])/5):(bh+1)*int(np.size(ocm0_post_[1])/5)]
        ocm1_post_3d[:, :, bh] = ocm1_post[:, bh*int(np.size(ocm0_post_[1])/5):(bh+1)*int(np.size(ocm0_post_[1])/5)]
        ocm2_post_3d[:, :, bh] = ocm2_post[:, bh*int(np.size(ocm0_post_[1])/5):(bh+1)*int(np.size(ocm0_post_[1])/5)]
    print('ocm0 shape post for state 2', ocm0_post_3d.shape) # (350, 3296, 5)
    print('ocm1 shape post for state 2', ocm1_post_3d.shape) # (350, 3296, 5)
    print('ocm2 shape post for state 2', ocm2_post_3d.shape) # (350, 3296, 5)

    # Split to train and test bh
    ocm0_train = ocm0_pre[:, :num_train*int(np.size(ocm0_pre[1])/5)]
    ocm1_train = ocm1_pre[:, :num_train*int(np.size(ocm1_pre[1])/5)]
    ocm2_train = ocm2_pre[:, :num_train*int(np.size(ocm2_pre[1])/5)]
    ocm0_test_pre = ocm0_pre[:, num_train*int(np.size(ocm0_pre[1])/5):-1]
    ocm1_test_pre = ocm1_pre[:, num_train*int(np.size(ocm1_pre[1])/5):-1]
    ocm2_test_pre = ocm2_pre[:, num_train*int(np.size(ocm2_pre[1])/5):-1]

    for ocm_ch in range(0, 4):
        # Chose which data type you want
        ocm_channel = ocm_ch # 0, 1, 2, and 3. 3 is combination of three OCMs
        # Use individual OCM
        if ocm_channel is 0: # OCM0
            ocm_train = ocm0_train
            ocm_test_pre = ocm0_test_pre
            ocm_test_post = ocm0_post_3d
        elif ocm_channel is 1: # OCM1
            ocm_train = ocm1_train
            ocm_test_pre = ocm1_test_pre
            ocm_test_post = ocm1_post_3d
        elif ocm_channel is 2: # OCM2
            ocm_train = ocm2_train
            ocm_test_pre = ocm2_test_pre
            ocm_test_post = ocm2_post_3d
        # Combine three OCM
        elif ocm_channel is 3: # All OCM
            ocm_train = np.concatenate([ocm0_train, ocm1_train, ocm2_train], 0)
            ocm_test_pre = np.concatenate([ocm0_test_pre, ocm1_test_pre, ocm2_test_pre], 0)
            ocm_test_post = np.concatenate([ocm0_post_3d, ocm1_post_3d, ocm2_post_3d], 0)

        print('ocm_train shape', ocm_train.shape) # (350, 10110)
        print('ocm_test_pre shape', ocm_test_pre.shape) # (350, 6743)
        print('ocm_test_post shape', ocm_test_post.shape) # (350, 3296, 5)
        print(type(ocm_train))

        x_train = ocm_train.T
        np.random.shuffle(x_train)
        x_train_val = x_train[int((1-val_split)*x_train.shape[0]):, :]
        x_train = x_train[:int((1-val_split)*x_train.shape[0]), :]

        x_test = ocm_test_pre.T # (6743, 1050)

        #anomaly_test = ocm_test_post.T # state 2
        anomaly_test = np.transpose(ocm_test_post, (1, 0, 2)) # When use 3D anomaly

        print('x_train shape', x_train.shape) # (10110, 1050)
        print('x_train_val shape', x_train_val.shape) #
        print('x_test shape', x_test.shape) # (23227, 1050)
        print('anomaly_test shape', anomaly_test.shape) # (23227, 350, 5)

        #scaler = preprocessing.StandardScaler()
        #x_train, x_test = scaler.fit_transform(x_train), scaler.fit_transform(x_test)

        # Build the model and print summary;
        original_shape = x_train.shape[1:]
        original_dim = np.prod(original_shape)
        latent_dim = 2  # number of latent variables to learn
        hidden_size = 128  # size of the hidden layer in encoder and decoder
        final_size = 64

        in_layer = Input(shape=(original_shape))
        t = BatchNormalization()(in_layer)
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

        z = Lambda(sampling, output_shape=(latent_dim,), name='z_sampled')([z_mean, z_log_var])

        decoder_f = Dense(final_size, activation='relu', name='decoder_hidden_f')
        decoder_h = Dense(hidden_size, activation='relu', name='decoder_hidden_h')
        decoder_mean = Dense(original_dim, activation='sigmoid', name='decoded_mean')

        f_decoded = decoder_f(z)
        h_decoded = decoder_h(f_decoded)
        x_decoded_mean = decoder_mean(h_decoded)

        # instantiate VAE model
        vae = Model(in_layer, x_decoded_mean)

        # Compute VAE loss
        xent_loss = original_dim * metrics.binary_crossentropy(in_layer, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)

        vae.add_loss(vae_loss)
        vae.compile(optimizer='rmsprop')
        vae.summary()
        plot_model(vae, to_file='model.png', show_shapes=True) # save model structure

        # train model
        n_epochs = 500
        batch_size = 128

        early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=1e-5) #stop training if loss does not decrease with at least 0.00001
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, min_delta=1e-5, factor=0.2) #reduce learning rate (divide it by 5 = multiply it by 0.2) if loss does not decrease with at least 0.00001

        callbacks = [early_stopping, reduce_lr]

        #collect training data in history object
        history = vae.fit(x_train,
                          shuffle=True,
                          epochs=n_epochs,
                          batch_size=batch_size,
                          validation_data=(x_train_val, None),
                          callbacks=callbacks)

        fig = plt.figure(figsize=(14, 6))
        ax = fig.gca()
        ax.plot(history.history['val_loss'], label="loss for training");
        ax.plot(history.history['loss'], label="loss for validation");
        ax.set_title('model loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend(loc='upper right')
        fig.savefig('history_'+Sub_run_name+'_ch'+str(ocm_channel)+'_'+filt_str+'.png')

        encoder = Model(in_layer, z_mean)
        # display a 2D plot of the classes in the latent space
        X_test_encoded = encoder.predict(x_test, batch_size=batch_size)

        # anomaly_encoded for 3D
        print('anomaly_test shape', anomaly_test.shape)
        anomaly_encoded = np.zeros((anomaly_test.shape[0], 2, 5))
        anomaly_test_2d = np.zeros((anomaly_test.shape[0], anomaly_test.shape[1]))

        for bh in range(0, 5):
            anomaly_test_2d = anomaly_test[:, :, bh]
            anomaly_encoded[:, :, bh] = encoder.predict(anomaly_test_2d, batch_size=batch_size)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        # visualize bh=4,5 (green) vs bh=6 (red)
        color=['b', 'g', 'r', 'c', 'm', 'y']
        for bh in range(0, 6):
            if bh is 0:
                plt.scatter(X_test_encoded[:,0], X_test_encoded[:,1], c=color[bh], alpha = 0.7, s=5, label='Bh=4 and 5')
            else:
                plt.scatter(anomaly_encoded[:,0, bh-1], anomaly_encoded[:,1, bh-1], c=color[bh], alpha = 0.7, s=5, label='Bh='+str(bh+5))
        #plt.scatter(np.concatenate([X_test_encoded[:,0], anomaly_encoded[:,0, 0]],0), np.concatenate([X_test_encoded[:,1], anomaly_encoded[:,1, 0]],0),
        #            c=(['g']*X_test_encoded.shape[0])+['r']*anomaly_encoded.shape[0], alpha = 0.5)
        ax.legend()
        #plt.colorbar()
        ax.set_ylabel('Latent variable (${z_{1}}$)')
        ax.set_xlabel('Latent variable (${z_{2}}$)')
        fig.savefig('zmap_'+Sub_run_name+'_ch'+str(ocm_channel)+'_'+filt_str+'.png')

        ## Evaluate our detector
        model_mse = lambda t: np.mean(np.square(t - vae.predict(t, batch_size = batch_size)), (1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
        ax1.plot([0, 1], [0, 1], 'k-', label='Random Guessing')

        TPR_my = [0, 0, 0, 0, 0]
        for bh in range(0, 5):
            anomaly_test_2d = anomaly_test[:, :, bh]
            mse_score = np.concatenate([model_mse(x_test), model_mse(anomaly_test_2d)],0)
            true_label = [0]*x_test.shape[0]+[1]*anomaly_test_2d.shape[0]

            if roc_auc_score(true_label, mse_score)<0.5:
                mse_score *= -1

            fpr, tpr, thresholds = roc_curve(true_label, mse_score)
            auc_score = roc_auc_score(true_label, mse_score)
            ax1.plot(fpr, tpr, label='bh='+str(bh+6)+' ROC Curve (%2.2f)' % auc_score)

            ## Get a specific TPR corresponds to the FPR we define
            FPR_my = 0.1
            temp_arg = np.where(fpr < FPR_my)  # get argument where lower than FPR_my
            TPR_my[bh] = tpr[len(temp_arg[0])]  # Get TPR corresponds to FPR_my
            print(TPR_my[bh])

            ## save parameters needed to calculate ROC curve
            fname = 'roc_data_' + Sub_run_name + '_ch' + str(ocm_channel)+ '_bh'+ str(bh) + '_' + filt_str + '.pkl'
            with open(fname, 'wb') as f:
                pickle.dump([true_label, mse_score], f)

        ax1.legend();
        ax1.set_xlabel('FPR')
        ax1.set_ylabel('TPR')

        ax2.plot(np.linspace(6, 10, 5), TPR_my[:], marker='o', label='TPR at FPR = '+str(FPR_my))
        plt.xticks(np.arange(6, 11, 1))
        plt.ylim(0, 1)
        if ocm_channel is 3:
            ax2.set_title('All OCM combined')
        else:
            ax2.set_title('OCM'+str(ocm_channel))
        ax2.set_xlabel('Breath-hold number')
        ax2.set_ylabel('TPR when FPR is 0.1')
        fig.savefig('roc_'+Sub_run_name+'_ch'+str(ocm_channel)+'_'+filt_str+'.png')

        model_json_str = vae.to_json()
        open('model_'+Sub_run_name+'_ch'+str(ocm_channel)+'_'+filt_str+'.json', 'w').write(model_json_str)
        vae.save_weights('model_'+Sub_run_name+'_ch'+str(ocm_channel)+'_'+filt_str+'_weights.h5');
