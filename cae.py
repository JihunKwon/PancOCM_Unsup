'''
Anomaly Detection by autoencoder. For more detail, see (https://github.com/hiram64/ocsvm-anomaly-detection)
'''

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import seaborn as sns

from model import build_cae_model
from keras.models import Model
from keras import backend as K
import time

plt.rcParams['font.family'] ='sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

start = time.time()

def parse_args():
    parser = argparse.ArgumentParser(description='Train Convolutional AutoEncoder and inference')
    parser.add_argument('--num_epoch', default=1, type=int, help='the number of epochs')
    parser.add_argument('--batch_size', default=100, type=int, help='mini batch size')
    parser.add_argument('--output_path', default='./data/bef_cae.npz', type=str, help='path to directory to output')

    args = parser.parse_args()

    return args

def load_data(sub_run):
    """load data
    """
    #sr_name = 'ocm012_' + Sub_run + '.pkl'  # filtered
    #sr_name = 'Raw_det_ocm012_' + sub_run + '.pkl'  # raw
    sr_name = 'Raw_det_ocm012_' + sub_run + '_d384.pkl'  # raw

    with open(sr_name, 'rb') as f:
        ocm0, ocm1, ocm2 = pickle.load(f)

    return ocm0, ocm1, ocm2


def process_split(ocm0, ocm1, ocm2):
    print(ocm0.shape)  # (350, 10245, 3)
    depth = np.linspace(0, ocm0.shape[0] - 1, ocm0.shape[0])
    num_max = ocm0.shape[1]  # Total num of traces
    bh = ocm0.shape[1]//5

    # Before water intake
    phase = 0
    ocm0_bef = ocm0[:, 0:bh * 4, phase]  # Train with bh=1~4, and use bh=5 later as the test set.
    ocm1_bef = ocm1[:, 0:bh * 4, phase]
    ocm2_bef = ocm2[:, 0:bh * 4, phase]
    # After water intake
    phase = 1
    ocm0_aft = ocm0[:, 0:bh * 4, phase]
    ocm1_aft = ocm1[:, 0:bh * 4, phase]
    ocm2_aft = ocm2[:, 0:bh * 4, phase]
    # allocate to one variable
    ocm_bef = np.zeros((ocm0_bef.shape[0], ocm0_bef.shape[1], 3))
    ocm_aft = np.zeros((ocm0_aft.shape[0], ocm0_aft.shape[1], 3))

    ocm_bef[:, :, 0] = ocm0_bef[:, :]
    ocm_bef[:, :, 1] = ocm1_bef[:, :]
    ocm_bef[:, :, 2] = ocm2_bef[:, :]
    ocm_aft[:, :, 0] = ocm0_aft[:, :]
    ocm_aft[:, :, 1] = ocm1_aft[:, :]
    ocm_aft[:, :, 2] = ocm2_aft[:, :]
    print(ocm_bef.shape)  # (350, 10245, 3)

    # Transpose
    ocm_bef = np.einsum('abc->bac', ocm_bef)
    ocm_aft = np.einsum('abc->bac', ocm_aft)
    print('before:', ocm_bef.shape)
    print('after:', ocm_bef.shape)

    return ocm_bef, ocm_aft


def flat_feature(enc_out):
    enc_out_flat = []

    for i, con in enumerate(enc_out):
        s1, s2 = con.shape
        enc_out_flat.append(con.reshape((s1 * s2,)))

    return np.array(enc_out_flat)


def main():
    """main function"""
    # Hyperparameters
    args = parse_args()
    epochs = args.num_epoch
    batch_size = args.batch_size
    output_path = args.output_path

    # sr_list = ['s1r1', 's1r2', 's2r1', 's2r2', 's3r1', 's3r2']
    sr_list = ['s2r2']

    for fidx in range(0, np.size(sr_list)):
        Sub_run = sr_list[fidx]
        ocm0, ocm1, ocm2 = load_data(Sub_run)

        # Split and process. Get Before and After
        ocm_bef, ocm_aft = process_split(ocm0, ocm1, ocm2)
        n_timesteps, n_features = ocm_bef.shape[1], ocm_bef.shape[2]

        # build model and train
        autoencoder = build_cae_model(n_timesteps, n_features)
        autoencoder.summary()
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        print('here1!')
        autoencoder.fit(ocm_bef, ocm_bef,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True)
        print('here2!')

        # inference from encoder
        layer_name = 'enc'
        encoded_layer = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(layer_name).output)
        enc_out = encoded_layer.predict(ocm_bef)
        print('enc_out:', enc_out.shape)

        enc_out = flat_feature(enc_out)  # 次元要チェック

        # save cae output
        np.savez(output_path, ae_out=enc_out, labels=ocm_aft)  # ラベル、これじゃダメなはず


if __name__ == '__main__':
    main()

print((time.time() - start)/60, 'min')