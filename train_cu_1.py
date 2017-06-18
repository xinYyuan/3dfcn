
import os
import h5py
import network
import  network2d
import network_decov
import numpy as np
from scipy import misc

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Convolution2D
from keras.optimizers import Adam

import  network_test
import network2d

import argparse
import keras


def train():

    model = network2d.srcnn1 ()

    model.summary()

    #output_file = './data.h5'
    h5f = h5py.File(args.input_data, 'r')
    X = h5f['data'].value
    y = h5f['label'].value

    X = np.swapaxes(X, 1, 2)
    X = np.swapaxes(X, 2, 3)

    y = np.swapaxes(y, 1, 2)
    y = np.swapaxes(y, 2, 3)

    y=y[:,6:-6,6:-6,:]

    print X.shape
    print y.shape


    n_epoch = args.n_epoch

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    for epoch in range(0, n_epoch,50):
        model.fit(X, y, batch_size=64, nb_epoch=50, shuffle='batch')
        if args.save:
            print("Saving model ", epoch + 50)
            model.save(os.path.join(args.save, 'model_1bands_%d.h5' %(epoch+50)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--save',
                        default='./save_cu',
                        dest='save',
                        type=str,
                        nargs=1,
                        help="Path to save the checkpoints to")
    parser.add_argument('-D', '--data',
                        default='/home/yx/HSSR_2.0/data/cuprite/cu_train(mirror9)_1bands.mat',
                        dest='input_data',
                        type=str,
                        nargs=1,
                        help="Training data directory")
    parser.add_argument('-E', '--epoch',
                        default=150,
                        dest='n_epoch',
                        type=int,
                        nargs=1,
                        help="Training epochs must be a multiple of 5")
    args = parser.parse_args()
    print(args)
    train()
