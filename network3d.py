from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Convolution2D ,Convolution3D,MaxPooling3D,Activation,PReLU,LeakyReLU
#from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
import  keras
import h5py
from keras.optimizers import Adam,SGD,Nadam,Adadelta,Adagrad,RMSprop,Adamax

def srcnn3(input_shape=(33,33,110,1)):
    print '82'
    model = Sequential()
    model.add(Convolution3D(64, 9, 9, 7, input_shape=input_shape, activation='relu'))
    model.add(Convolution3D(32, 1, 1, 1, activation='relu'))
    model.add(Convolution3D(9, 1, 1, 1, activation='relu'))
    model.add(Convolution3D(1, 5, 5, 3))
    model.compile(Adam(lr=0.00005), 'mse')
    return model


def srcnn(input_shape=(33,33,111,1)):
    print( '82')
    model = Sequential()
    model.add(Convolution3D(64, 9, 9, 7, input_shape=input_shape, activation='relu'))
    model.add(Convolution3D(32, 1, 1, 1, activation='relu'))
    model.add(Convolution3D(9, 1, 1, 1, activation='relu'))
    model.add(Convolution3D(1, 5, 5, 3))
    model.compile(Adam(lr=0.00005), 'mse')
    return model

def srcnn2(input_shape=(33,33,232,1)):
    print( '82')
    model = Sequential()
    model.add(Convolution3D(64, 9, 9, 7, input_shape=input_shape, activation='relu'))
    model.add(Convolution3D(32, 1, 1, 1, activation='relu'))
    model.add(Convolution3D(9, 1, 1, 1, activation='relu'))
    model.add(Convolution3D(1, 5, 5, 3))
    model.compile(Adam(lr=0.00005), 'mse')
    return model

def srcnn3(input_shape=(33,33,111,1)):
    print '110'

    model=srcnn(input_shape=input_shape)
    f = h5py.File('./save_all/save2/model_pa_300.h5', mode='r')
    model.load_weights_from_hdf5_group(f['model_weights'])
    return model

def srcnn4(input_shape=(33,33,218,1)):
    print( '82')
    model = Sequential()
    model.add(Convolution3D(64, 9, 9, 7, input_shape=input_shape, activation='relu'))
    model.add(Convolution3D(32, 1, 1, 1, activation='relu'))
    model.add(Convolution3D(9, 1, 1, 1, activation='relu'))
    model.add(Convolution3D(1, 5, 5, 3))
    model.compile(Adam(lr=0.00005), 'mse')
    return model