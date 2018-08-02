# Sript using the trained encoder part of a CVAE
# to generate the encoded dataset
import numpy as np
import pandas
import random
import tensorflow as tf
from keras.layers    import Input, Dense, Flatten, Reshape
from keras.layers    import Conv3D,Cropping3D,UpSampling3D
from keras.models    import Model
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import Adam, Nadam, SGD
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.regularizers import l2

from keras.utils import HDF5Matrix, Sequence, multi_gpu_model

import h5py


def Encoder(batch_size):
    x_inp  = Input(batch_shape=(batch_size,31,31,31,2))
    x = Conv3D(16, kernel_size=7,strides=2,padding="same")(x_inp)
    x = LeakyReLU(0.3)(x)
    x = Conv3D(32, kernel_size=3,strides=2,padding="same")(x)
    x = LeakyReLU(0.3)(x)
    x = Conv3D(64, kernel_size=3,strides=2,padding="same")(x)
    x = LeakyReLU(0.3)(x)
    x = Flatten()(x)
    x = Dense(4096,)(x)
    x = LeakyReLU(0.3)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(2048)(x)
    model = Model(x_inp,x)
    return model

encoder = Encoder(batch_size=100)
encoder.load_weights("Encoder.h5")
encoder.summary()

fhkl = HDF5Matrix("dataset.h5","X",end=82700)
ads  = HDF5Matrix("dataset.h5","Y",end=82700)
L = len(ads)

encoded_fhkl = encoder.predict(fhkl,batch_size=100,verbose=1)

with h5py.File("dataset-encoded.h5","w") as outh5:
    X = outh5.create_dataset("X",(L,2048))
    Y = outh5.create_dataset("Y",(L,2))
    X[:,:] = np.asarray(encoded_fhkl)
    Y[:,:] = np.asarray(ads)    
