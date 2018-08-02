import numpy as np
import pandas
import random
import tensorflow as tf
from keras.layers    import Input, Dense, Flatten, Reshape,Lambda
from keras.layers    import Conv3D,Cropping3D,UpSampling3D
from keras.models    import Model
from keras.losses import binary_crossentropy, kullback_leibler_divergence
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import Adam, Nadam, SGD

from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.regularizers import l2

from keras.utils import HDF5Matrix, Sequence, multi_gpu_model

import keras.backend as K

class Gen(Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size
    def __len__(self):
        return (len(self.x) // self.batch_size)-1
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array(batch_x)
        batch_x = batch_x[:,:,:,:,:2]
        batch_x[:,:,:,:,0] /= 8000.0
        batch_x[:,:,:,:,1] += np.pi
        batch_x[:,:,:,:,1] /= (2.0*np.pi + 1e-5)
        return batch_x,batch_x

def sampling(args):
    encoder_mean, encoder_log_sigma = args
    epsilon = K.random_normal_variable(shape=(batch_size, 2048), mean=0., scale=0.1)
    return encoder_mean + K.exp(encoder_log_sigma) * epsilon

def vae_loss(y_true, y_pred):
    kl = kullback_leibler_divergence(y_true, y_pred)
    xe = binary_crossentropy(y_true, y_pred)
    return kl+xe

batch_size = 10
L = 49600
X_train = HDF5Matrix("dataset.h5", "X", start=0, end=None)
Y_train = HDF5Matrix("dataset.h5", "Y", start=0, end=L)

X_test  = HDF5Matrix("dataset.h5", "X", start=L, end=75200)
Y_test  = HDF5Matrix("dataset.h5", "Y", start=L, end=75200)

gen_train = Gen(X_train,batch_size=batch_size)
gen_test  = Gen(X_test ,batch_size=batch_size)


x_inp  = Input(batch_shape=(batch_size,31,31,31,2))
encoder = Conv3D(16, kernel_size=7,strides=2,padding="same")(x_inp)
encoder = LeakyReLU(0.3)(encoder)
encoder = Conv3D(32, kernel_size=3,strides=2,padding="same")(encoder)
encoder = LeakyReLU(0.3)(encoder)
encoder = Conv3D(64, kernel_size=3,strides=2,padding="same")(encoder)
encoder = LeakyReLU(0.3)(encoder)
encoder = Flatten()(encoder)
encoder = Dense(4096,)(encoder)
encoder = LeakyReLU(0.3)(encoder)
encoder = Dense(2048, activation='relu')(encoder)
encoder_mean = Dense(2048)(encoder)
encoder_log_sigma = Dense(2048)(encoder)
encoded = Lambda(sampling)([encoder_mean, encoder_log_sigma])
### if generator is needed
#encoded_inp = Input((2048,))
###
decoder = Dense(4096, activation='relu')(encoded)
decoder = Reshape((4,4,4,64))(decoder)
decoder = Conv3D(64, kernel_size=3,strides=1,padding="same")(decoder)
decoder = LeakyReLU(0.3)(decoder)
decoder = UpSampling3D(size=2)(decoder)
decoder = Conv3D(32, kernel_size=3,strides=1,padding="same")(decoder)
decoder = LeakyReLU(0.3)(decoder)
decoder = UpSampling3D(size=2)(decoder)
decoder = Conv3D(16, kernel_size=3,strides=1,padding="same")(decoder)
decoder = LeakyReLU(0.3)(decoder)
decoder = UpSampling3D(size=2)(decoder)
decoder = Conv3D(2 , kernel_size=1,strides=1,padding="same")(decoder)
decoded = Cropping3D(cropping=((1,0),(1,0),(1,0)))(decoder)

VAE= Model(x_inp, decoded)
VAE.summary()
Encoder = Model(x_inp,encoded)

VAE.compile(optimizer="rmsprop", loss=vae_loss, metrics=["mean_absolute_percentage_error"])

chkpntr = ModelCheckpoint(filepath="VAE.h5", save_best_only=True,verbose=1,save_weights_only=True)
lrrdcr  = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-7,verbose=1)
erlstpp = EarlyStopping(monitor='val_loss', patience=4)
_CALLBACKS = [chkpntr,erlstpp,lrrdcr]

#VAE.load_weights("VAE.h5")
VAE.fit_generator(gen_train,
                  epochs=500,
                  steps_per_epoch=len(gen_train),
                  validation_data=gen_test,
                  validation_steps=len(gen_test),
                  verbose=1,
                  callbacks=_CALLBACKS)

Encoder.save_weights("Encoder.h5")


# mofnn.load_weights("model.h5")
# 
# DATA = pandas.DataFrame(columns=["Calculated CH4 ads. 1 bar","Predicted CH4 ads. 1 bar"])
# 
# X = HDF5Matrix("dataset.h5", "X",end=82730)
# Y = HDF5Matrix("dataset.h5", "Y",end=82730)
# 
# pred = mofnn.predict(X,batch_size=batch_size,verbose=1)
# DATA.loc[:,"Calculated CH4 ads. 1 bar"] = np.array(Y)[:,1]
# DATA.loc[:,"Predicted CH4 ads. 1 bar" ] = pred*150.0
# DATA.to_csv("predicted_1bar.csv")

