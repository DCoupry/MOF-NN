# RESNET-like for property prediction. This is for CH4 adsorption at 1 and 100 bar
# but is easily adapted
import numpy as np
import threading
import pandas
import seaborn
import random

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from keras.layers    import Input, Dense, Lambda, Flatten, Reshape, Layer, Dropout
from keras.layers    import BatchNormalization, Activation, Concatenate, Add, Multiply
from keras.layers    import Conv3D,MaxPooling3D,AveragePooling3D, GaussianNoise, AlphaDropout, Cropping3D
from keras.models    import Model
from keras.layers import add
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU
from keras.optimizers import Adam, Nadam

from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint,TerminateOnNaN
from keras.regularizers import l1,l2

from keras.utils import HDF5Matrix,Sequence

from keras import backend as K
from keras import losses

class K_gen(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))-1

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        batch_y[:,0]/=300.0
        batch_y[:,1]/=150.0
        return batch_x, batch_y


def conv(x,filters, kernel_size,strides=1):
    x = Conv3D(filters=filters,
               kernel_size=kernel_size,
               padding="same",
               kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = GaussianNoise(0.01)(x)
    x = MaxPooling3D(pool_size=strides)(x)
    x = Dropout(0.3)(x)
    return x

def dens(x,nodes):
    x = Dense(nodes,kernel_regularizer=l2(1e-5))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)
    return x


def identity_block(input_tensor, kernel_size, filters):

    filters1, filters2, filters3 = filters

    x = conv(x=input_tensor,filters=filters1, kernel_size=1)
    x = conv(x=x,filters=filters2, kernel_size=kernel_size)
    x = conv(x=x,filters=filters3, kernel_size=1)
    x = Add()([x, input_tensor])
    x = Activation("relu")(x)
    return x


def conv_block(input_tensor, kernel_size, filters, strides=2):

    filters1, filters2, filters3 = filters
    x = conv(x=input_tensor,filters=filters1, kernel_size=1,strides=strides)
    x = conv(x=x,filters=filters2, kernel_size=kernel_size)
    x = conv(x=x,filters=filters3, kernel_size=1)
    shortcut = conv(x=input_tensor,filters=filters3, kernel_size=1,strides=strides)
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


batch_size = 10
L = 60000
X_train = HDF5Matrix("dataset.h5", "X", start=0, end=L)
Y_train = HDF5Matrix("dataset.h5", "Y", start=0, end=L)

X_test  = HDF5Matrix("dataset.h5", "X", start=L, end=L+22730)
Y_test  = HDF5Matrix("dataset.h5", "Y", start=L, end=L+22730)

gen_train = K_gen(X_train,Y_train,batch_size=batch_size)
gen_test  = K_gen(X_test,Y_test,batch_size=batch_size)


x_inp  = Input(batch_shape=(batch_size,31,31,31,2))
x  = BatchNormalization()(x_inp)
x  = GaussianNoise(0.01)(x) 
x = conv_block    (x, 3, [32 , 32 , 128 ])
x = identity_block(x, 3, [32 , 32 , 128 ])
x = conv_block    (x, 3, [64, 64, 256 ])
x = identity_block(x, 3, [64, 64, 256 ])
x = identity_block(x, 3, [64, 64, 256 ])
x = conv_block    (x, 3, [128, 128, 512])
x = identity_block(x, 3, [128, 128, 512])

x = AveragePooling3D(3, name='final_pool')(x)
x = Flatten()(x)

x = dens(x,512)
x = dens(x ,128)
x = dens(x,64)
out = GaussianNoise(0.01)(Dense(2,activation="linear")(x))

mofnn = Model(x_inp, out)

optimizer = Adam (lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
mofnn.compile(optimizer=optimizer, loss="mse")
mofnn.summary()

from keras.utils import plot_model
plot_model(mofnn, to_file='model.png')


chkpntr = ModelCheckpoint(filepath="model.h5", save_best_only=True,verbose=1,save_weights_only=True)
lrrdcr  = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-7,verbose=1)
erlstpp = EarlyStopping(monitor='val_loss', patience=4)
trmtnan = TerminateOnNaN()
_CALLBACKS = [chkpntr,erlstpp,lrrdcr,trmtnan]

mofnn.fit_generator(gen_train,
                  epochs=500,
                  validation_data=gen_test,
                  verbose=1,
                  callbacks=_CALLBACKS)

# mofnn.load_weights("model.h5")

# DATA = pandas.DataFrame(columns=["Calculated CH4 ads. 1 bar","Predicted CH4 ads. 1 bar"])

# pred = mofnn.predict(X_test,batch_size=batch_size,verbose=1)
# DATA.loc[:,"Calculated CH4 ads. 1 bar"] = np.array(Y_test)[:,1]
# DATA.loc[:,"Predicted CH4 ads. 1 bar" ] = pred*150.0
# DATA.to_csv("predicted_1bar.csv")

