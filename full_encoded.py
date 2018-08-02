# script to predict a property using the CVAE encoded inputs
import numpy as np
import pandas
import random
from keras.layers    import Input, Dense, GaussianNoise, Dropout, BatchNormalization, PReLU
from keras.models    import Model
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Adadelta

from keras.utils import HDF5Matrix, Sequence

class Gen(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
    def __len__(self):
        return (len(self.x) // self.batch_size)-1
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)[:,1]/100.0
        return batch_x,batch_y

def MOFNN(batch_size):
    kwargs = {"activation":"relu",
              "kernel_regularizer":l2(1e-3),
              "kernel_initializer":'glorot_uniform', 
              "bias_initializer":'glorot_uniform'}
    x_inp  = Input(batch_shape=(batch_size,2048))
    x = BatchNormalization()(x_inp)
    x = GaussianNoise(0.05)(x)
    x = Dropout(0.2)(x)
    x = Dense(2048,**kwargs)(x)
    x = BatchNormalization()(x)
    x = GaussianNoise(1e-4)(x)
    x = Dropout(0.2)(x)
    x = Dense(1024,**kwargs)(x)
    x = BatchNormalization()(x)
    x = GaussianNoise(1e-4)(x)
    x = Dropout(0.2)(x)
    x = Dense(1024,**kwargs)(x)
    x = BatchNormalization()(x)
    x = GaussianNoise(1e-4)(x)
    x = Dropout(0.2)(x)
    x = Dense(512,**kwargs)(x)
    x = Dense(1)(x)
    x = GaussianNoise(0.05)(x)
    model = Model(x_inp,x)
    return model

batch_size = 100
optimizer = Adadelta(lr=1.0e-2,clipvalue=.1)
mofnn = MOFNN(batch_size=batch_size)
mofnn.compile(optimizer=optimizer,loss="mse",metrics=["mean_absolute_percentage_error"])
mofnn.summary()

X_train = HDF5Matrix("dataset-encoded.h5","X",  end=60000)
Y_train = HDF5Matrix("dataset-encoded.h5","Y",  end=60000)
X_valid = HDF5Matrix("dataset-encoded.h5","X",start=60000)
Y_valid = HDF5Matrix("dataset-encoded.h5","Y",start=60000)
gen_train = Gen(X_train,Y_train,batch_size=batch_size)
gen_test  = Gen(X_valid,Y_valid,batch_size=batch_size)

chkpntr = ModelCheckpoint(filepath="1_bar_encoded.h5", save_best_only=True,verbose=1,save_weights_only=True)
lrrdcr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7,verbose=1)
erlstpp = EarlyStopping(monitor='val_loss', patience=10)
cb = [chkpntr,erlstpp,lrrdcr]


mofnn.fit_generator(gen_train,
                  epochs=500,
                  steps_per_epoch=len(gen_train),
                  validation_data=gen_test,
                  validation_steps=len(gen_test),
                  verbose=1,
                  callbacks=cb)
# mofnn.load_weights("1_bar_encoded.h5")


# X = HDF5Matrix("dataset-encoded.h5","X",end=82700)
# Y = HDF5Matrix("dataset-encoded.h5","Y",end=82700)
# DATA = pandas.DataFrame(columns=["Calculated CH4 ads. 1 bar","Predicted CH4 ads. 1 bar"])
# pred = mofnn.predict(X,batch_size=batch_size,verbose=1)
# DATA.loc[:,"Calculated CH4 ads. 1 bar"] = np.array(Y)[:,1]
# DATA.loc[:,"Predicted CH4 ads. 1 bar" ] = pred*100.0
# DATA.to_csv("predicted_1bar_encoded.csv")