# script to fill a hdf5 file with two datasets:
# X is the structure factor arrays (n datapoints x 31,31,31,2)
# Y is any data you want to predict (n datapoints x dim)
# easily adaptable
import h5py
import numpy
import pandas

path = 'Fhkl-hypmofs.h5'

o_Y = pandas.read_csv("ch4_adsorption.csv",index_col="ID")
o_Y = o_Y.dropna()
o_Y = o_Y.drop_duplicates()
print(o_Y)
ii=0
with h5py.File("dataset.h5","w") as out5:
    print("....")
    X = out5.create_dataset("X",(1,31,31,31,2),maxshape=(None,31,31,31,2))
    Y = out5.create_dataset("Y",(1,2),maxshape=(None,2))
    print("....")
    with h5py.File(path,"r") as in5:
        print("....")
        o_X  = in5["X"]
        o_ID = in5["ID"]
        print("....")
        for o_idx,o_id in numpy.ndenumerate(o_ID):
            print(o_idx)
            o_id = str(o_id,encoding="utf8")
            if o_id in o_Y.index:
                print("\t|-->",o_id)
                x = o_X[o_idx[0],:,:,:,:]
                y = o_Y.loc[o_id].values
                X.resize((ii+1,31,31,31,2))
                Y.resize((ii+1,2))
                X[ii]=x
                Y[ii]=y
                ii+=1


