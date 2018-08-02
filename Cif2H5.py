from Fhkl   import structurefactors
from ase.io import read
import numpy as np
import warnings
import h5py
import os

warnings.filterwarnings('error')

with h5py.File('hypmofs.h5', 'w') as h5f:
    ID   = h5f.create_dataset("ID"  , (1,)          , maxshape=(None,),dtype="S100")
    X    = h5f.create_dataset("X"   , (1,31,31,31,2), maxshape=(None,31,31,31,2))
    idx  = 0
    for _file in [f for f in os.listdir(os.getcwd()) if f.endswith(".cif")]:
        _id = ".".join(_file.split(".")[:-1])
        try:
            mof = read(_file)
        except Exception:
            continue
        ID.resize((idx+1,))
        ID[idx] = _id
        xyz   = mof.get_scaled_positions(wrap=True)
        numbers=mof.get_atomic_numbers()
        f_hkl = np.array(structurefactors(xyz,numbers))
        X.resize((idx+1,31,31,31,2))
        X[idx,:,:,:,:] = f_hkl
        print ("{0} --> Number {1}".format(_id, idx+1))
        idx += 1
