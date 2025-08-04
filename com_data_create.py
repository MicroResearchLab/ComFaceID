from toolbox.MGF import MGF, write
import os

from itertools import combinations
import numpy as np
import time
from toolbox.Spectrum import Spectrum

com_tor = 18

def com_data_generate(specs:dict):
    in_mz = specs['mz']
    in_ins = specs["intensities"]
    sp = []
    mz = []
    ins = []
    for index in range(len(in_mz)):
        sp.append([in_mz[index], in_ins[index]])
        mz.append(in_mz[index])
        ins.append(in_ins[index])
    # if len(sp) <= 4:
    #     continue
    mass = float(specs['metadata']["pepmass"][0])
    for peak_com in sp:
        t_mass = abs(mass - peak_com[0])
        t_value = peak_com[1]
        if(t_mass < com_tor):
            continue
        t_mass = round(t_mass*100, 0)/100
        mz.append(t_mass)
        ins.append(t_value)
    mz = np.array(mz)
    ins = np.array(ins) 
    idx_sorted = np.argsort(mz)
    mz = mz[idx_sorted]
    ins = ins[idx_sorted]
    res = {"params":specs['metadata'],
                    "m/z array": mz, "intensity array": ins}
    return res,Spectrum(mz=mz,
                       intensities=ins,
                       metadata=specs['metadata'],
                       metadata_harmonization=False)


