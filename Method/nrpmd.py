import numpy as np
import sys, os

import random

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


#===========initializing mapping variables==========
def initMap(param):
    """
    initialize Mapping variables q and p
    dimensionality q[nstate,nb] so do p
    """
    q = np.zeros((param.nstate,param.nb))
    p = np.zeros((param.nstate,param.nb))
    i0 = param.initState
    for i in range(param.nstate):
       for ib in range(param.nb):
            η = np.sqrt(1 + 2*(i==i0))
            θ = random() * 2 * np.pi
            q[i,ib] = η * np.cos(θ) 
            p[i,ib] = η * np.sin(θ) 
    return p,q
#============================================================