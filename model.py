#def param():

import numpy as np

class parameters():
   dt = 0.01
   NSteps = 10000

def Hel():
    vij = np.zeros((2,2))
    vij[0,0] = 0.1
    vij[1,0] = 0.01
    vij[0,1] = 0.01
    vij[1,1] = 0.0
    return vij

#def dHel():
