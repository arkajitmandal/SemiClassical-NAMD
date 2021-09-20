import numpy as np
from numpy import exp
from numpy import array as A
from numpy import diag_indices as Dii

class parameters():
   NSteps = 3000  
   NTraj = 5
   dtN = 1
   dtE = dtN/100
   NStates = 3
   M = 20000
   initState = 0
   nskip = 10
   ndof  = 1

def Hel(R):
    D =  A([0.020,  0.020,  0.003])
    b =  A([0.400,  0.650,  0.650])
    Re = A([4.000,  4.500,  6.000])
    c =  A([0.020,  0.000,  0.020])


    
    Aij =  A([0.005,  0.005])
    Rij =  A([3.400,  4.970])
    a   =  A([32.00,  32.00])

    Vij = np.zeros((3,3))
    # Diagonal
    Vij[Dii(3)] = D * ( 1 - exp( -b * (R - Re) ) )**2 + c

    # Off-Diagonal
    Vij[0,1] = Aij[0] * exp( -a[0] * (R - Rij[0])**2 )
    Vij[0,2] = Aij[1] * exp( -a[1] * (R - Rij[1])**2 )

    # Symmetric Vij
    Vij[2,0], Vij[1,0] = Vij[0,2], Vij[0,1]

    return Vij
    
def dHel0(R):
    return np.zeros((len(R)))

def dHel(R):
    D =  A([0.020,  0.020,  0.003])
    b =  A([0.400,  0.650,  0.650])
    Re = A([4.000,  4.500,  6.000])
    
    Aij =  A([0.005,  0.005])
    Rij =  A([3.400,  4.970])
    a   =  A([32.00,  32.00])

    dVij = np.zeros((3,3,1))
    dVij[0,0,0] = 2 * D[0] * b[0] * ( 1 - exp( -b[0] * (R- Re[0]) ) ) * exp( -b[0] * (R- Re[0]) )
    dVij[1,1,0] = 2 * D[1] * b[1] * ( 1 - exp( -b[1] * (R- Re[1]) ) ) * exp( -b[1] * (R- Re[1]) )
    dVij[2,2,0] = 2 * D[2] * b[2] * ( 1 - exp( -b[2] * (R- Re[2]) ) ) * exp( -b[2] * (R- Re[2]) )

    dVij[0,1,0] = -2 * a[0] * Aij[0] * exp( -a[0] * (R - Rij[0])**2 ) * ( R - Rij[0] )
    dVij[0,2,0] = -2 * a[1] * Aij[1] * exp( -a[1] * (R - Rij[1])**2 ) * ( R - Rij[1] )

    dVij[1,0,0] = dVij[0,1,0]
    dVij[2,0,0] = dVij[0,2,0]

    return dVij

def initR():
    R0 = 2.1
    P0 = 0.0
    M = parameters.M
    ω = 5*10**(-3.0)
    
    sigP = np.sqrt( M * ω/2.0 )
    sigR = np.sqrt( 1/(2.0* M * ω) )

    R = np.random.normal()*sigR + R0  
    P = np.random.normal()*sigP + P0

    return np.array([R]), np.array([P])

#------ only required for NRPMD----------
def initHel0(R):
    M = parameters.M
    ω = 5*10**(-3.0)
    R0 = 2.1
    
    return  np.sum(0.5 *M* ω**2 * (R-R0)**2.0)