import numpy as np
from numpy import array as A

def model(M=3):

    #        |  M0 |  M1 |  M2  |  M3 |  M4 |  M5 |
    ε   = A([  0.0,   0.0,  1.0,  1.0,  0.0,  5.0  ])
    ξ   = A([ 0.09,  0.09,  0.1,  0.1,  2.0,  4.0  ])
    β   = A([  0.1,   5.0, 0.25,  5.0,  1.0,  0.1  ])
    ωc  = A([  2.5,   2.5,  1.0,  2.5,  1.0,  2.0  ])
    Δ   = A([  1.0,   1.0,  1.0,  1.0,  1.0,  1.0  ])
    N   = A([  100,   100,  100,  100,  400,  400  ])

    return ε[M], ξ[M], β[M], ωc[M], Δ[M], N[M]  


def bathParam(ξ, ωc, ndof):
    ωm = 4.0
    ω0 = ωc * ( 1-np.exp(-ωm) ) / ndof
    c = np.zeros(( ndof ))
    ω = np.zeros(( ndof ))
    for d in range(ndof):
        ω[d] =  -ωc * np.log(1 - (d+1)*ω0/ωc)
        c[d] =  np.sqrt(ξ * ω0) * ω[d]  
    return c, ω


class parameters():
   NSteps = 200 #int(2*10**6)
   NTraj = 200
   dtN = 0.01
   dtE = dtN/20
   NStates = 2
   M = 1
   initState = 0
   nskip = 10
   ε, ξ, β, ωc, Δ, ndof = model(3) # model3
   c, ω  = bathParam(ξ, ωc, ndof)



def Hel(R):
    c = parameters.c
    Δ = parameters.Δ 
    ε = parameters.ε

    Vij = np.zeros((2,2))

    Vij[0,0] =   np.sum(c * R) + ε
    Vij[1,1] = - np.sum(c * R) - ε

    Vij[0,1], Vij[1,0] = Δ, Δ 
    return Vij


def dHel0(R):
    c = parameters.c
    ω = parameters.ω
    dH0 = ω**2 * R 
    return dH0


def dHel(R):
    c = parameters.c
    ω = parameters.ω
    
    dHij = np.zeros(( 2,2,len(R)))
    dHij[0,0,:] = c   
    dHij[1,1,:] = -c  
    return dHij

def initR():
    R0 = 0.0
    P0 = 0.0
    β  = parameters.β
    ω = parameters.ω
    ndof = parameters.ndof

    sigP = np.sqrt( ω / ( 2 * np.tanh( 0.5*β*ω ) ) )
    sigR = sigP/ω

    R = np.zeros(( ndof ))
    P = np.zeros(( ndof ))
    for d in range(ndof):
        R[d] = np.random.normal()*sigR[d]  
        P[d] = np.random.normal()*sigP[d]  
    return R, P