from numpy import random as rn
import numpy as np

class parameters():
    au = 27.2113961 # 27.2114
    ps = 41341.374575751
    fs = 41.34136 # 41.341374575751 fs to au!
    cm = 1/4.556335e-6 # 219474.63
    kB = 3.166829e-6 
    NSteps = 16000 # number of nuclear steps in the simulation
    NStepsPrint = 500 # number of nuclear steps that are stored/printed/outputed
    nskip = int(NSteps/NStepsPrint) # used to enforce NStepsPrint
    totalTime = 4000*fs # total amount of simulation time
    dtN = totalTime/NSteps # nuclear timestep 
    dtE = dtN /10# /1 # electronic timestep (should be smaller than nuclear)
    NTraj = 20
    NStates = 7
    NBath = 60
    NR = NBath*NStates
    ndof = NR
    M = np.ones(NR)
    initState = 0
    # MASH-specific parameters
    λ = 35/cm
    τc = 50*fs
    ωc = 106.14/cm # 1/τc
    ωk = ωc * np.tan(0.5*np.pi*(np.arange(NBath)+0.5)/NBath)
    #ωk = ωc * np.tan( np.pi * (1 - np.arange(1,NBath+1)/(NBath + 1)) / 2)
    ck = ωk * np.sqrt(2*λ/NBath)
    #ck = ωk * np.sqrt(2*λ/(NBath+1))
    β = 1 / (300 * kB) # 1053


def Hel(R):
    H = np.array([[200,     -87.7,  5.5,    -5.9,   6.7,    -13.7,  -9.9    ],
                  [-87.7,   320,    30.8,   8.2,    0.7,    11.8,   4.3     ],
                  [5.5,     30.8,   0,      -53.5,  -2.2,   -9.6,   6.0     ],
                  [-5.9,    8.2,    -53.5,  110,    -70.7,  -17.0,  -63.3   ],
                  [6.7,     0.7,    -2.2,   -70.7,  270,    81.1,   -1.3    ],
                  [-13.7,   11.8,   -9.6,   -17.0,  81.1,   420,    39.7    ],
                  [-9.9,    4.3,    6.0,    -63.3,  -1.3,   39.7,   230     ]])/parameters.cm
    for j in range(parameters.NStates):
        H[j, j] += np.sum(parameters.ck * R[j*parameters.NBath:(j+1)*parameters.NBath])
    return H


def dHel(R):
    dH = np.zeros((parameters.NStates,parameters.NStates,parameters.NR))
    for j in range(parameters.NStates):
        dH[j,j,j*parameters.NBath:(j+1)*parameters.NBath] = parameters.ck
    return dH 

 
def dHel0(R):
    dH0 = np.zeros(parameters.NR) # state independent, so only need to do NR derivatives once
    for j in range(parameters.NStates):
        dH0[j * parameters.NBath : (j + 1) * parameters.NBath] = parameters.ωk**2 * R[j * parameters.NBath : (j + 1) * parameters.NBath]
    return dH0 
 
def initR():
    R0 = 0.0 # average initial nuclear position (could be array specific to each nuclear DOF)
    P0 = 0.0 # average initial nuclear momentum (could be array specific to each nuclear DOF)
    #σP = np.sqrt(ωk/(2.0 * np.tanh(0.5 * β * ωk))) # standard dev of nuclear momentum (mass = 1)
    #σR = σP/(ωk) # standard dev of nuclear position (mass = 1)
    σP = np.sqrt(1/parameters.β)*np.ones(parameters.NBath)
    σR = 1/np.sqrt(parameters.β*parameters.ωk**2)
    R = np.zeros((parameters.NR))
    P = np.zeros((parameters.NR))
    for Ri in range(parameters.NStates*parameters.NBath):
        imode = (Ri%parameters.NBath) # current bath mode
        R[Ri] = rn.normal() * σR[imode] + R0 # gaussian random variable centered around R0
        P[Ri] = rn.normal() * σP[imode] + P0 # gaussian random variable centered around P0
    return R, P
