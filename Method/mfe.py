import numpy as np


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# Initialization of the electronic part
def initElectronic(Nstates, initState = 0):
    c = np.zeros((Nstates), dtype='complex128')
    c[initState] = 1.0
    return c

def Uqm(dat, dt):
    c = dat.ci * 1.0
    # https://thomasbronzwaer.wordpress.com/2016/10/15/numerical-quantum-mechanics-the-time-dependent-schrodinger-equation-ii/
    ck1 = (-1j) * (dat.Hij @ c)
    ck2 = (-1j) * (dat.Hij @ c + (dt/2.0) * ck1 )
    ck3 = (-1j) * (dat.Hij @ c + (dt/2.0) * ck2 )
    ck4 = (-1j) * (dat.Hij @ c + (dt) * ck3 )
    c = c + (dt/6.0) * (ck1 + 2.0 * ck2 + 2.0 * ck3 + ck4)
    c /= np.sum(c.conjugate()*c) # renormalization
    dat.ci = c
    return dat

def Force(dat):

    dH = dat.dHij #dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    dH0  = dat.dH0 

    ci = dat.ci

    F = -dH0 #np.zeros((len(dat.R)))
    for i in range(len(ci)):
        F -= dH[i,i,:]  * (ci[i] * ci[i].conjugate() ).real
        for j in range(i+1, len(ci)):
            F -= 2.0 * dH[i,j,:]  * (ci[i].conjugate() * ci[j] ).real
    return F


def pop(dat):
    ci =  dat.ci
    return np.outer(ci.conjugate(),ci)

def runTraj(parameters):
    #------- Seed --------------------
    try:
        np.random.seed(parameters.SEED)
    except:
        pass
    #------------------------------------
    ## Parameters -------------
    NSteps = parameters.NSteps
    NTraj = parameters.NTraj
    NStates = parameters.NStates
    initState = parameters.initState # intial state
    nskip = parameters.nskip
    #---------------------------
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1
    rho_ensemble = np.zeros((NStates,NStates,NSteps//nskip + pl), dtype=complex)
    # Ensemble
    for itraj in range(NTraj): 
        # Trajectory data
        dat = Bunch(param =  parameters )
        dat.R, dat.P = parameters.initR()
        
        # set propagator
        dat.force = Force
        dat.Uqm = Uqm
        vv  = parameters.vv

        # Call function to initialize mapping variables
        dat.ci = initElectronic(NStates, initState) # np.array([0,1])

        #----- Initial QM --------
        dat.Hij  = parameters.Hel(dat.R)
        dat.dHij = parameters.dHel(dat.R)
        dat.dH0  = parameters.dHel0(dat.R)
        dat.F1 = Force(dat) # Initial Force
        #----------------------------
        iskip = 0 # please modify
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                rho_ensemble[:,:,iskip] += pop(dat)
                iskip += 1
            #-------------------------------------------------------
            dat = vv(dat)

    return rho_ensemble