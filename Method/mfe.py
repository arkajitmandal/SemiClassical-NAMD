import numpy as np
# jit
import time
# from numba import jit, objmode

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# Initialization of the electronic part
def initElectronic(Nstates, initState = 0):
    #global qF, qB, pF, pB, qF0, qB0, pF0, pB0
    c = np.zeros((Nstates), dtype='complex128')
    c[initState] = 1.0
    return c

#@jit(nopython=False)
def propagateCi(ci,Vij, dt):
    c = ci * 1.0
    # https://thomasbronzwaer.wordpress.com/2016/10/15/numerical-quantum-mechanics-the-time-dependent-schrodinger-equation-ii/
    ck1 = (-1j) * (Vij @ c)
    ck2 = (-1j) * Vij @ (c + (dt/2.0) * ck1 )
    ck3 = (-1j) * Vij @ (c + (dt/2.0) * ck2 )
    ck4 = (-1j) * Vij @ (c + (dt) * ck3 )
    c = c + (dt/6.0) * (ck1 + 2.0 * ck2 + 2.0 * ck3 + ck4)
    return c

#@jit(nopython=False)
def Force(dHij, dH0, ci):

    # dH = dat.dHij #dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    # dH0  = dat.dH0 
    # ci = dat.ci

    F = -dH0 #np.zeros((len(dat.R)))
    #F -= np.real(np.einsum('ijk,i,j->k', dHij, ci.conjugate(), ci))

    for i in range(len(ci)):
        F -= dHij[i,i,:]  * (ci[i] * ci[i].conjugate() ).real
        for j in range(i+1, len(ci)):
            F -= 2.0 * dHij[i,j,:]  * (ci[i].conjugate() * ci[j] ).real

    return F

def VelVer(dat) : 
    par =  dat.param
    v = dat.P/par.M
    F1 = dat.F1 
    # electronic wavefunction
    ci = dat.ci * 1.0
    
    EStep = int(par.dtN/par.dtE)
    dtE = par.dtN/EStep

    # half electronic evolution
    for t in range(int(np.floor(EStep/2))):
        ci = propagateCi(ci, dat.Hij, dtE)  
    ci /= np.sum(ci.conjugate()*ci) 
    dat.ci = ci * 1.0 

    # ======= Nuclear Block ==================================
    dat.R += v * par.dtN + 0.5 * F1 * par.dtN ** 2 / par.M
    
    #------ Do QM ----------------
    dat.Hij  = par.Hel(dat.R) + 0j
    dat.dHij = par.dHel(dat.R)
    dat.dH0  = par.dHel0(dat.R)
    #-----------------------------
    F2 = Force(dat.dHij, dat.dH0, dat.ci) # force at t2
    v += 0.5 * (F1 + F2) * par.dtN / par.M
    dat.F1 = F2
    dat.P = v * par.M
    # ======================================================
    # half electronic evolution
    for t in range(int(np.ceil(EStep/2))):
        ci = propagateCi(ci, dat.Hij, dtE)  
    ci /= np.sum(ci.conjugate()*ci)  
    dat.ci = ci * 1.0 

    return dat


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
        vv  = VelVer

        # Call function to initialize mapping variables
        dat.ci = initElectronic(NStates, initState) # np.array([0,1])

        #----- Initial QM --------
        dat.Hij  = parameters.Hel(dat.R) + 0j
        dat.dHij = parameters.dHel(dat.R)
        dat.dH0  = parameters.dHel0(dat.R)
        dat.F1 = Force(dat.dHij, dat.dH0, dat.ci) # Initial Force
        #----------------------------
        iskip = 0 # please modify
        t0 = time.time()
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                rho_ensemble[:,:,iskip] += pop(dat)
                iskip += 1
            #-------------------------------------------------------
            dat = vv(dat)
        time_taken = time.time()-t0
        print(f"Time taken: {time_taken} seconds")


    return rho_ensemble


