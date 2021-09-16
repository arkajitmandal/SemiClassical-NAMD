import numpy as np
import random
"""
This is a Focused-spinPLDM code 
Here sampled mean all combinations 
of forward-backward initialization 
Here focused mean only forward backward 
focused on initial state
"""


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# Initialization of the mapping Variables
def initMapping(NStates, F = 0, B = 0):
    """
    Returns np.array zF and zB (complex)
    Only Focused-PLDM implemented
    (Originally by Braden Weight)
    """

    gw = (2/NStates) * (np.sqrt(NStates + 1) - 1)

    # Initialize mapping radii
    rF = np.ones(( NStates )) * np.sqrt(gw)

    randStateF = F
    rF[randStateF] = np.sqrt( 2 + gw ) # Choose initial mapping state randomly

    rB = np.ones(( NStates )) * np.sqrt(gw)

    randStateB = B
    rB[randStateB] = np.sqrt( 2 + gw ) # Choose initial mapping state randomly

    zF = np.zeros(( NStates ),dtype=complex)
    zB = np.zeros(( NStates ),dtype=complex)

    for i in range(NStates):
        phiF = random.random() * 2 * np.pi # Azimuthal Angle -- Always Random
        zF[i] = rF[i] * ( np.cos( phiF ) + 1j * np.sin( phiF ) )
        phiB = random.random() * 2 * np.pi # Azimuthal Angle -- Always Random
        zB[i] = rB[i] * ( np.cos( phiB ) + 1j * np.sin( phiB ) )

    return zF,zB


def Umap(z, dt, VMat):
    """
    Updates mapping variables
    """
        
    Zreal = np.real(z) 
    Zimag = np.imag(z) 

    # Propagate Imaginary first by dt/2
    Zimag -= 0.5 * VMat @ Zreal * dt

    # Propagate Real by full dt
    Zreal += VMat @ Zimag * dt
    
    # Propagate Imaginary final by dt/2
    Zimag -= 0.5 * VMat @ Zreal * dt

    return  Zreal + 1j*Zimag


def Force(dat):
    R  = dat.R
    dH = dat.dHij  
    dH0 = dat.dH0
    gw  = dat.gw 
    NStates = dat.param.NStates

    zF = dat.zF
    zB = dat.zB

    η = 0.5 * np.real( ( np.outer( zF.conjugate(), zF ) + np.outer( zB.conjugate(), zB ) - 2 * gw * np.identity(NStates) ) )

    F = np.zeros((len(R)))
    F -= dH0
    for i in range(NStates):
        F -= 0.5 * dH[i,i,:] * η[i,i]
        for j in range(i+1,NStates): # Double counting off-diagonal to save time
            F -= 2 * 0.5 * dH[i,j,:] * η[i,j]
    return F

def VelVer(dat) :

    # data 
    zF, zB = dat.zF * 1.0, dat.zB *  1.0 
    par =  dat.param
    v = dat.P/par.M
    EStep = int(par.dtN/par.dtE)

    # half-step mapping
    for t in range(EStep):
        zF = Umap(zF, par.dtE/2, dat.Hij)
        zB = Umap(zB, par.dtE/2, dat.Hij)
    dat.zF, dat.zB = zF * 1, zB * 1 

    # ======= Nuclear Block ==================================
    F1    =  Force(dat) # force with {qF(t+dt/2)} * dH(R(t))
    dat.R += v * par.dtN + 0.5 * F1 * par.dtN ** 2 / par.M
    
    #------ Do QM ----------------
    dat.Hij  = par.Hel(dat.R)
    dat.dHij = par.dHel(dat.R)
    dat.dH0  = par.dHel0(dat.R)
    #-----------------------------
    F2 = Force(dat) # force with {qF(t+dt/2)} * dH(R(t+ dt))
    v += 0.5 * (F1 + F2) * par.dtN / par.M

    dat.P = v * par.M
    # =======================================================
    
    # half-step mapping
    dat.Hij = par.Hel(dat.R) # do QM


    for t in range(EStep):
        zF = Umap(zF, par.dtE/2, dat.Hij)
        zB = Umap(zB, par.dtE/2, dat.Hij)
    dat.zF, dat.zB = zF * 1, zB * 1 
    

    #---- Propagate Ugamma ----------
    E, U = np.linalg.eigh(dat.Hij)
    # Transform eigenvalues
    Udt = U @  np.diag( np.exp( -1j * E * par.dtN) ) @ U.T  
    dat.Ugam = Udt @ dat.Ugam 

    return dat



def pop(dat):
    NStates = dat.param.NStates
    iState = dat.param.initState
    rho = np.zeros((NStates,NStates),dtype=complex)
    γ = dat.gw * dat.Ugam

    rhoF = ( dat.zB[:].conjugate() * dat.zB0 - γ[:,iState].conjugate())
    rhoB = ( dat.zF[:] * dat.zF0.conjugate() - γ[:,iState] )

    return 0.25 * np.outer(rhoF, rhoB)

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
    stype = parameters.stype
    nskip = parameters.nskip
    #---------------------------
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1
    rho_ensemble = np.zeros((NStates,NStates,NSteps//nskip + pl), dtype=complex)


    #---------------------------
    # Ensemble
    for itraj in range(NTraj): 
      # Stype (Forward-Backward combinations)
      if stype == "sampled":
        FB = [[initState,i] for i in range(NStates)]

      if stype == "focused":
        FB = [[initState,initState]]
      #---------------------------------------
      for iFB in FB:
        # weight for this trajectory
        F, B = iFB[0], iFB[1]
        W = 1.0 + (F!=B) * 1.0 
 
        gw = (2/NStates) * (np.sqrt(NStates + 1) - 1)
        # Trajectory data
        dat = Bunch(param =  parameters, gw = gw)
        dat.Ugam = np.identity(NStates)

        # initialize R, P
        dat.R, dat.P = parameters.initR()

        # set propagator
        vv  = VelVer
 
        # Call function to initialize mapping variables
 
        # various 
        dat.zF, dat.zB  = initMapping(NStates, F, B) 

        # Set initial values of fictitious oscillator variables for future use
        dat.zF0, dat.zB0 =  dat.zF[initState], dat.zB[initState]

        #----- Initial QM --------
        dat.Hij  = parameters.Hel(dat.R)
        dat.dHij = parameters.dHel(dat.R)
        dat.dH0  = parameters.dHel0(dat.R)
        #----------------------------
        iskip = 0  
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                rho_ensemble[:,:,iskip] += pop(dat) * W
                iskip += 1
            #-------------------------------------------------------
            dat = vv(dat)

    return rho_ensemble

if __name__ == "__main__": 
    import spinBoson as model
    par =  model.parameters
    
    par.dHel = model.dHel
    par.dHel0 = model.dHel0
    par.initR = model.initR
    par.Hel   = model.Hel

    rho_ensemble = runTraj(par)
    
    NSteps = model.parameters.NSteps
    NTraj = model.parameters.NTraj
    NStates = model.parameters.NStates

    PiiFile = open("Pii.txt","w") 
    for t in range(NSteps):
        PiiFile.write(f"{t * model.parameters.nskip} \t")
        for i in range(NStates):
            PiiFile.write(str(rho_ensemble[i,i,t].real / NTraj) + "\t")
        PiiFile.write("\n")
    PiiFile.close()

