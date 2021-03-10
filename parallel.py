#!/software/python3/3.8.3/bin/python3
#SBATCH -p action 
#SBATCH -o my_output_%j
#SBATCH --mem-per-cpu=1GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
import sys
sys.path.append('/scratch/amandal4/PLDM-python/clean/PLDM-python')
import pldm
import model
from multiprocessing import Pool
import time 
import numpy as np

t0 = time.time()
#----------------
trajs = 40 
#----------------

#------------------------------------------------------------------------------------------
#--------------------------- SBATCH -------------------------------------------------------
sbatch = [i for i in open('parallel.py',"r").readlines() if i[:10].find("#SBATCH") != -1 ]
cpu = int(sbatch[-1].split("=")[-1].replace("\n","")) 
nodes = int(sbatch[-2].split()[-1].replace("\n",""))
print (f"nodes : {nodes} | cpu : {cpu}")
procs = cpu * nodes
print (f"Total trajectories {procs * trajs}")
ntraj = procs * trajs
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

t1 = time.time()
with Pool(cpu) as p:

    NSteps = model.parameters.NSteps
    NTraj = model.parameters.NTraj
    NStates = model.parameters.NStates

    #------ Arguments for each CPU------------------
    args = []
    for j in range(procs):
        par = model.parameters() 
        par.ID   = j
        par.SEED   = np.random.randint(0,100000000)
        args.append(par)
    #-----------------------------------------------
    print("Running : " + str(par.NTraj*cpu) + " trajectories in %s cpu"%(cpu) )

    #------------------- parallelization -----------------------------------
    rho_ensemble  = p.map(pldm.runTraj, args)
    #-----------------------------------------------------------------------

    #------------------- Gather --------------------------------------------
    rho_sum = np.zeros(rho_ensemble[0].shape, dtype = rho_ensemble[0].dtype)
    for i in range(cpu):
        for t in range(NSteps):
            rho_sum[:,:,t] += rho_ensemble[i][:,:,t]


PiiFile = open("Pii.txt","w") 
NTraj = model.parameters().NTraj
for t in range(NSteps):
    PiiFile.write(str(t) + "\t")
    for i in range(NStates):
        PiiFile.write(str(rho_sum[i,i,t].real / (  cpu * NTraj ) ) + "\t")
    PiiFile.write("\n")
PiiFile.close()
        
print(time.time()-t1)