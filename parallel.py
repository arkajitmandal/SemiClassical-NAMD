#!/software/anaconda3/2020.07/bin/python
#SBATCH -p action 
#SBATCH -o output.log
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
trajs = model.parameters.NTraj
#----------------

#------------------------------------------------------------------------------------------
#--------------------------- SBATCH -------------------------------------------------------
sbatch = [i for i in open('parallel.py',"r").readlines() if i[:10].find("#SBATCH") != -1 ]
cpu = int(sbatch[-1].split("=")[-1].replace("\n","")) 
nodes = int(sbatch[-2].split()[-1].replace("\n",""))
print (f"nodes : {nodes} | cpu : {cpu}")
procs = cpu * nodes
ntraj = procs * trajs
print (f"Total trajectories {ntraj}")
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

t1 = time.time()
with Pool(procs) as p:

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
    print(f"Running : {par.NTraj*procs}  trajectories in {procs} cpu" )

    #------------------- parallelization -----------------------------------
    rho_ensemble  = p.map(pldm.runTraj, args)
    #-----------------------------------------------------------------------

#------------------- Gather --------------------------------------------
rho_sum = np.zeros(rho_ensemble[0].shape, dtype = rho_ensemble[0].dtype)
for i in range(procs):
    for t in range(rho_ensemble[0].shape[-1]):
        rho_sum[:,:,t] += rho_ensemble[i][:,:,t]


PiiFile = open("Pii.txt","w") 
NTraj = model.parameters().NTraj
for t in range(rho_ensemble[0].shape[-1]):
    PiiFile.write(f"{t * model.parameters.nskip} \t")
    for i in range(NStates):
        PiiFile.write(str(rho_sum[i,i,t].real / (  procs * NTraj ) ) + "\t")
    PiiFile.write("\n")
PiiFile.close()
t2 = time.time()-t1
print(f"Total Time: {t2}")
print(f"Time per trajectory: {t2/ntraj}")