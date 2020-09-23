#!/software/python3/3.8.3/bin/python3
#SBATCH -p action 
#SBATCH -o my_output_%j
#SBATCH --mem-per-cpu=1GB
#SBATCH -t 1:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=24
import sys
sys.path.append('/scratch/amandal4/PLDM+R/PLDM-python')
import pldm
import model
from multiprocessing import Pool
import time 
import numpy as np
cpu = 24 * 2

t1 = time.time()
with Pool(cpu) as p:

    NSteps = model.parameters.NSteps
    NTraj = model.parameters.NTraj
    NStates = model.parameters.NStates

    params = model.parameters
    print("Running : " + str(params.NTraj*cpu) + " trajectories in %s cpu"%(cpu) )
    rho_ensemble  = p.map(pldm.runTraj, [params for i in range(cpu)])
    rho_sum = np.zeros(rho_ensemble[0].shape, dtype = rho_ensemble[0].dtype)
    for i in range(cpu):
        for t in range(NSteps):
            rho_sum[:,:,t] += rho_ensemble[i][:,:,t]


PiiFile = open("Pii.txt","w") 
for t in range(NSteps):
    PiiFile.write(str(t) + "\t")
    for i in range(NStates):
        PiiFile.write(str(rho_sum[i,i,t].real / (  cpu * NTraj) ) + "\t")
    PiiFile.write("\n")
PiiFile.close()
        
print(time.time()-t1)