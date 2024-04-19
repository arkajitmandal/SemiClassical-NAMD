#!/sw/eb/sw/Anaconda3/2024.02-1/bin/python

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=NAMD         #Set the job name to "JobExample1"
#SBATCH --time=01:30:00         #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1               #Request 1 task
#SBATCH --mem=2560M                   
#SBATCH --output=output/out.%j            


import sys, os

import numpy as np

sys.path.append(os.popen("pwd").read().replace("\n","")+"/Method")
sys.path.append(os.popen("pwd").read().replace("\n","")+"/Model")


#-------------------------
try:
    inputtxt = open(sys.argv[1], 'r').readlines()
    print(f"Reading {sys.argv[1]}")
except:
    print("Reading input.txt")
    inputtxt = open('input.txt', 'r').readlines()


def getInput(input,key):
    try:
        txt = [i for i in input if i.find(key)!=-1][0].split("=")[1].split("#", 1)[0].replace("\n","")
    except:
        txt = ""
    return txt.replace(" ","")

model_ =  getInput(inputtxt,"Model")
method_ = getInput(inputtxt,"Method").split("-")
exec(f"import {model_} as model")
exec(f"import {method_[0]} as method")
try:
    stype = method_[1]
except:
    stype = "_"
#-------------------------
import time 
import numpy as np

t0 = time.time()
#----------------
trajs = model.parameters.NTraj
#----------------
try:
    fold = sys.argv[2]
    os.system(f"rm -rf {fold}")
    os.system(f"mkdir -p {fold}")
    
except:
    os.system(f"rm -rf {fold}")
    os.system("mkdir -p output")
    fold = "./output"


ID = ''
try :
    ID = sys.argv[3]
    ID = "-" + ID
except:
    pass

#------------------------------------------------------------------------------------------
procs = 1
ntraj = trajs
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

t1 = time.time()


NSteps = model.parameters.NSteps
NTraj = model.parameters.NTraj
NStates = model.parameters.NStates

#------ Arguments------------------
par = model.parameters() 
par.ID     = np.random.randint(0,100)
par.SEED   = np.random.randint(0,100000000)
    
#---- methods in model ------
par.dHel = model.dHel
par.dHel0 = model.dHel0
par.initR = model.initR
par.Hel   = model.Hel
par.stype = stype

if method_[0]=="nrpmd":
    par.initHel0 = model.initHel0
    

#---- overriden parameters ------

parameters = [i for i in inputtxt if i.split("#")[0].split("=")[0].find("$") !=- 1]
for p in parameters:
    exec(f"par.{p.split('=')[0].split('$')[1]} = {p.split('=')[1].split('#')[0]}")
    print(f"Overriding parameters: {p.split('=')[0].split('$')[1]} = {p.split('=')[1].split('#')[0]}")
#--------------------------------

    
#------------------- run --------------- 
rho_sum  = method.runTraj(par)
#--------------------------------------- 


try:    
    PiiFile = open(f"{fold}/{method_[0]}-{method_[1]}-{model_}{ID}.txt","w") 
except:
    PiiFile = open(f"{fold}/{method_[0]}-{model_}{ID}.txt","w") 

NTraj = par.NTraj

if (method_[0] == 'sqc'):
    for t in range(rho_sum.shape[-1]):
        PiiFile.write(f"{t * model.parameters.nskip * model.parameters.dtN} \t")
        norm = 0
        for i in range(NStates):
            norm += rho_sum[i,i,t].real
        for i in range(NStates):
            PiiFile.write(str(rho_sum[i,i,t].real / ( norm ) ) + "\t")
        PiiFile.write("\n")
    PiiFile.close()
    
else:
    for t in range(rho_sum.shape[-1]):
        PiiFile.write(f"{t * model.parameters.nskip * model.parameters.dtN} \t")
        for i in range(NStates):
            PiiFile.write(str(rho_sum[i,i,t].real / ( NTraj ) ) + "\t")
        PiiFile.write("\n")
    PiiFile.close()


t2 = time.time()-t1
print(f"Total Time: {t2}")
print(f"Time per trajectory: {t2/ntraj}")

