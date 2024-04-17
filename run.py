import os,sys 
sys.path.append(os.popen("pwd").read().replace("\n","")+"/Model")
def getInput(input,key):
    try:
        txt = [i for i in input if i.find(key)!=-1][0].split("=")[1].split("#", 1)[0].replace("\n","")
    except:
        txt = ""
    return txt.replace(" ","")


try:
    inputfile =  sys.argv[1]
    input = open(inputfile, 'r').readlines()
except:
    inputfile =  "input.txt"
    input = open(inputfile, 'r').readlines()

print(f"Reading {inputfile}")

fold = 'output'
try :
    fold = sys.argv[2]

except: 
    pass
os.system(f"rm -rf {fold}")
os.system(f"mkdir -p {fold}")
# System
system = getInput(input,"System")
#print (system)
# SLURM
if system == "slurm" or system == "htcondor":
    print (f"Running jobs in a {system}")
    model = getInput(input,"Model")
    exec(f"from {model} import parameters")
    ntraj = parameters.NTraj 


    ncpus     = int(getInput(input,"Cpus"))
    totalTraj = ntraj * ncpus
    print(f"Using {ncpus} CPUs")
    print("-"*50)
    print(f"Total Number of Trajectories = {totalTraj}")
    print(f"Trajectories per CPU         = {ntraj}")

    print("-"*50)
    
    if system == "slurm":
        for i in range(ncpus):
            os.system(f"sbatch serial.py {inputfile} {fold} {i}")
    if system == "htcondor":
        os.system(f"condor_submit condor.sub input={inputfile} output={fold} -queue {ncpus}")

# PC
else:
    print ("Running jobs in your local machine (like a PC)")
    # Some messages 
    model = getInput(input,"Model")
    exec(f"from {model} import parameters")
    ntraj = parameters.NTraj 
    print("-"*50)
    print(f"Total Number of Trajectories = {ntraj}")
    print("-"*50)
    os.system(f"python3 serial.py {inputfile} {fold}")



