import os,sys 
sys.path.append(os.popen("pwd").read().replace("\n","")+"/Model")
def getInput(input,key):
    try:
        txt = [i for i in input if i.find(key)!=-1][0].split("=")[1].split("#", 1)[0].replace("\n","")
    except:
        txt = ""
    return txt.replace(" ","")


def sbatch(filename, option="", pre = ""):
    """
    Submit a job and get the job id returned
    """
    import os
    job    = f"sbatch {pre} {filename} {option}"
    submit = os.popen(job).read()
    subId = submit.split()[3].replace("\n","")
    return subId


try:
    inputfile =  sys.argv[1]
    input = open(inputfile, 'r').readlines()

except:
    inputfile =  "input.txt"
    input = open(inputfile, 'r').readlines()

print(f"Reading {inputfile}")



# System
system = getInput(input,"System")
#print (system)
# SLURM
if system == "slurm":
    print ("Running jobs in a HPCC")
    model = getInput(input,"Model")
    exec(f"from {model} import parameters")
    ntraj = parameters.NTraj 

    
    nodes     = int(getInput(input,"Nodes"))
    ncpus     = int(getInput(input,"Cpus"))
    totalTraj = ntraj * nodes * ncpus
    print(f"Using {nodes} Nodes each with {ncpus} CPUs")
    print("-"*50)
    print(f"Total Number of Trajectories = {totalTraj}")
    print(f"Trajectories per CPU         = {ntraj}")

    print("-"*50)
    with open("output.log", "w+") as output:
        output.write(f"Total Number of Trajectories = {totalTraj}\n")
        output.write(f"Trajectories per CPU         = {ntraj}\n")
        output.write(f"Using {nodes} Nodes each with {ncpus} CPUs\n")
        output.write("-"*50 + "\n")

    os.system(f"rm -rf RUN")
    os.mkdir("RUN")
    ids = []
    for i in range(nodes):
        # Run the jobs
        os.mkdir(f"RUN/run-{i}")

        partition = getInput(input,"Partition")
        
        method   = getInput(input,"Method")
        options = f"--partition {partition} \
                    --ntasks-per-node {ncpus}\
                    --job-name={model}-{method}\
                    --open-mode=append"

        ids.append(sbatch("parallel.py", f"{inputfile} RUN/run-{i}", options)) 
        print (f"Submitted Job {ids[-1]}")
    

    # Gather and average
    jobs = ":".join(ids)
    pre = f"--dependency=afterok:{jobs} --partition {partition} --output=output.log --open-mode=append"
    sbatch("avg.py", inputfile, pre)

elif system == "htcondor":
    print ("Running jobs in a HTC")
    # read input-------------------
    cpus   = getInput(input,"Cpus")
    model  = getInput(input,"Model")
    method = getInput(input,"Method")
    # read input-------------------
    exec(f"from {model} import parameters")

    ntraj = parameters.NTraj
    print(f"Total Number of Trajectories = {int(float(ntraj) * float(cpus))}")
    os.system(f"rm -rf RUN")
    os.mkdir("RUN")
    os.chdir("RUN") 
    for ic in range(int(cpus)):
        os.mkdir(f"run-{ic}")
        os.mkdir(f"run-{ic}/Model")
        os.mkdir(f"run-{ic}/Method")
        os.mkdir(f"run-{ic}/log")
        
        os.system(f"cp ../Model/{model}.py run-{ic}/Model/")
        os.system(f"cp ../Method/{method}.py run-{ic}/Method/")

        os.system(f"cp ../condor.sh run-{ic}")
        os.system(f"cp ../serial.py run-{ic}")
        os.system(f"cp ../condor.sub run-{ic}")
        os.system(f"cp ../input.txt run-{ic}")

        os.chdir(f"run-{ic}") 
        os.system("condor_submit condor.sub") 
        os.chdir(f"../") 
# PC
else:
    print ("Running jobs in your local machine (like a PC)")
    # Some messages 
    ignoreList = []
    try :
        getInput(input,"Nodes")
        ignoreList.append("Nodes")
        
    except:
        pass
    try :
        getInput(input,"Cpus")
        ignoreList.append("Cpus")
    except:
        pass
    try :
        getInput(input,"Partition")
        ignoreList.append("Partition")
    except:
        pass
    print(f"Ignoring {ignoreList} in {inputfile}")
    model = getInput(input,"Model")
    exec(f"from {model} import parameters")
    ntraj = parameters.NTraj 
    print("-"*50)
    print(f"Total Number of Trajectories = {ntraj}")
    print("-"*50)
    os.system(f"python3 serial.py {inputfile}")



