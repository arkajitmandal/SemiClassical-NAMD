import os,sys 
fold = int(sys.argv[1]) 
os.system(f"rm -rf RUN")
os.mkdir("RUN")
for i in range(fold):

    os.mkdir(f"RUN/run-{i}")
    os.system(f"sbatch parallel.py RUN/run-{i}")
    