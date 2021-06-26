import os,sys 
fold = int(sys.argv[1]) 
os.system(f"rm -rf RUN")
for i in range(fold):
    try:
        os.mkdir(f"RUN/run-{i}")
    except:
        pass
    os.system(f"sbatch parallel.py RUN/run-{i}")
    