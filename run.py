import os,sys 
fold = int(sys.argv[1]) 
os.system(f"rm -rf run-*")
for i in range(fold):
    try:
        os.mkdir(f"run-{i}")
    except:
        pass
    os.system(f"sbatch parallel.py run-{i}")
    