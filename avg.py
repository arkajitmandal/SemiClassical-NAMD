import sys
import os
import numpy as np
filename = sys.argv[1]
 
outName = filename
fold = int(sys.argv[2])

dirs = [f"run-{i}" for i in  range(fold)]

dat = np.loadtxt(dirs[0]+"/"+filename) 
for i in range(1, fold):
    dat += np.loadtxt(dirs[i]+"/"+filename)
np.savetxt(outName,dat/fold )