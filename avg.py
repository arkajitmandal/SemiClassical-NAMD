import sys
import numpy as np
from glob import glob

try:
    filename = sys.argv[1]
    filenames = [filename]
except:
    print ("Averaging all .txt files")
    filenames = [i.replace("RUN/run-0/","") for i in glob("RUN/run-0/*.txt")]
    print (filenames)

try:
    fold = int(sys.argv[2])
except:
    print (f"Detected {len(glob('RUN/run-*'))} folders")
    fold = len(glob("RUN/run-*"))
    if fold == 0:
        print("No outputs found!")
        
for filename in filenames:

    
    outName = filename


    dirs = [f"RUN/run-{i}" for i in  range(fold)]

    dat = np.loadtxt(dirs[0]+"/"+filename) 
    for i in range(1, fold):
        dat += np.loadtxt(dirs[i]+"/"+filename)
    np.savetxt(outName,dat/fold )