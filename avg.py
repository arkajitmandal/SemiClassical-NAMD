#!/usr/bin/env python3
import sys
import numpy as np
from glob import glob
print("-"*50)

try:
    fold = sys.argv[1]
except:
    fold = "output"


try:
    filename = sys.argv[2]
    filenames = [filename]
except:
    print ("Averaging all .txt files")
    fnames = glob(f"{fold}/*.txt")
    # get name from filenames: {name}-{number}.txt 
    # the name itself could include a dash
    names = ["-".join(f.split("/")[-1].split("-")[:-1]) for f in fnames]
    filenames = list(set(names))
    
    print(f"Averaging files that have the name:")
    for f in filenames:
        print(f"{f}-*.txt")
 
        
for filename in filenames:
    try:
        outName = filename + ".txt"

        fnames = glob(f"{fold}/{filename}-*.txt")
        dat = 0
        for i in fnames:
            dat += np.loadtxt(i)
        N = len(fnames)
        if i.find('sqc')!= -1:
            norm = np.sum(dat[:,1:], axis=1)
            for t in range(len(dat[:,0])):
                norm = np.sum(dat[t,1:])
                dat[t,1:] = dat[t,1:]/norm
                dat[t,0]  = dat[t,0]/N
            np.savetxt(outName, dat)
        else:
            np.savetxt(outName, dat/N)
        print ("Averaging done!")
    except Exception as e:
        print(f"Could not average {filename}-*.txt :")
        print(e)

