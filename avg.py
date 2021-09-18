#!/usr/bin/env python
import sys
import numpy as np
from glob import glob
print("-"*50)

def getInput(input,key):
    txt = [i for i in input if i.find(key)!=-1][0].split("=")[1].split("#", 1)[0].replace("\n","")
    return txt.replace(" ","")

try:
    input = open(sys.argv[1], 'r').readlines()
    print(f"Reading {sys.argv[1]} for average")
except:
    print("Reading input.txt for average")
    input = open('input.txt', 'r').readlines()

try:
    filename = sys.argv[2]
    filenames = [filename]
except:
    print ("Averaging all .txt files")
    filenames = [i.replace("RUN/run-0/","") for i in glob("RUN/run-0/*.txt")]
    print (filenames)

try:
    fold = int(sys.argv[3])
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
    method = getInput(input,"Method")
    if method.find('sqc')!= -1:
        norm = np.sum(dat[:,1:], axis=1)
        for t in range(len(dat[:,0])):
            norm = np.sum(dat[t,1:])
            dat[t,1:] = dat[t,1:]/norm
            dat[t,0]  = dat[t,0]/fold
        np.savetxt(outName, dat)
    else:
        np.savetxt(outName, dat/fold)

print ("Gathered all data, feel free to remove the folder 'RUN'")
