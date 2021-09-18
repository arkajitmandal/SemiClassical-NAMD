# Code for Performing Semi-Classical Quantum Dynamics
The methods that are implimented in this code are : PLDM, spin-PLDM, MFE (Mean-Field Ehrenfest) and SQC. The present code works for slurm based High-Performance Computing Cluster (HPCC) as well as on personal computers.  

# Usage  
### Step 1
Create a folder and git clone this repository.
```
git clone https://github.com/arkajitmandal/SemiClassical-NAMD
```
### Step 2
Code up the model system in a python file inside the "Model" folder and name it  'whateverModelName.py'.  

The 'whateverModelName.py' should look like:
```py
import numpy as np

class parameters():
   # some parameters
   # Nuclear Timestep (a.u.)
   dtN = 2

   # Number of Nuclear Steps 
   # length of simulation : dtN x Nsteps
   NSteps = 600  
   
   # Number trajectories per cpu
   NTraj = 100

   # Electronic Timestep (a.u.)
   # Please use even number
   dtE = dtN/40

   # Mass of nuclear particles in a.u.
   # is a vector of length NR : number 
   # of nuclear DOF
   M = np.array([1836.0])

   # Initial electronic state
   initState = 0
   
   # Save data every nskip steps
   nskip = 5

def Hel(R):
    # Diabatic potential energies Vij(R) in a.u., 
    # return Vij(R) : NxN Matrix 
    # (N  = number of electronic states)
    # R : Vector of length NR
    # (NR = number of nuclear degrees of freedom)
    # See details below 
    return Vij

def dHel0(R):
    # State independent gradient 
    # return dV0(R) : Vector of length NR
    # (NR = number of nuclear degrees of freedom) 
    # See details below
    return dV0

def dHel(R):
    # Gradient of the diabatic
    # potential energy matrix elements
    # Vij(R) in a.u.  
    # return dVij(R) : Array of dimention (N x N x NR)
    # See details below
    return dVij

def initR():
    # Provide initial values of R, P in a.u.
    # R, P : Both are vectors of length NR
    # (NR = number of nuclear degrees of freedom) 
    # See details below
    return R, P
```

You can find several examples of model files inside the "Model" folder. I will explain each parts of this file in more detain in a section below.


### Step 3 (simple)
Prepare an input file (name it : 'whateverInput.txt'):
```
Model                = tully2
Method               = pldm-focused 
```

* Model : The right hand side of the first line, _tully2_, tells the code to look for tully2.py inside the folder "Model". If you name your model file as  whateverModelName.py then you should write 'Model = whateverModelName' (without the '.py' part). 
* Method : Written as, method.methodOption. Select a quantum dynamics method. The available methods are :
  - **mfe** : Mean-Field Ehrenfest Approach. Kind of worst approach you can think of.
   - **pldm-focused** : Partial Linearized Density Matrix with focused initial conditions. Should be similar to mfe. Maybe slightly better. 
   - **pldm-sampled** : Partial Linearized Density Matrix (PLDM) with sampled initial conditions or the original PLDM approach. Most of the time works well, sometimes does not. Very good if your potentials are Hermonic (like Spin-Boson systems)
   - **spinpldm-all**: The Spin-Mapping PLDM approach with full sampling. Often better than PLDM. Reliable but slighly slow. If your initial electronic state is a pure state |i⟩⟨i| (you could start from a super position state, but you have to hack into this code to do that) use spinpldm-half to get the same result but much faster (by half).
   - **spinpldm-half**: The Spin-Mapping PLDM approach, but with our in-house approximation. Works perfectly if starting with an initial electronic state that is a pure state |i⟩⟨i| (you could start from a super position state, but you have to hack into this code to do that). 
   - **spinpldm-focused**: The Spin-Mapping PLDM approach, approximated. Good for short-time calculation and to get a general trend for longer time. 
   - **sqc-square**: The Symmetric Quasi-Classical Approach, with square window. Better than MFE. Cannot use it for more than several electronic states.  
   - **sqc-triangle**: The Symmetric Quasi-Classical Approach, with triangle window. Better than sqc-square.   

The output file containing population dynamics is 'method-methodOption-modelName.txt', for the above input file it would be: 

_pldm-focused-tully2.txt_

### Step 3 (for slurm on HPCC) 
Prepare an input file (name it : 'whateverInput.txt') for slurm submission in your computing cluster:
```
Model                = tully2
Method               = pldm-focused 

System               = slurm
Nodes                = 2
Cpus                 = 24
Partition            = action    
```
For first two lines see previous section. 

Last four lines provide additional commands for slurm submission. For adding additional slurm ('#SBATCH') command, add them in the preamble of the 'parallel.py'. The default preamble looks like:
```py
#!/usr/bin/env python
#SBATCH -o output.log
#SBATCH -t 1:00:00
```
Please dont add lines like "#SBATCH -N 1" or 
"#SBATCH --ntasks-per-node=24" in the preamble as they are declared in the input file. 


The output file containing population dynamics is 'method-methodOption-modelName.txt', for the above input file it would be: 

_pldm-focused-tully2.txt_

### Step 4 

Run the this code with python3. 

```
python3 run.py whateverInput.txt
```
Where 'whateverInput.txt' is the input file described above.
If your inputfile is named 'input.txt' then you could also just run,
```
python3 run.py
```

# More details into Model

A molecular Hamiltonian in the diabatic representation is written as:
![Hm](eqns/Hm.svg)

where __P<sub>k</sub>__ is the momentum for the __k__ th nuclear degrees of freedom with mass __M<sub>k</sub>__. Further, __V<sub>0</sub>(\{R<sub>k</sub>})__  and  __V<sub>ij</sub>(\{R<sub>k</sub>})__ are the state-independent and state-dependent part of the electronic Hamiltonian __H<sub>el</sub>(\{R<sub>k</sub>})__ in the diabatic basis {|i⟩}. That is:  __⟨i| Ĥ - ∑<sub>k</sub> P<sup>2</sup><sub>k</sub>/2M<sub>k</sub> |j⟩ = V<sub>ij</sub>(\{R<sub>k</sub>}) + V<sub>0</sub>(\{R<sub>k</sub>})δ<sub>ij</sub>__ .

_____________
_to be continued_...
<!---
That is, $V_{ij}(\{R_{k}\}) = \langle i| \hat{H} - \sum_{k}{{P}^{2}_{k}\over 2M_{k}} |j\rangle$. Of course most of times, we wave our hands, and make up models that describe $V_{ij}(\{R_{k}\})$ with some functions. If you know the analytical form of $V_{ij}(\{R_{k}\})$ you can write a model file: whateverModelName.py. 


For example consider a 1D dimentional model system, called the Tully's Model II. It has two electronic states and one nuclear DOF. Thus we write the Hamiltonian with one set of  $\{R,P\}$. _to be continued_...
-->

email: arkajitmandal@gmail.com