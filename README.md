# Code for Performing Semi-Classical Quantum Dynamics
The methods that are implimented in this code are : PLDM (Parital Linearized Density Matrix), spin-PLDM, MFE (Mean-Field Ehrenfest), various SQC (Symmetric Quasi-Classical Approach) and N-RPMD (Nonadiabatic Ring-Polymer Molecular Dynamics). The present code works for slurm based High-Performance Computing Cluster (HPCC), HTcondor based High-Throughput Computing (HTC) as well as on personal computers.  

# Usage  
### Step 1
Create a folder and git clone this repository.
```
git clone https://github.com/arkajitmandal/SemiClassical-NAMD
```
### Step 2
Code up the model system in a python file inside the "Model" folder and name it  'modelName.py'.  

The 'modelName.py' should look like:
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

def initHel0(R):
    #------ This part will be only for NRPMD----------
    #-------while running condensed phase system------
    #R : is a 2D array of dimensionality ndof,nb
         #where, nb is the bead and ndof is the number of dofs
    #M : mass of the particle
    #ω = frequency of the particle
    #R0 = initial positon of the photo-excitation 
    # see details below
    return  np.sum(0.5 *M* ω**2 * (R-R0)**2.0)
```

You can find several examples of model files inside the "Model" folder. I will explain each parts of this file in more detain in a section below.


### Step 3 (simple | serial; on your pc/mac/linux)
Prepare an input file (name it : 'whateverInput.txt'):
```
Model                = tully2
Method               = pldm-focused 
```

* Model : The right hand side of the first line, _tully2_, tells the code to look for tully2.py inside the folder "Model". If you name your model file as  modelName.py then you should write 'Model = modelName' (without the '.py' part). 
* Method : Written as, method-methodOption. Select a quantum dynamics method. The available methods are :
  - **mfe** : Mean-Field Ehrenfest Approach. Kind of worst approach you can think of.
   - **pldm-focused** : Partial Linearized Density Matrix (PLDM) [1] with focused initial conditions. Should be similar to mfe. Maybe slightly better. 
   - **pldm-sampled** : Partial Linearized Density Matrix (PLDM) [1] with sampled initial conditions or the original PLDM approach. Most of the time works well, sometimes does not. Very good if your potentials are Hermonic (like Spin-Boson systems)
   - **spinpldm-all**: The Spin-Mapping PLDM [2] approach with full sampling. Often better than PLDM. Reliable but slighly slow. If your initial electronic state is a pure state |i⟩⟨i| (you could start from a super position state, but you have to hack into this code to do that) use spinpldm-half to get the same result but much faster (by half).
   - **spinpldm-half**: The Spin-Mapping PLDM approach, but with our in-house approximation. Works perfectly if starting with an initial electronic state that is a pure state |i⟩⟨i| (you could start from a super position state, but you have to hack into this code to do that). 
   - **spinpldm-focused**: The Spin-Mapping PLDM approach, approximated. Good for short-time calculation and to get a general trend for longer time. 
   - **sqc-square**: The Symmetric Quasi-Classical Approach, with square window [3]. Better than MFE. Cannot use it for more than several electronic states.  
   - **sqc-triangle**: The Symmetric Quasi-Classical Approach, with triangle window [4]. Better than sqc-square.   
   - **zpesqc-triangle**: The zero-point energy corrected Symmetric Quasi-Classical Approach [5], with triangle window. As good as spin-PLDM or better.  
   - **zpesqc-square**: The zero-point energy corrected Symmetric Quasi-Classical Approach [5], with square window. Slightly worse than zpesqc-triangle.
   - **spinlsc**: Spin-LSC approach, sort of simpler version of Spin-PLDM. I think this is actually a great method. 

   - **nrpmd-n** : The non-adiabatic ring polymer molecular dynamics[6] framework for aims to captures nuclear quantum effects while predicting efficient short-time and reliable longer time
   dynamics. Reasonable results for electron/charge transfer dynamics. Here n represents the number of beads, i.e. nrpmd-5 means each nuclear degrees of freedom is described with 5 ring-polymer beads.  

   - **mash** : Multistate Mapping Approach to Surface Hopping approach. [7] 

The output file containing population dynamics is 'method-methodOption-modelName.txt', for the above input file it would be: 

_pldm-focused-tully2.txt_

### Step 3 (advanced | parallel jobs; slurm on HPCC) 
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

Last four lines provide additional commands for slurm submission. For adding additional slurm ('#SBATCH') command, add them in the preamble of the 'serial.py'. The default preamble looks like:
```py
#!/usr/bin/env python
#SBATCH -o output.log
#SBATCH -t 1:00:00
```
Please dont add lines like "#SBATCH -N 1" or 
"#SBATCH --ntasks-per-node=24" in the preamble as they are declared in the input file. 


The output file containing population dynamics is 'method-methodOption-modelName.txt', for the above input file it would be: 

_pldm-focused-tully2.txt_

### Step 3 (advanced | parallel jobs; htcondor on HTC) 
Prepare an input file (name it : 'whateverInput.txt') for slurm submission in your computing cluster:
```
Model                = morse1
Method               = mfe

System               = htcondor
Cpus                 = 10

```
For first two lines described the model and method. 

Last line provide just as before indicate the number of cpus to be used.
After all jobs are done, run the following python script to get the output file. 
```
$ python avg.py
```

The output file containing population dynamics is 'method-methodOption-modelName.txt', for the above input file it would be: 

_mfe-morse1.txt_


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

# Details of Model Hamiltonian
In all of the approaches coded up here, the nuclear DOF $\{R_k, P_k\}$ are evolved classically (their equation motion evolves under a classical like force) and the electronic DOF are described with the diabatic electronic states $\{|i\rangle\}$.  

A molecular Hamiltonian in the diabatic representation is written as:

$$\hat{H} = \frac{P_k^2}{2M_k} + V_{0}(\{R_k\}) + \sum_{ij}V_{ij}(\{R_k\})|i \rangle \langle j| = \sum_{k} T_{R_k} + \hat{H}_{el}(\{R_k\})$$


where $P_k$ is the momentum for the $k$ th nuclear degrees of freedom with mass $M_k$. Further, $V_{0}(\{R_k\})$ and  $V_{ij}(\{R_k\})$ are the state-independent and state-dependent part of the electronic Hamiltonian ${\hat{H}_{el}(\{R_k\})}$ in the diabatic basis $\{|i\rangle\}$. That is: ${\langle i | \hat{H}_{el}(\{R_k\}) |j \rangle} =  V_{0}(\{R_k\})\delta_{ij} + V_{ij}(\{R_k\})$. Write the analytical form of $V_{ij}(\{R_k\})$ you can write a model file: **modelName.py**. 


One can always, set $V_{0}(\{R_k\})= 0$, and instead redefine $V_{ij}(\{R_k\}) \rightarrow V_{0}(\{R_k\})\delta_{ij} + V_{ij}(\{R_k\})$ and they should be equivalent in principle. However, some of the semiclassical approaches (**pldm-sampled**, **sqc-square** and **sqc-triangle**) produce results that depend on how one separates the state-independent and state-dependent parts of the gradient of the electronic Hamiltonian. The nuclear forces computed in all of these approaches assumes this general form:

$F_k = - \nabla_k V_{0}(\{R_k\}) - \sum_{ij}  \nabla_k V_{ij}(\{R_k\}) \cdot \Lambda_{ij}$

where the definition of $\Lambda_{ij}$ depends on the quantum dynamics method. For example, in MFE, $\Lambda_{ij} = c_i^* c_j$.  For methods that have ∑<sub>i</sub>Λ<sub>ii</sub> = 1 (like MFE) for individual trajectories this separation of state-dependent and independent does not matter. 

## Details of a model file ('modelName.py')
**_NOTE:_** You **dont** need to code up  $V_{0}(\{R_k\})$. 

### Hel(R)
In the Hel(R) function inside the 'modelName.py' one have to define NxN matrix elements of the state-dependent electronic part of the Hamiltonian. Here you will code up  $V_{ij}(\{R_k\})$.

### dHel(R)
In the dHel(R) function inside the 'modelName.py' one have to define NxNxNR matrix elements of the state-dependent gradient electronic part of the Hamiltonian. Here you will code up  $\nabla_k V_{ij}(\{R_k\})$.

### dHel0(R)
In the dHel0(R) function inside the 'modelName.py' one have to define a array of length NR describing the state-independent gradient of electronic part of the Hamiltonian. Here you will code up  $\nabla_k V_{0}(\{R_k\})$.


### initR()
Sample $R, P$ from a Wigner distribution. To obtain the wigner distribution, one needs to start with an initial density matrix. For example, for an wavefunction $|\chi \rangle$ write the density matrix $\hat{\rho}_N = |\chi  \rangle \langle \chi|$, then the Wigner transform is performed as,

${\hat{\rho}^W_N}({R, P}) = \frac{1}{\pi\hbar} \int_{-\infty}^{\infty} \langle {R} - \frac{S}{2}|\hat{\rho}_N |{R} + \frac{S}{2} \rangle e^{iPS} dS$

The $R, P$ is then sampled from $\hat{\rho}_N^{W}({R, P})$.

_____________
_to be continued_...
<!--- 

For example consider a 1D dimentional model system, called the Tully's Model II. It has two electronic states and one nuclear DOF. Thus we write the Hamiltonian with one set of  $\{R,P\}$. _to be continued_...

[6] Braden, Mandal and Huo __J. Chem. Phys. 155, 084106__
-->
## Authors
* Arkajit Mandal
* Braden Weight
* Sutirtha Chowdhury
* Eric Koessler
* Elious M. Mondal
* Haimi Nguyen
* Muhammad R. Hasyim
* Wenxiang Ying
* James F. Varner

## References
_____________
[1] Huo and Coker __J. Chem. Phys. 135, 201101 (2011)__\
[2] Mannouch and Richardson __J. Chem. Phys. 153, 194109 (2020)__\
[3] Cotton and Miller __J. Chem. Phys. 139, 234112 (2013)__\
[4] Cotton and Miller __J. Chem. Phys. 145, 144108 (2016)__\
[5] Cotton and Miller __J. Chem. Phys. 150, 194110 (2019)__\
[6] S. N. Chowdhury and P.Huo __J. Chem. Phys. 147, 214109 (2017)__\
[7] J. E. Runeson and D. E. Manolopoulos __J. Chem. Phys. 150, 244102 (2019)__ 


email: arkajitmandal@gmail.com