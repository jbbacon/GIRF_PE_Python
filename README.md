# GIRF_PE_Python
Python code to obtain GIRF including sequence, processing and calculation 

## Installation and dependencies
1. Install [Pixi](https://pixi.sh) as described [online](https://pixi.sh/latest/), e.g.

```
curl -fsSL https://pixi.sh/install.sh | bash
```

2. Clone this repository
```
git clone https://github.com/jbbacon/GIRF_PE_Python.git
```

3. CD to directory
```
cd GIRF_PE_Python
```

4. Run a scrip using `pixi run {script}```. For example
```
pixi run python Pypulseq_GIRF_PE.py
```

## Contents
Pypulseq_GIRF_PE.py - this is the first script to run. This creates the .seq file which can be used on the scanner to collect the GIRF data. 
Inspired by https://cds.ismrm.org/protected/22MProceedings/PDFfiles/0641.html for the optimised GIRF calcualtion and https://onlinelibrary.wiley.com/doi/10.1002/mrm.27902 for the 2D Phase Encoding. 
Notes: Leave RTri and RRef = 1, the averaging is done by the phase encoding. Script is written to generate data in 1 direction at a time. Needs to be run 3 times for x, y and z directions. Scan Time ~2 hours per direction. 

GIRF_PE_Processing.ipynb has an input of the .dat file from the scanner and performs coil combination, 2D FT and orgnaises the data sensibly to be used for the GIRF calcualtion. Ensure to add the InputGradients.mat file to the output folder this script will generate. This will take ~ 10 minutes to run per direction due to the meory requirements for such a large dataset.

PE_script_OptimizedGIRFCalculation.py is the main script to use in order to obain the GIRF. With some minor modifications this is a translation of the MATLAB code provided in https://cds.ismrm.org/protected/22MProceedings/PDFfiles/0641.html. Ensure to set the start point, end point, direction and Linear/B0 term before running the script. The help_functions_GIRF folder contains the definitions this script will call. 
