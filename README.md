# GIRF_PE_Python
Full Python code and terminal integration to generate the GIRF sequence, process the data from the scanner, calculate and view the GIRF. No field camera/additional hardware required. This repositry allows the GIRF to be calculated up to third order spherical harmonics.
GIRF (Gradient Impulse Response Function) as described in https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/mrm.24263.

## Installation and dependencies
1. Install [Pixi](https://pixi.sh) as described [online](https://pixi.sh/latest/), e.g.

```
curl -fsSL https://pixi.sh/install.sh | bash
```

2. Clone this repository
```
git clone https://github.com/jbbacon/GIRF_PE_Python.git
```

3. Change to the cloned directory
```
cd GIRF_PE_Python
```

4. Run a script using "pixi run python {script}". For example
```
pixi run python Pypulseq_GIRF_PE.py
```
_or_ use one of the predefined tasks
```
pixi run gen-seq
```

## Contents
**Pypulseq_GIRF_PE.py** - Pypulseq code to create the .seq file used on the scanner to collect GIRF data. 

Inspired by https://cds.ismrm.org/protected/22MProceedings/PDFfiles/0641.html and https://onlinelibrary.wiley.com/doi/10.1002/mrm.27583 for the optimised GIRF calcualtion, and https://onlinelibrary.wiley.com/doi/10.1002/mrm.27902 for the 2D Phase Encoding. Pypulseq: https://github.com/imr-framework/pypulseq.
Phase Encoding is elliptical.

Notes: The ADC event has 50000 points, when on the scanner this should be split up into readout events of length of 5000. Lib Balance should be disabled on the scanner. Scanner requires pulseq compiler >= 1.4.3.

Choice of phantom is important; we recommend a spherical water based phantom with a small diameter (~10cm), agar has also worked. This must be spectroscopically simple (contain only one chemical shift environment).

Run the script through terminal as
```
pixi run gen-seq --direction x --output /path/to/output/folder
```

Other tags exists for plotting capability, adjusting slice thickness, slice offsets, fov, number of phase encoding steps. 

To fully characterise the scanner this should be called 3 times with the direction being x, y, z respectively. The default options will create a sequence which is ~90 minutes for each direction.

The function will output the .seq file, a .csv containing information about the gradient order, a .json file containing parameters of the sequence and a .npz file containing additional gradient information.

---

**GIRF_PE_Processing.py** - This process the raw scanner data (currently only Siemens) performing 2D Fourier Transform, coil combination and arranging the data into a sensible format. Zero padding can be defined before the Fourier Transform to improve the spatial resolution.

Run the script through terminal as
```
pixi run proc-data --mri_file /path/to/mri_data.dat --csv_file /path/to/pulse_order_log_x.csv

--json_file /path/to/parameters.json --npz_file /path/to/InputGradients.npz

--direction x --output_folder /path/to/output/folder
```

This will need to be run 3 times for the 3 directions. Options for 2D filters and zero padding can be applied before Fourier Transform.

---

**PE_script_OptimizedGIRFCalculation.py** - Main script to calculate and view the GIRF. Inspiration of this code was originigally a translation of the MATLAB code provided in https://cds.ismrm.org/protected/22MProceedings/PDFfiles/0641.html https://github.com/BRAIN-TO/girfISMRM2022.git, it has since been heavily modified. Helper functions are all contained in help_functions_GIRF.

Run the script through terminal as
```
pixi run get-girf --data_path /path/to/data --direction x --json_file /path/to/json/file
```
Other tags to adjust the plotting, order of the spherical harmonics, file saving, the start point and end point of the data, thresholding for voxel selection.

For troubleshooting reduce the number of points (--f tag), or increase/decrease the max/min threshold.

---

## Example Data 

Example processed data from a 3T Siemens Prisma scanner can be found here: https://zenodo.org/records/17019062. 
