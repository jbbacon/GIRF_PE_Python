# GIRF_PE_Python
Full Python code and terminal integration to generate the GIRF sequence, process the data from the scanner, calculate and view the GIRF.

Pypulseq_GIRF_PE.py - this is the first script to run. This creates the .seq file which can be used on the scanner to collect the GIRF data. 
Inspired by https://cds.ismrm.org/protected/22MProceedings/PDFfiles/0641.html for the optimised GIRF calcualtion and https://onlinelibrary.wiley.com/doi/10.1002/mrm.27902 for the 2D Phase Encoding. 
Notes: The ADC event has 80000 points, when on the scanner this should be split up into readout events with a length of 8000. Lib Balance should also be turned off on the scanner. 

The function should be called through terrminal as 
python Pypulseq.py --direction x --output /path/to/output/folder 
Many other tags also exists for plotting capabaility, adjusting slice thickness, slice offset, fov, number of phase encoding steps, number of gradients 
To fully characterise the scanner this should be called 3 times with the direction being x, y, z respectively. 
The function will output the .seq file, a .csv containing information about the gradient order and a .mat file containing parameters of the sequence and a second .mat file called InputGradients.mat. The full sequence will take ~2 hours to run per primary gradient direction


GIRF_PE_Processing.py - this takes the .dat file output from the scanner and will process it to perform coil combination, 2D Fourier Transform and arange the data into a sensible format. This may take ~10 minutes to run per direction.

This function should be called through terminal as 
python GIRF_PE_Processing --mri_filename /path/to/.dat/file --csv_filename /path/to/pulse_order_log.csv --mat_filename /path/to/parameters.mat --direction x --output_folder /path/to/output/folder
This will again need to be run 3 times for the 3 directions. 
Additionally the InputGradient.mat file geenerate previously needs to be moved into the output_folder from this function. 


PE_script_OptimizedGIRFCalculation.py - Main script to use in order to obain the GIRF. With some minor modifications this is a translation of the MATLAB code provided in https://cds.ismrm.org/protected/22MProceedings/PDFfiles/0641.html. Helper functions are all contained in help_functions_GIRF

This function should be called through terminal as
python PE_script_OptimizedGIRFCalculation.py --data_path /path/to/data --direction x 
There are then many options to adjust the plotting, whether this displays the B0 or self term of the GIRF, if the file is saved, the start point and end point of the data 
