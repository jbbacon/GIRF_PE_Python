"""
Pypulseq code to generate the pulse sequence for GIRF calcualtion

Inspired by https://cds.ismrm.org/protected/22MProceedings/PDFfiles/0641.html for the optimised GIRF calculation 
and https://onlinelibrary.wiley.com/doi/10.1002/mrm.27902 for the 5*5 phase encoding

The code is written to have a slice offset in just one direction at a time. This can be changed using the directions variable. Set it to
directions  = ['z', 'x', 'y'] for the GIRF in the Z direction
directions  = ['y', 'z', 'x'] for the GIRF in the Y direction
directions  = ['x', 'y', 'z'] for the GIRF in the X direction

Takes approximately 2 hours per direction - Leave the repetitions at 1, the averaging comes from using the phase encoding 
"""

import pypulseq as pp
import numpy as np

Nx= 5
Ny = 5
fov = 200e-3
deltak = 1/fov


def ref(seq, system, slice_thickness=1e-3, slice_offsets=None, repetitions=1, directions=None):
    
    if slice_offsets is None:
        slice_offsets = [17e-3, -17e-3]  # Default slice offsets for each direction

    if directions is None:
        directions = ['x', 'y', 'z']  # Default slice offsets for each direction

# Iterate through each direction (X, Y, Z)
    # Loop over the phase encoding steps 
    # Loop over slice positions (positive and negative offset)

    for j in range(Nx):
        for k in range(Ny):
            for slice_offset in slice_offsets:
                # Design RF pulse
                flip_angle = 90  # Flip angle in degrees
                rf_duration = 4e-3  # Duration of RF pulse (3 ms)

                # Create a slice-selective Sinc pulse
                rf, gx, gxReph = pp.make_sinc_pulse(
                    flip_angle=np.deg2rad(flip_angle), 
                    duration=rf_duration, 
                    slice_thickness=slice_thickness, 
                    apodization=0.5, 
                    time_bw_product=4, 
                    system=system, 
                    return_gz=True
                )

                # Set the appropriate channel based on the direction
                gx.channel = directions[0]
                gxReph.channel = directions[0]

                # Apply frequency offset to target slice position
                rf.freq_offset = slice_offset * gx.amplitude  # Frequency offset in Hz

                # Create the phase encoding gradients 
                gxPE = pp.make_trapezoid(channel = directions[1], area = (j-(Nx-1)/2)*deltak, duration  = pp.calc_duration(gxReph), system = system)
                gyPE = pp.make_trapezoid(channel = directions[2], area = (k-(Ny-1)/2)*deltak, duration  = pp.calc_duration(gxReph), system = system)


                # Create the ADC event
                adc = pp.make_adc(num_samples=80000, dwell=5e-6, system=system)

                # Define the spoiler gradient in the appropriate direction
                gx_spoil = pp.make_trapezoid(channel=gx.channel, area=-gx.area / 2, system=system)
                grad_spoilx = pp.make_trapezoid(channel= directions[1], area=-gx.area / 2, system=system)
                grad_spoily = pp.make_trapezoid(channel= directions[2], area=-gx.area / 2, system=system)

                # Repeat the pulse sequence for each slice
                for _ in range(repetitions):
                    # Add the RF pulse and slice-select gradient to the sequence

                    seq.add_block(rf, gx)
                    seq.add_block(gxReph, gxPE, gyPE)

                    # Add the ADC event to the sequence
                    seq.add_block(adc)

                    # Add the spoiler gradient after the ADC (Only applied in the Offset slice)
                    seq.add_block(gx_spoil)

                    # Add a delay after the spoiler gradient (e.g., 2 ms)
                    TR = pp.make_delay(3)  # 3s delay
                    seq.add_block(TR)

def triangle(seq, system, triangular_amplitude_mT_per_m, slice_thickness=1e-3, slice_offsets=None, repetitions=1, directions=None):
    
    # Set default slice offsets if none are provided
    if slice_offsets is None:
        slice_offsets = [17e-3, -17e-3]  # Default to ±17 mm

    if directions is None:
        directions = ['x', 'y', 'z']

    # Define RF pulse parameters
    flip_angle = 90                 # Flip angle in degrees
    rf_duration = 4e-3              # Duration of RF pulse (3 ms)

    # Convert triangular gradient amplitude from mT/m to Hz/m
    triangular_amplitude_hz_per_m = triangular_amplitude_mT_per_m * system.gamma/1000  # system.gamma converts to Hz/m
    rise_time = triangular_amplitude_hz_per_m / system.max_slew  # Calculate rise time in seconds

    # Loop through each direction (X, Y, Z)
    for j in range(Nx):
        for k in range(Ny):
            for amplitude_sign in [1, -1]: 
                for i in range(repetitions):
                    for offset in slice_offsets:
                        # Create the RF pulse with updated frequency offset for each slice position
                        rf, grad, gradReph = pp.make_sinc_pulse(
                            flip_angle=np.deg2rad(flip_angle), 
                            duration=rf_duration, 
                            slice_thickness=slice_thickness, 
                            apodization=0.5, 
                            time_bw_product=4, 
                            system=system, 
                            return_gz=True
                        )
                        
                        # Set gradients to the appropriate channel for slice selection
                        grad.channel = directions[0]
                        gradReph.channel = directions[0]

                        # Apply frequency offset to target the desired slice position
                        rf.freq_offset = offset * grad.amplitude  # Frequency offset in Hz
                        gxPE = pp.make_trapezoid(channel = directions[1], area = (j-(Nx-1)/2)*deltak, duration  = pp.calc_duration(gradReph), system = system)
                        gyPE = pp.make_trapezoid(channel = directions[2], area = (k-(Ny-1)/2)*deltak, duration  = pp.calc_duration(gradReph), system = system)

                        # Create the ADC event
                        adc = pp.make_adc(num_samples=80000, dwell=5e-6, system=system)

                        # Define the spoiler gradient in the current direction
                        grad_spoilz = pp.make_trapezoid(channel= directions[0], area=-grad.area / 2, system=system)
                        grad_spoilx = pp.make_trapezoid(channel= directions[1], area=-grad.area / 2, system=system)
                        grad_spoily = pp.make_trapezoid(channel= directions[2], area=-grad.area / 2, system=system)

                        # Create the triangular gradient in the current direction with adjusted sign
                        grad_triangular = pp.make_trapezoid(
                            channel=directions[0],
                            amplitude=amplitude_sign * triangular_amplitude_hz_per_m,
                            flat_time=0,  # No flat time for triangular shape
                            rise_time=rise_time,
                            fall_time=rise_time,
                            delay=2e-3,  # Start 2 ms after ADC start
                            system=system
                        )
                    
                        # Add the RF pulse and slice-select gradient to the sequence
                        seq.add_block(rf, grad)
                        seq.add_block(gradReph, gxPE, gyPE)
                        # Add the ADC event and triangular gradient to the sequence
                        seq.add_block(adc, grad_triangular)

                        # Add the spoiler gradient after the ADC
                        seq.add_block(grad_spoilz)

                        # Add a delay after the spoiler gradient (e.g., 2 ms)
                        post_spoiler_delay = pp.make_delay(3)  # 2 ms delay
                        seq.add_block(post_spoiler_delay)
    



seq=pp.Sequence()
system = pp.Opts(max_grad=40, grad_unit='mT/m', max_slew=180, slew_unit='T/m/s', rf_ringdown_time=20e-6, rf_dead_time=100e-6, grad_raster_time=10e-6, adc_dead_time=1e-5)
slice_thickness = 1e-3  # Slice thickness in meters
slice_offsets = [17e-3, -17e-3]  # Slice offsets in meters
repetitionsRef = 1  # Number of repetitions per configuration, LEAVE AT 1
repetitionsTri = 1


triangular_amplitudes = [9, 10.8, 12.6, 14.4, 16.2, 18, 19.8, 21.6, 23.4, 25.2, 27, 28.8, 30.6, 32.4, 34.2, 36, 37.8, 39.6]

directions  = ['z', 'x', 'y']

for i, amplitude in enumerate(triangular_amplitudes):
    if i % 3 == 0:  # Every even index (0, 3, 6, ...) means call ref first
        ref(seq=seq, system=system, slice_thickness=slice_thickness, repetitions=repetitionsRef, directions=directions, slice_offsets=slice_offsets)
    
    triangle(seq=seq, system=system, triangular_amplitude_mT_per_m=amplitude, slice_thickness=slice_thickness, repetitions=repetitionsTri, directions=directions, slice_offsets=slice_offsets)

#Can adjust for a better shim if doing just 1 direction 
seq.set_definition(key='FOV', value=[fov, fov, fov])

isok, report = seq.check_timing()
print(isok)
print(report)

#Optional plotting 
#seq.plot(time_range=(0,200))


filename = f'/Users/jamesbacon/Library/CloudStorage/OneDrive-Nexus365/Sequences130325/ZFull.seq'
#seq.write(filename)


