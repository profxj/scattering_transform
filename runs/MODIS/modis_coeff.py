''' Module to generate Scattering and coefficients for MODIS data'''

import numpy as np
import h5py
import sys, os

import torch

sys.path.append('../../')
import scattering

from IPython import embed

def coeff_from_preproc(preproc_file:str, outfile:str,
                       nsub:int=10000, debug:bool=False): 
    # Load up
    data = h5py.File(preproc_file, 'r')['valid']
    nimages = data.shape[0]

    if debug:
        nimages = 1000
        nsub = 400

    # Prep Scattering trnasform 
    M=N=64
    J=5
    L=4
    if not torch.cuda.is_available(): device='cpu'
    wavelets='morlet'
    l_oversampling=1
    frequency_factor=1
    st_calc = scattering.Scattering2d(M, N, J, L, device, wavelets, 
                        l_oversampling=l_oversampling, 
                        frequency_factor=frequency_factor)

    # Loop
    i0 = 0
    S0s, S1s, S2s = [], [], []
    while(i0 < nimages):
        i1 = min(i0+nsub, nimages)
        print(f'Processing images {i0} - {i1}')

        # Get images
        images = data[i0:i1,0,...]

        # Coeffs
        coeffs = st_calc.scattering_coef(images)

        # Save
        S0s.append(coeffs['S0'].numpy())
        S1s.append(coeffs['S1'].numpy())
        S2s.append(coeffs['S2'].numpy())

        # Add
        i0 += nsub

    # Concatenate
    S0 = np.concatenate(S0s)
    S1 = np.concatenate(S1s)
    S2 = np.concatenate(S2s)
    del S0s, S1s, S2s

    # Save
    with h5py.File(outfile, 'w') as f:
        f.create_dataset('S0', data=S0)
        f.create_dataset('S1', data=S1)
        f.create_dataset('S2', data=S2)

    print(f'Saved to {outfile}')

if __name__ == '__main__':
    preproc_file = feature_path = os.path.join(
        os.getenv('OS_SST'), 'MODIS_L2', 'PreProc', 
        'MODIS_R2019_2003_95clear_128x128_preproc_std.h5')
    coeff_from_preproc(preproc_file,
                       'MODIS_coeff.h5',
                       debug=True)
    