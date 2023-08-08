''' Module to generate Scattering and coefficients for MODIS data'''

import numpy as np
import h5py
import sys, os

import torch

sys.path.append('../../')
import scattering

from IPython import embed

def coeff_from_preproc(preproc_file:str, outfile:str=None,
                       nsub:int=20000, debug:bool=False): 
    """ Calculate Scattering Coefficients for a PreProc file

    Args:
        preproc_file (str): PreProc file of cutouts
        outfile (str, optional): _description_. Defaults to None.
        nsub (int, optional): _description_. Defaults to 20000.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # Load up
    print(f"Loading {preproc_file}")
    data = h5py.File(preproc_file, 'r')['valid']
    nimages = data.shape[0]

    if debug:
        nimages = 1000
        nsub = 400

    print(f"Processing {nimages} images in {nsub} sub-batches")

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

    S0s, S1s, S2s = [], [], []
    I02s = []
    # Loop
    i0 = 0
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

        # I0
        I02s.append(np.mean(images**2, axis=(-1,-2)))
        #embed(header='64 of modis')

        # Next loop
        i0 += nsub

    # Concatenate
    S0 = np.concatenate(S0s)
    S1 = np.concatenate(S1s)
    S2 = np.concatenate(S2s)
    I02 = np.concatenate(I02s)
    del S0s, S1s, S2s, I02s

    # Save
    if outfile is None:
        pass
    elif debug and os.path.isfile(outfile):
        print(f"Not over-writing file in debug mode")
    else:
        with h5py.File(outfile, 'w') as f:
            f.create_dataset('S0', data=S0)
            f.create_dataset('S1', data=S1)
            f.create_dataset('S2', data=S2)
            f.create_dataset('I02', data=I02)
        print(f'Saved to {outfile}')

    return S0, S1, S2, I02
