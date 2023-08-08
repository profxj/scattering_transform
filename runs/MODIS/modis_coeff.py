''' Module to generate Scattering and coefficients for MODIS data'''

import numpy as np
import h5py
import sys, os

import torch

sys.path.append('../../')
import scattering
sys.path.append('../')
import cutout_utils

from IPython import embed

if __name__ == '__main__':

    # 2009
    outfile = os.path.join(os.getenv('OS_SST'), 'MODIS_L2', 'Scattering', 
        'MODIS_2009_ST_coeff.h5')
    preproc_file = os.path.join(
        os.getenv('OS_SST'), 'MODIS_L2', 'PreProc', 
        'MODIS_R2019_2009_95clear_128x128_preproc_std.h5')
    
    cutout_utils.coeff_from_preproc(preproc_file, outfile, debug=False)
    