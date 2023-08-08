''' Module to generate Scattering and coefficients for VIIRS data'''

import numpy as np
import h5py
import sys, os

import torch

sys.path.append('../../')
import scattering
sys.path.append('../')
import cutout_utils

from ulmo import io as ulmo_io

from IPython import embed

if os.getenv('OS_SST') is not None:
    local_viirs98_file = os.path.join(os.getenv('OS_SST'),
                                  'VIIRS', 'Tables', 
                                  'VIIRS_all_98clear_std.parquet')


def run_viirs98(debug:bool=False):

    outfile = os.path.join(
        os.getenv('OS_SST'), 'VIIRS', 'Scattering', 
        'VIIRS_98clear_all.h5')
    if debug:
        outfile = 'tst.h5'

    # Load in the VIIRS 98 clear table
    tbl = ulmo_io.load_main_table(local_viirs98_file)

    pre_proc_files = np.unique(tbl.pp_file.values)

    if debug:
        pre_proc_files = pre_proc_files[:2]

    f = h5py.File(outfile, 'w')
    for pp_file in pre_proc_files: 
        base = os.path.basename(pp_file)
        year = base[6:10]

        # Grab the year
        f.create_group(year)

        # Go local
        preproc_file = os.path.join(
            os.getenv('OS_SST'), 'VIIRS', 'PreProc', 
            base)

        #  Run
        S0, S1, S2, I02 = cutout_utils.coeff_from_preproc(
            preproc_file, debug=debug)

        # Write 
        f[year].create_dataset('S0', data=S0)
        f[year].create_dataset('S1', data=S1)
        f[year].create_dataset('S2', data=S2)
        f[year].create_dataset('I02', data=I02)
    
    f.close()

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # VIIRS 98 clear
    if flg & (2**0):
        run_viirs98()#debug=True)


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- VIIRS 98
    else:
        flg = sys.argv[1]

    main(flg)
