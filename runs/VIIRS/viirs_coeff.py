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

coeff_file = os.path.join(
        os.getenv('OS_SST'), 'VIIRS', 'Scattering', 
        'VIIRS_98clear_all.h5')

scatt_tbl_file = os.path.join(
        os.getenv('OS_SST'), 'VIIRS', 'Scattering', 
        'VIIRS_all_98clear_scatt.parquet')

def run_viirs98(debug:bool=False):

    outfile = coeff_file

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
        if year != '2012':
            continue

        # Grab the year
        f.create_group(year)

        # Go local
        preproc_file = os.path.join(
            os.getenv('OS_SST'), 'VIIRS', 'PreProc', 
            base)

        #  Run
        print(f'Processing year={year}')
        S0, S1, S2, I02 = cutout_utils.coeff_from_preproc(
            preproc_file, debug=debug)

        # Write 
        f[year].create_dataset('S0', data=S0)
        f[year].create_dataset('S1', data=S1)
        f[year].create_dataset('S2', data=S2)
        f[year].create_dataset('I02', data=I02)
    
    f.close()

def slurp_coeffs(debug:bool=False, only_2012:bool=True):

    # Load in the VIIRS 98 clear table
    tbl = ulmo_io.load_main_table(local_viirs98_file)
    pre_proc_files = np.unique(tbl.pp_file.values)

    # Load the coefficients
    f_coeff = h5py.File(coeff_file, 'r')

    J = f_coeff['2012']['S1'].shape[-2]
    L = f_coeff['2012']['S1'].shape[-1]

    # Init
    dum = np.ones(len(tbl), dtype=float) * np.nan
    for jj in range(J):
        tbl[f'S1_iso_{jj}'] = dum
    tbl['s21'] = dum
    tbl['s22'] = dum

    # Loop on the years
    for year in f_coeff.keys():
        if only_2012 and year != '2012':
            continue
        scoeffs = scattering.io.load_coeffs(f_coeff[year])
        # Indexing
        idx = tbl.pp_file.str.contains(year)
        sub_tbl = tbl[idx]

        # S1_iso
        for jj in range(J):
            tbl[f'S1_iso_{jj}'].values[idx] = scoeffs['S1_iso'][sub_tbl.pp_idx, jj]

        # s21
        for key in ['s21', 's22']:
            tbl[key].values[idx] = scoeffs[key][sub_tbl.pp_idx]

    # Write
    ulmo_io.write_main_table(tbl, scatt_tbl_file, to_s3=False)

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # VIIRS 98 clear
    if flg & (2**0):
        run_viirs98()#debug=True)

    # Slurp
    if flg & (2**1):
        slurp_coeffs()


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- VIIRS 98
        #flg += 2 ** 1  # 2 -- Slurp i coefficients
    else:
        flg = sys.argv[1]

    main(flg)
