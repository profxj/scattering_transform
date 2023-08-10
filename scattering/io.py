""" Module for I/O of coefficients"""
import numpy as np
import h5py

def load_coeffs(scatt_dict:dict):

    #f = h5py.File(coeff_file, 'r')

    J = scatt_dict['S1'].shape[-2]
    L = scatt_dict['S1'].shape[-1]

    # TODO -- move this to a more sensible spot
    # Calculate a few quantities
    S1 = scatt_dict['S1'][:]
    S1_iso = S1.mean(-1)

    S2 = scatt_dict['S2'][:]
    S2_iso = np.zeros((S2.shape[0],J,J,L))
    for l1 in range(L):
        for l2 in range(L):
            S2_iso [:,:,:,(l2-l1)%L] += S2 [:,:,:,l1,l2]
    S2_iso  /= L

    s21 = S2_iso.mean(-1) / S1_iso[:,:,None]
    s22 = S2_iso[:,:,:,0] / S2_iso[:,:,:,L//2]

    avg_s21 = np.nanmean(s21, axis=(-1,-2))
    avg_s22 = np.nanmean(s22, axis=(-1,-2))

    # Pack it up
    coeffs = {}
    coeffs['S1'] = S1
    coeffs['S1_iso'] = S1_iso
    coeffs['S2_iso'] = S2_iso
    coeffs['s21'] = avg_s21
    coeffs['s22'] = avg_s22
    coeffs['I02'] = scatt_dict['I02'][:]

    return coeffs