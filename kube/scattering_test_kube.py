import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.fft

import sys
sys.path.append('/scattering_transform/')

import scattering
import h5py

if __name__ == '__main__':
    feature_path = '../data/modis_data/MODIS_R2019_2003_95clear_128x128_preproc_std.h5'
    features = h5py.File(feature_path, 'r')['valid']
    num_samples = min(features.shape[0], 100)
    image_syn_list = []
    np.random.seed(0)
    # load image
    for i in range(num_samples-1):
        image_target = features[i]
        image_b = features[i+1]
        # s_cov_iso single field
        image_syn = scattering.synthesis('s_cov_iso', image_target, seed=0, image_init='random phase')
        image_syn_list.append(image_syn)

    with h5py.File('../data/modis_data/modis_image_syn_scattering_test.h5', 'w') as f:
        f.create_dataset("scattering_test", data=image_syn_list)