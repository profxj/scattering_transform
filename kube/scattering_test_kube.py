import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.fft
from utils import *

import sys
#sys.path.append('/home/jovyan/scattering_transform/')
sys.path.append('/scattering_transform/')

import scattering
import h5py

if __name__ == '__main__':
    feature_path = '../data/modis_data/MODIS_R2019_2003_95clear_128x128_preproc_std.h5'
    modis_data = modis_loader(feature_path)
    num_features = len(modis_data)
    num_samples = 100000
    num_samples = min(num_features, num_samples)
    image_syn_list = []
    np.random.seed(0)
    num_iter = 0
    # load image
    for image_target, image_b in modis_data:
        if num_iter >= num_samples:
            break
        # s_cov_iso single field
        image_syn = scattering.synthesis('s_cov_iso', mode='image', target=image_target, image_b=image_b, seed=0, image_init='random phase')
        image_syn_list.append(image_syn)
        num_iter += 1

    with h5py.File('../data/modis_data/modis_image_syn_scattering_test.h5', 'w') as f:
        f.create_dataset("scattering_test", data=image_syn_list)
        