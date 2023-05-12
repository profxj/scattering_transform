import torch
from torch.utils.data import DataLoader, Dataset
import h5py 
import pandas
import random
import numpy as np

class MODISDataset(Dataset):
    """
    MODIS SST dataset for scattering transform.
    """
    def __init__(self, feature_path, data_key):
        self.data_key = data_key
        self.features = self._open_file(feature_path)
        num_samples = len(self.features)
        index_list = list(range(num_samples))
        random.shuffle(index_list)
        self.index_list = index_list
        
    def _open_file(self, feature_path):
        features = h5py.File(feature_path, 'r')[self.data_key]
        features = np.squeeze(features, axis=1)
        #print(features.shape)
        return features
               
    def __len__(self):
        num_sample_pairs = len(self.features) // 2
        return num_sample_pairs

    def __getitem__(self, global_idx):
        index_pair = (self.index_list[global_idx * 2], self.index_list[global_idx * 2 + 1])
        feature_target = self.features[index_pair[0]]
        feature_b = self.features[index_pair[1]]
        return feature_b, feature_target

def modis_loader(feature_path, batch_size=1, data_key='valid'):
    """
    This is a function used to create a LLCFS data loader.
    
    Args:
        feuture_path: (str) path of feature file;
        label_path: (str) path of label file;
        file_id: (str) id of the file offerering the latents;
        batch_size: (int) batch size;
        train_flag: (str) flag of train or valid mode;
        
    Returns:
        loader: (Dataloader) MODIS DT Dataloader.
    """
    modis_dataset = MODISDataset(
        feature_path, 
        data_key=data_key,
    )
    loader = torch.utils.data.DataLoader(
        modis_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=False
    )
    return loader