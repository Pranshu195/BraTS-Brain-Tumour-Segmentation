import torch
import cv2
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

#Analyse
data_file1 = os.path.join("./MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz")
data = nib.load(data_file1)
print(data.shape)
nd_arr = np.asarray(data)
print(nd_arr.shape)


class GlobalConfig:
    root_dir = '../dataset'
    train_root_dir = '../dataset/MICCAI_BraTS2020_TrainingData'
    test_root_dir = '../dataset/MICCAI_BraTS2020_ValidationData'
    path_to_csv = './train_data.csv'
    pretrained_model_path = ''
    seed = 3333
    
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


config = GlobalConfig()
seed_everything(config.seed)

survival_info_df = pd.read_csv('../dataset/MICCAI_BraTS2020_TrainingData/survival_info.csv')
name_mapping_df = pd.read_csv('../dataset/MICCAI_BraTS2020_TrainingData/name_mapping.csv')


