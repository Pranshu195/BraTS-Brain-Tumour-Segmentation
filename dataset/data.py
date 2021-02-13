import torch
import cv2
import os
import nibabel as nib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from albumentations import Compose, HorizontalFlip
from skimage.util import montage
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

class BratsDataset(Dataset):
    def __init__(self, path_to_csv, phase: str="test", is_resize: bool=False, fold: int=0):
        df = pd.read_csv(path_to_csv)
        # print(df['fold'])
        train_df = df.loc[df['fold'] != fold].reset_index(drop=True)
        val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

        df = train_df if phase == "train" else val_df

        self.df = df
        self.phase = phase
        self.augmentations = self.get_augmentations(phase)
        self.data_types = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']
        self.is_resize = is_resize
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
        # load all modalities
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)#.transpose(2, 0, 1)
            
            if self.is_resize:
                img = self.resize(img)
    
            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        
        if self.phase != "test":
            mask_path =  os.path.join(root_path, id_ + "_seg.nii.gz")
            mask = self.load_img(mask_path)
            
            if self.is_resize:
                mask = self.resize(mask)
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                mask = np.clip(mask, 0, 1)
            mask = self.preprocess_mask_labels(mask)
    
            augmented = self.augmentations(image=img.astype(np.float32), 
                                           mask=mask.astype(np.float32))
            
            img = augmented['image']
            mask = augmented['mask']
    
        
            return {
                "Id": id_,
                "image": img,
                "mask": mask,
            }
        
        return {
            "Id": id_,
            "image": img,
        }
    
    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data
    
    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)
    
    def resize(self, data: np.ndarray):
        data = resize(data, (78, 120, 120), preserve_range=True)
        return data
    
    def preprocess_mask_labels(self, mask: np.ndarray):

        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask
    
    def get_augmentations(self, phase):
        list_transforms = []
        
        list_trfms = Compose(list_transforms)
        return list_trfms



if __name__ == "__main__":
    #Analyse
    # data_file1 = os.path.join("./dataset/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz")
    # data = nib.load(data_file1)
    # print(data.shape)
    # nd_arr = np.asarray(data)
    # print(nd_arr.shape)


    # class GlobalConfig:
    #     root_dir = './dataset'
    #     train_root_dir = './dataset/MICCAI_BraTS2020_TrainingData'
    #     test_root_dir = './dataset/MICCAI_BraTS2020_ValidationData'
    #     path_to_csv = './train_data.csv'
    #     pretrained_model_path = ''
    #     seed = 3333
        
    # def seed_everything(seed: int):
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(seed)


    # config = GlobalConfig()
    # seed_everything(config.seed)

    # survival_info_df = pd.read_csv('./dataset/MICCAI_BraTS2020_TrainingData/survival_info.csv')
    # name_mapping_df = pd.read_csv('./dataset/MICCAI_BraTS2020_TrainingData/name_mapping.csv')



    # name_mapping_df.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True) 


    # df = survival_info_df.merge(name_mapping_df, on="Brats20ID", how="right")

    # paths = []
    # for _, row  in df.iterrows():
        
    #     id_ = row['Brats20ID']
    #     phase = id_.split("_")[-2]
        
    #     if phase == 'Training':
    #         path = os.path.join(config.train_root_dir, id_)
    #     else:
    #         path = os.path.join(config.test_root_dir, id_)
    #     paths.append(path)
        
    # df['path'] = paths

    #split data on train, test, split
    #train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=69, shuffle=True)
    #train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    # train_data = df.loc[df['Age'].notnull()].reset_index(drop=True)
    # train_data["Age_rank"] =  train_data["Age"] // 10 * 10
    # train_data = train_data.loc[train_data['Brats20ID'] != 'BraTS20_Training_355'].reset_index(drop=True, )

    # skf = StratifiedKFold(
    #     n_splits=7, random_state=config.seed, shuffle=True
    # )
    # for i, (train_index, val_index) in enumerate(skf.split(train_data, train_data["Age_rank"])):
    #         train_data.loc[val_index, "fold"] = i

    # train_df = train_data.loc[train_data['fold'] != 0].reset_index(drop=True)
    # val_df = train_data.loc[train_data['fold'] == 0].reset_index(drop=True)

    # test_df = df.loc[~df['Age'].notnull()].reset_index(drop=True)
    # print("train_df ->", train_df.shape, "val_df ->", val_df.shape, "test_df ->", test_df.shape)
    # train_data.to_csv("train_data.csv", index=False)

    # print(train_df.head)

    def get_dataloader(
        dataset: torch.utils.data.Dataset,
        path_to_csv: str,
        phase: str,
        fold: int = 0,
        batch_size: int = 1,
        num_workers: int = 4
        ):
        
        '''Returns: dataloader for the model training'''
        df = pd.read_csv(path_to_csv)
        
        train_df = df.loc[df['fold'] != fold].reset_index(drop=True)
        val_df = df.loc[df['fold'] == fold].reset_index(drop=True)

        df = train_df if phase == "train" else val_df
        dataset = dataset(df, phase)
        # dataset = BratsDataset(train_df, phase)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,   
        )
        return dataloader

    dataloader = get_dataloader(dataset=BratsDataset, path_to_csv='./train_data.csv', phase='train')
    print(len(dataloader))
    data = next(iter(dataloader))
    print(data['Id'], data['image'].shape, data['mask'].shape)

    img_numpy = data['image'].squeeze()[0].cpu().detach().numpy() 
    mask_numpy = data['mask'].squeeze()[0].squeeze().cpu().detach().numpy()
    print("Num uniq Image values :", len(np.unique(img_numpy, return_counts=True)[0]))
    print("Min/Max Image values:", img_numpy.min(), img_numpy.max())
    print("Num uniq Mask values:", np.unique(mask_numpy, return_counts=True))

    # image = np.rot90(montage(img_numpy))
    # mask = np.rot90(montage(mask_numpy))
    print(img_numpy.shape)
    cv2.imshow("image", img_numpy[100, :, :])
    
    # plt.imshow(img_numpy[100, :, :], cmap='bone')
    # plt.show()
    cv2.imshow("mask", np.ma.masked_where(mask_numpy[100, :, :]==False, mask_numpy[100, :, :]))
    cv2.waitKey(0)
    # exit(0)
    # cv2.imshow("image", img_numpy[10, :, :])
    # cv2.imshow("mask", mask_numpy[10, :, :])
    # plt.imshow(image, cmap ='bone')
    # plt.plot()
    # fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    # ax.imshow(np.ma.masked_where(mask == False, mask),cmap='cool', alpha=0.6)
    # plt.plot()

    # for i, data in enumerate(dataloader):
    #     print(data['Id'])
    #     print(data['image'].shape)
    #     print(data['mask'].shape)
    #     img_tensor = data['image'].squeeze()[0].cpu().detach().numpy() 
    #     mask_tensor = data['mask'].squeeze()[0].squeeze().cpu().detach().numpy()
    #     print("Num uniq Image values :", len(np.unique(img_tensor, return_counts=True)[0]))
    #     print("Min/Max Image values:", img_tensor.min(), img_tensor.max())
    #     print("Num uniq Mask values:", np.unique(mask_tensor, return_counts=True))

    #     image = np.rot90(montage(img_tensor))
    #     plt.imshow(image)
    #     plt.plot()
    #     mask = np.rot90(montage(mask_tensor)) 

    #     fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    #     ax.imshow(image, cmap ='bone')
    #     ax.imshow(np.ma.masked_where(mask == False, mask),
    #             cmap='cool', alpha=0.6)
    #     plt.plot()
    #     # plt.plot()
    #     # print(images.shape)
    #     exit(0)