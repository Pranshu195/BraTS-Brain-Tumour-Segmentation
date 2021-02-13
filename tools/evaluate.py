import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, data_set, criterion):
    dataloader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=1)
    val_loss = 0.0
    for i, data in enumerate(tqdm(dataloader)):
        image= data["image"].to(device)
        mask = data["mask"].to(device)
        out= model(image)
        loss = criterion(out, mask)
        val_loss += loss.item()
    return val_loss/len(dataloader)
