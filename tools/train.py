import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from dataset.data import BratsDataset
from model.btsmodel import BraTS_Model
from loss.loss import BCEDiceLoss
from tools.evaluate import evaluate
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
import argparse
import numpy as np
import random
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    train_data = BratsDataset(path_to_csv = "./train_data.csv", phase='train')
    val_data = BratsDataset(path_to_csv= './train_data.csv', phase='train')
    train_loader = DataLoader(train_data, drop_last=True, batch_size=args.batch_size, num_workers = 4, pin_memory=True, shuffle=True)
    
    model = BraTS_Model(in_channels=4, n_classes=3, n_channels=24).to(device)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    epochs = args.num_epochs

    best_val_loss = None
    best_val_acc = None

    with torch.no_grad():
        best_val_acc, best_val_loss = evaluate(model, val_data, criterion)
    
    print("Best val accuracy = {}, val loss = {}".format(best_val_acc, best_val_loss))
    exit(0)
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            images = data['image'].to(device)
            mask = data['mask'].to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, mask)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if i % 4 == 0:
                print("Epoch: {}/{}, Step: {}/{}, Loss: {:.4f}"
                .format(epoch+1, epochs, i+1, len(train_loader), loss.item()))
        with torch.no_grad():
            val_acc, val_loss = evaluate(model, val_data, criterion)
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "./tools/best_epoch_{}_brats_model_1e-4_16_adam.pth".format(epoch+1))
        
        print('Best Validation Loss : {}'.format(best_val_loss))
        print('Best Validation Accuracy : {}'.format(best_val_acc))
        print('Best Epoch: {}'.format(best_epoch + 1))
        print('Epoch {}/{} Done | Train Loss : {:.4f} | Validation Loss : {:.4f} | Validation Accuracy : {:.4f}'
              .format(epoch + 1, 30, train_loss / len(train_loader), val_loss, validation_acc))
    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_step', type=int, default=2000)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1E-4)
    parser.add_argument('--freeze_layers', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1111)

    
    args = parser.parse_args()
    print(args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    best_val_loss = train(args=args)
    print(best_val_loss)