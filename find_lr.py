import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from models import dla34

def find_lr(model:nn.Module, dataloader:DataLoader, optimizer:optim.Optimizer, criterion, 
            init_value=1e-8, final_value=10.0, beta=0.98):
    num = len(dataloader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.0
    best_loss = 0.0
    losses = []
    log_lrs = []
    
    with tqdm(total=len(dataloader), desc='') as it:
        for idx, (x, labels) in enumerate(dataloader):
            # get output
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, labels)

            # compute the smoothed loss
            avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**(idx+1))

            # stop if the loss is exploding
            if idx > 0 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses

            # record the best loss
            if smoothed_loss < best_loss or idx == 0:
                best_loss = smoothed_loss

            # store the values
            losses.append(smoothed_loss)
            log_lrs.append(lr)

            loss.backward()
            optimizer.step()

            # update progress bar
            it.set_description('[B:{:05d}] lr:{:.8f} best_loss:{:.3f}'.format(idx+1, lr, best_loss))
            it.update(1)
            
            # update learning rate
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr
        
        return log_lrs, losses

def save_figure(log_lrs, losses, save_file='lr_loss_curve.png'):
        # save figure    
        plt.plot(log_lrs[10:-5], losses[10:-5])
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.savefig(save_file)
        print('saved ->', save_file)

def run_finder():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    
    model = dla34(num_classes=10, pool_size=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
 
    log_lrs, losses = find_lr(model, train_dl, optimizer, criterion, init_value=1e-8, final_value=10.0, beta=0.98)
    
    save_figure(log_lrs, losses)
       
    return log_lrs, losses
