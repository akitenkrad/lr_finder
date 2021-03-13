import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import click
from sklearn.metrics import accuracy_score

from models import dla34

@click.command()
@click.option('--epochs', type=int, default=10, help='epochs')
def run(epochs):
    # data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)
    
    # labels
    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
    
    # model
    model = dla34(num_classes=10, pool_size=1)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    with tqdm(desc='', total=epochs, leave=True) as epoch_it:
        for epoch in range(epochs):
            
            with tqdm(desc='', total=len(train_ds), leave=False) as train_it:
                batch_loss = 0.0
                for i, (x, labels) in enumerate(train_dl):
                    optimizer.zero_grad()

                    out = model(x)
                    loss = criterion(out, labels)
                    loss.backward()
                    optimizer.step()

                    batch_loss += loss.item()

                    train_it.set_description(
                        '[E:{:04d} B:{:05d}] loss:{:.3f}'.format(epoch+1, i+1, batch_loss / (i+1))
                    )
                    train_it.update(1)

            with tqdm(desc='', total=len(test_ds), leave=False) as test_it:
                pred, gt = [], []
                for x, labels in test_dl:
                    out = model(x)
                    
                    pred = np.concatenate([pred, out.argmax(dim=1).numpy()])
                    gt = np.concatenate([gt, labels.numpy()])
                
                accuracy = accuracy_score(gt, pred)
                
            epoch_it.set_description(
                '[E:{:04d}] loss:{:.3f} acc:{:.3f}'.format(epoch+1, batch_loss / len(train_ds), accuracy)
            )
            epoch_it.update(1)
            
    print('Finished.')

if __name__ == '__main__':
    run()
    