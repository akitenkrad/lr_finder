# Learning Rate Finder

```python
>>> import torch.nn as nn
>>> import torch.optim as optim
>>> import torchvision
>>> import torchvision.transforms as transforms
>>> from models import dla34
>>> from find_lr import find_lr, save_figure
>>> transform = transforms.Compose([
...     transforms.ToTensor(),
...     transforms.Normalize((0.5,), (0.5,)),
... ])
>>> train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
>>> train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
>>> model = dla34(num_classes=10, pool_size=1)
>>> criterion = nn.CrossEntropyLoss()
>>> optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
>>> log_lrs, losses = find_lr(model, train_dl, optimizer, criterion, init_value=1e-8, final_value=10.0, beta=0.98)
>>> save_figure(log_lrs, losses)
```