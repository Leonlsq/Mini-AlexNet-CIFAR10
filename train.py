import torch
import torch.nn as nn
import torch.optim as optim

from model import MiniAlexNet
from data_loader import test_loader, train_loader


device = torch.device("mps")
# Because I'm locally using Mac(M3 pro) to train this Model
# If you happend to get a GPU to use, just change it to "cuda"

model = MiniAlexNet(num_classes=10).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# emprically I choose 0.001 as Learning Rate
optimizer = optim.Adam(model.parameters(), lr=0.001)
