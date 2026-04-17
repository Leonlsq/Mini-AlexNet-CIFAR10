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

num_epochs = 5

print("Start training...")

for epoch in range(num_epochs):
    model.train()
    tot_loss = 0

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # last gradients to 0
        optimizer.zero_grad()

        # forward
        outputs = model(images)

        cur_loss = criterion(outputs, labels)
        cur_loss.backward()

        # perform optimization step with backpropagated gradients
        optimizer.step()

        # accumulate total loss
        tot_loss += cur_loss.item()

        # print the average loss every 100 batches
        if (i+1) % 100 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0


print("Training complete")
