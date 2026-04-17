import torch
import torch.nn as nn
import torch.optim as optim

from model import MiniAlexNet
from data_loader import test_loader, train_loader


def main():

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

        train_correct = 0
        train_total = 0

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

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # print the average loss every 100 batches
            if (i+1) % 100 == 0:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {tot_loss/100:.4f}')
                tot_loss = 0.0

        train_accuracy = 100 * train_correct / train_total
        print(
            f'==> Epoch [{epoch+1}/{num_epochs}] is done, accuracy: {train_accuracy:.2f}%\n')

    print("Training complete")

    print("\nTraining on the test_set")

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # the concrete value is not important, all we care about is which labels it predicted
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Model\'s accuracy on 10000 images from test_set: {accuracy:.2f}%')


if __name__ == '__main__':
    main()
