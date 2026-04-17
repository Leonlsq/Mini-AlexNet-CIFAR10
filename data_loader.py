import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([

    transforms.ToTensor(),
    # [0,1] -> [-1,1] (mean:0.5, std:0.5)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=64,
    shuffle=False,
    num_workers=2
)
