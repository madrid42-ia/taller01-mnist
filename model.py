import torch
from torch import nn
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

std_transform = Compose([
      ToTensor(),
      Normalize((0.1307), (0.3081)), # mean and standard deviation
])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(2, 4, 3, 1, 1),
            nn.ReLU(),
        )
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1),
        )
        self.org = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1),
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.lin(x)
        return x

    def transforms(self, x):
        return std_transform(x)
