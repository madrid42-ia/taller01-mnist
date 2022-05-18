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
       self.layers = nn.Sequential(
           nn.Flatten(),
           nn.Linear(28 * 28, 50),
           nn.ReLU(),
           nn.Linear(50, 50),
           nn.ReLU(),
           nn.Linear(50, 10),
           nn.Softmax(dim=1),
       )

   def transforms(self, x):
      return std_transform(x)

   def forward(self, x):
       x = self.layers(x)
       return x
