import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 19)

    def forward(self, x):
        x = self.fc1(x)
        x=nn.ReLU(x)
        x = self.fc2(x)
        x=nn.ReLU(x)
        x=self.fc3(x)
        
        #do not need to use softmax or sigmoid because we use cross entropy loss and it does it for us
        
        return x
    