import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# Define a gradient reversal layer
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

# Feature extractor network
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.layer_1 = nn.Linear(3, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        return x
    
# Particle classifier network
class ClassClassifier(nn.Module):
    def __init__(self):
        super(ClassClassifier, self).__init__()
        self.layer_1 = nn.Linear(32, 16)
        self.layer_2 = nn.Linear(16, 8)
        self.layer_out = nn.Linear(8, 3)
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_out(x)
        return x

# Domain classifier network
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer_1 = nn.Linear(32, 16)
        self.layer_2 = nn.Linear(16, 8)
        self.layer_out = nn.Linear(8, 2)
        
    def forward(self, x, reverse=True):
        x = grad_reverse(x) if reverse else x
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.softmax(self.layer_out(x), dim=1)
        return x

# Combining the networks   
class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__inito

# Print the model
print(DANN())