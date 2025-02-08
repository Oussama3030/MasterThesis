import torch
from torch import nn


class PIDModelv1(nn.Module):
    def __init__(self):
        super(PIDModelv1, self).__init__()
        self.fc1 = nn.Linear(11, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 5)
            
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PIDModelv2(nn.Module):
    def __init__(self):
        super(PIDModelv2, self).__init__()
        
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 5)
        
        # Dropout
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = torch.relu(x)    
        # Layer 2
        x = self.fc2(x)
        x = torch.relu(x)        
        # Layer 3
        x = self.fc3(x)
        x = torch.relu(x)        
        # Output Layer
        x = self.fc4(x)
        
        return x

# Should be equivalent
    
class PIDModelv3(nn.Module):
    def __init__(self):
        super(PIDModelv3, self).__init__()
        self.fc1 = nn.Linear(11, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 5)
            
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PIDModelv4(nn.Module):
    def __init__(self):
        super(PIDModelv4, self).__init__()
        
        self.layer_1 = nn.Linear(11, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, 5) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x


class PIDModelv5(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(PIDModelv5, self).__init__()
        self.layer_1 = nn.Linear(11, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.layer_2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.layer_3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.layer_4 = nn.Linear(32, 1)



    def forward(self, x):
        x = self.layer_1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.layer_2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.layer_3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        x = self.layer_4(x)
        return x


class PIDModelv6(nn.Module):
    def __init__(self, input_channels=11, dropout_rate=0.3):
        super(PIDModelv6, self).__init__()
        
        # 1D Convolutional Layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * input_channels, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(dropout_rate)
        
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        # Reshape input to (batch_size, 1, input_channels)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout4(x)
        
        x = self.fc2(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.dropout5(x)
        
        x = self.fc_out(x)
        return x