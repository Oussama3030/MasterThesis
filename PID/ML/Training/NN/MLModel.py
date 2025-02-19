import torch
from torch import nn


class PIDModelv1(nn.Module):
    def __init__(self):
        super(PIDModelv1, self).__init__()
        self.fc1 = nn.Linear(3, 256)
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
        
        # Layer 1
        self.fc1 = nn.Linear(3, 256)
        # self.batchnorm1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.2)
        
        # Layer 2
        self.fc2 = nn.Linear(256, 512)
        # self.batchnorm2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=0.2)
        
        # Layer 3
        self.fc3 = nn.Linear(512, 256)
        # self.batchnorm3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(p=0.2)
        
        # Layer 4
        self.fc4 = nn.Linear(256, 126)
        # self.batchnorm4 = nn.BatchNorm1d(126)
        self.dropout4 = nn.Dropout(p=0.2)
        
        # Output Layer
        self.fc5 = nn.Linear(126, 5)
        
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        # x = self.batchnorm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        # x = self.batchnorm2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        # x = self.batchnorm3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        # Layer 4
        x = self.fc4(x)
        # x = self.batchnorm4(x)
        x = torch.relu(x)
        x = self.dropout4(x)
        
        # Output Layer (no dropout or batch norm)
        x = self.fc5(x)
        
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
        
        self.layer_1 = nn.Linear(3, 512)
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
        self.layer_1 = nn.Linear(3, 128)
        # self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.layer_2 = nn.Linear(128, 64)
        # self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.layer_3 = nn.Linear(64, 32)
        # self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.layer_4 = nn.Linear(32, 1)



    def forward(self, x):
        x = self.layer_1(x)
        # x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.layer_2(x)
        # x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.layer_3(x)
        # x = self.bn3(x)
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
    

class DANN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=5):
        super(DANN, self).__init__()
        
        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Domain Classifier which is a binary classifier with gradient reverse layer
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, alpha=0.1):
        # Feature Extractor
        x = self.feature_extractor(x)
        
        # Classifier
        class_output = self.classifier(x)
        
        # Domain Classifier
        domain_output = self.discriminator(x)
        
        return class_output, domain_output

