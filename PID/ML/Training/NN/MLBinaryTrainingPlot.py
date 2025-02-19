# Import all the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import uproot
import ROOT
import pandas as pd
from sklearn.neural_network import MLPClassifier
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  
import seaborn as sns
from sklearn import metrics 

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

# Models 

from MLModel import PIDModelv1 as CurrentModelV1
from MLModel import PIDModelv2 as CurrentModelV2
from MLModel import PIDModelv3 as CurrentModelV3
from MLModel import PIDModelv4 as CurrentModelV4
from MLModel import PIDModelv5 as CurrentModelV5

# Import the model NNModel.pth

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CurrentModelV5().to(device)
model.load_state_dict(torch.load('NNBinaryModel.pth'))
print(model)

