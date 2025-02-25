import numpy as np
import matplotlib.pyplot as plt
from DANN import FeatureExtractor, ClassClassifier, DomainClassifier
import torch

# Load the MC data
training_samples_mc = np.load('/Users/oussamabenchikhi/o2workdir/PID/ML/Data/DANN_training_samples_mc.npy')
mapped_targets_mc = np.load('/Users/oussamabenchikhi/o2workdir/PID/ML/Data/DANN_mapped_targets_mc.npy')

# Load the real data
training_samples_real = np.load('/Users/oussamabenchikhi/o2workdir/PID/ML/Data/DANN_training_samples_raw.npy')

# Load the models
device = "cuda" if torch.cuda.is_available() else "cpu"

FeatureExtractor = FeatureExtractor().to(device)
ClassClassifier = ClassClassifier().to(device)
DomainClassifier = DomainClassifier().to(device)

FeatureExtractor.load_state_dict(torch.load('DANN_FeatureExtractor.pth'))
ClassClassifier.load_state_dict(torch.load('DANN_ClassClassifier.pth'))
DomainClassifier.load_state_dict(torch.load('DANN_DomainClassifier.pth'))

def set_model_mode(mode='train', models=None):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()


set_model_mode('eval', [FeatureExtractor, ClassClassifier, DomainClassifier])