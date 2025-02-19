# Import all the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import uproot
import ROOT
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy.ma as ma

# Open the root file and access the TTree
file = uproot.open('/Users/oussamabenchikhi/o2workdir/PID/myTrees.root')

keys = [key for key in file.keys() if key.endswith("/TpcData;1")]#[:50]

branch1_data = []  # fSignal
branch2_data = []  # fPt
branch3_data = []  # fPdgID
branch4_data = []  # fTofBeta
branch5_data = []  # ftofNSigmaEl
branch6_data = []  # ftpcNSigmaEl
branch7_data = []  # ftofNSigmaPr
branch8_data = []  # ftpcNSigmaPr
branch9_data = []  # ftofNSigmaPi
branch10_data = []  # ftpcNSigmaPi
branch11_data = []  # ftofNSigmaKa
branch12_data = []  # ftpcNSigmaKa

for key in keys:
    tree = file[key]

    branch1_data.append(tree["fSignal"].array(library="np"))
    branch2_data.append(tree["fPt"].array(library="np"))
    branch3_data.append(tree["fPdgID"].array(library="np"))
    branch4_data.append(tree["fTofBeta"].array(library="np"))
    branch5_data.append(tree["ftpcNSigmaEl"].array(library="np"))
    branch6_data.append(tree["ftofNSigmaEl"].array(library="np"))
    branch7_data.append(tree["ftofNSigmaPr"].array(library="np"))
    branch8_data.append(tree["ftpcNSigmaPr"].array(library="np"))
    branch9_data.append(tree["ftofNSigmaPi"].array(library="np"))
    branch10_data.append(tree["ftpcNSigmaPi"].array(library="np"))
    branch11_data.append(tree["ftofNSigmaKa"].array(library="np"))
    branch12_data.append(tree["ftpcNSigmaKa"].array(library="np"))


branch1_data = np.concatenate(branch1_data)
branch2_data = np.concatenate(branch2_data)
branch3_data = np.concatenate(branch3_data)
branch4_data = np.concatenate(branch4_data)
branch5_data = np.concatenate(branch5_data)
branch6_data = np.concatenate(branch6_data)
branch7_data = np.concatenate(branch7_data)
branch8_data = np.concatenate(branch8_data)
branch9_data = np.concatenate(branch9_data)
branch10_data = np.concatenate(branch10_data)
branch11_data = np.concatenate(branch11_data)
branch12_data = np.concatenate(branch12_data)

# branch2_data = np.log1p(branch2_data)

training_samples_unmasked = np.vstack([branch1_data, branch2_data, branch4_data]).T #, branch5_data, branch6_data, branch7_data, branch8_data, branch9_data, branch10_data, branch11_data, branch12_data]).T
target_samples_unmasked = abs(branch3_data)

keep = [11, 13, 211, 321, 2212]

mask = np.isin(target_samples_unmasked, keep)
training_samples = training_samples_unmasked[mask]
target_samples = target_samples_unmasked[mask]

print(training_samples.shape)
print(target_samples.shape)
#print(training_samples)

data_frame = pd.DataFrame({"dE/dx": training_samples[:, 0],
    "pT": training_samples[:, 1],
    "label": target_samples
})

# Create a mapping of PDG codes to consecutive integers
unique_labels = np.unique(target_samples)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

# Convert PDG codes to consecutive integers
mapped_targets = np.array([label_to_idx[label] for label in target_samples])

# mapped_targets = np.array([1 if label == 11 else 0 for label in target_samples])


# Create DataFrame with both original and mapped labels
data_frame = pd.DataFrame({
    "dE/dx": training_samples[:, 0],
    "pT": training_samples[:, 1],
    "tofBeta": training_samples[:, 2],
    "original_label": target_samples,
    "mapped_label": mapped_targets
})

    # "tpcNSigmaEl": training_samples[:, 3],
    # "tofNSigmaEl": training_samples[:, 4], 
    # "tofNSigmaPr": training_samples[:, 5],
    # "tpcNSigmaPr": training_samples[:, 6],
    # "tofNSigmaPi": training_samples[:, 7],
    # "tpcNSigmaPi": training_samples[:, 8],
    # "tofNSigmaKa": training_samples[:, 9],
    # "tpcNSigmaKa": training_samples[:, 10],

# Print the mapping and some sample data
print("Label mapping:", label_to_idx)
print("\nSample data:")
print(data_frame.head(20))
print("\nLabel distribution:")
print(data_frame.mapped_label.value_counts())
print("\nPDG distribution:")
print(data_frame.original_label.value_counts())

import os

save_dir = "../../Data"

# Save the arrays with full paths
np.save(os.path.join(save_dir, 'training_samples.npy'), training_samples)
np.save(os.path.join(save_dir, 'mapped_targets.npy'), mapped_targets)
np.save(os.path.join(save_dir, 'target_samples.npy'), target_samples)


# np.save({
#     'training_samples': training_samples,
#     'mapped_targets': mapped_targets,
#     'original_targets': target_samples,
# }, 'data.txt')

# #print(filtered_branch1)
# print(len(filtered_branch2), len(filtered_branch1))

# Creating the histogram

# # Histogram definition
# bins = [5000, 5000]  # Number of bins for X (fPt) and Y (fSignal)

# # Histogram the data
# hh, locx, locy = np.histogram2d(branch2_data, branch1_data, bins=bins)

# # Sort the points by density, so that the densest points are plotted last
# z = np.array([hh[np.argmax(a <= locx[1:]), np.argmax(b <= locy[1:])] for a, b in zip(branch2_data, branch1_data)])
# idx = z.argsort()
# x_sorted, y_sorted, z_sorted = branch2_data[idx], branch1_data[idx], z[idx]

# plt.figure(figsize=(12, 8))
# scatter = plt.scatter(x_sorted, y_sorted, cmap='jet', marker='.', s=0.005)
# # plt.colorbar(scatter, label='Density')
# plt.xlabel('P')
# plt.xscale("log")
# plt.ylabel('dE/dx')
# plt.title('Scatter Plot of dE/dx vs P')
# plt.xlim(0, 20)
# plt.ylim(0, 500)
  
# # plt.show()
# import seaborn as sns
# # seaborn.set(style='ticks')

# sns.relplot(data=data_frame, x = "pT", y = "dE/dx", hue = 'original_label', kind = 'scatter' , legend = 'auto')
# plt.ylim(0, 500)
# plt.xlim(0,20)
# plt.show()