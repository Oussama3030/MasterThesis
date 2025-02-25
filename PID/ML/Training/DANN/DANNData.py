import numpy as np
import pandas as pd
import uproot
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch
import matplotlib.pyplot as plt
import os

# MC Data preparation 
file = uproot.open('/Users/oussamabenchikhi/o2workdir/PID/myTrees.root')

keys = [key for key in file.keys() if key.endswith("/TpcData;1")]#[:50]

branch1_data = []  # fSignal
branch2_data = []  # fPt
branch3_data = []  # fPdgID
branch4_data = []  # fTofBeta

for key in keys:
    tree = file[key]

    branch1_data.append(tree["fSignal"].array(library="np"))
    branch2_data.append(tree["fPt"].array(library="np"))
    branch3_data.append(tree["fPdgID"].array(library="np"))
    branch4_data.append(tree["fTofBeta"].array(library="np"))


branch1_data = np.concatenate(branch1_data)
branch2_data = np.concatenate(branch2_data)
branch3_data = np.concatenate(branch3_data)
branch4_data = np.concatenate(branch4_data)

training_samples_mc = np.vstack([branch1_data, branch2_data, branch4_data]).T
target_samples_mc = abs(branch3_data)

# Create a mapping of PDG codes to consecutive integers
unique_labels = np.unique(target_samples_mc)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

mapped_targets_mc = np.array([1 if label == 11 else 0 for label in target_samples_mc])

# Create a dataframe

data_frame_mc = pd.DataFrame({"dE/dx": training_samples_mc[:, 0],
    "pT": training_samples_mc[:, 1],
    "tofBeta": training_samples_mc[:, 2],
    "label": mapped_targets_mc
})

print(data_frame_mc)

# plt.scatter(data_frame_mc["pT"], data_frame_mc["dE/dx"], color='grey', label='Everything', s=0.5)
# plt.grid()
# plt.gcf().set_size_inches(15, 10)
# plt.xlabel('p')
# plt.ylim(-0.5, 0.5)
# plt.xlim(0, 3)
# plt.legend()
# plt.show()

save_dir = "../../Data"

# Save the arrays with full paths
np.save(os.path.join(save_dir, 'DANN_training_samples_mc.npy'), training_samples_mc)
np.save(os.path.join(save_dir, 'DANN_mapped_targets_mc.npy'), mapped_targets_mc)

# Print all unique labels and the amount of samples for each label
print(data_frame_mc.label.value_counts())

# Raw Data Preparation

file = uproot.open('/Users/oussamabenchikhi/o2workdir/PID/myTreesRAW.root')

keys = [key for key in file.keys() if key.endswith("/TpcData;1")]#[:50]

print(keys)

branch1_data = []  # fSignal
branch2_data = []  # fPt
branch3_data = []  # fTofBeta

for key in keys:
    tree = file[key]

    branch1_data.append(tree["fSignal"].array(library="np"))
    branch2_data.append(tree["fPt"].array(library="np"))
    branch3_data.append(tree["fTofBeta"].array(library="np"))

branch1_data = np.concatenate(branch1_data)
branch2_data = np.concatenate(branch2_data)
branch3_data = np.concatenate(branch3_data)

training_samples_raw = np.vstack([branch1_data, branch2_data, branch3_data]).T

# Create a dataframe

data_frame_raw = pd.DataFrame({"dE/dx": training_samples_raw[:, 0],
    "pT": training_samples_raw[:, 1],
    "tofBeta": training_samples_raw[:, 2]
})

print(data_frame_raw)

save_dir = "../../Data"

# Save the arrays with full paths
np.save(os.path.join(save_dir, 'DANN_training_samples_raw.npy'), training_samples_raw)

# # Make a scatter plot of the raw data p vs dE/dx
# plt.scatter(data_frame_raw["pT"], data_frame_raw["dE/dx"], color='grey', label='Everything', s=0.5)
# plt.grid()
# plt.gcf().set_size_inches(15, 10)
# plt.xlabel('p')
# plt.ylim(-0.5, 0.5)
# plt.xlim(0, 3)
# plt.legend()
# plt.show()
