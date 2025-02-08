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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

# Models
from MLModel import PIDModelv1 as CurrentModelV1
from MLModel import PIDModelv2 as CurrentModelV2

training_samples = np.load('training_samples.npy')
mapped_targets = np.load('mapped_targets.npy')

# Split the data
training_input, testing_input, training_target, testing_target = train_test_split(training_samples, 
                                                    mapped_targets, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    stratify=mapped_targets, # Ensures similar ratio's in test and training 
                                                    random_state=42) # make the random split reproducible

# # Compute weight

# class_weights = compute_class_weight('balanced', classes = np.unique(mapped_targets), y=mapped_targets)
# class_weights = 1. / np.bincount(training_target)
# class_weights_normalized = 1. / np.sum(class_weights)
# class_weights_normalized = torch.from_numpy(class_weights).type(torch.float)

# print(class_weights)

# # Compute weights
# class_counts = np.bincount(training_target)
# num_classes = len(class_counts)
# total_samples = len(training_target)

# class_weights = []
# for count in class_counts:
#     weight = 1 / (count / total_samples)
#     class_weights.append(weight)

# class_weights = np.array(class_weights)
# class_weights = torch.from_numpy(class_weights).type(torch.float)

# unique_classes = np.unique(mapped_targets)
# for class_label, weight in zip(unique_classes, class_weights):
#     print(f"Class {class_label}: Weight {weight}")


# #print(class_weights)

# Turn data into tensors
training_input = torch.from_numpy(training_input).type(torch.float)
training_target = torch.from_numpy(training_target).type(torch.long)

testing_input = torch.from_numpy(testing_input).type(torch.float)
testing_target = torch.from_numpy(testing_target).type(torch.long)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the datasets and loaders
train_input, train_output = training_input.to(device), training_target.to(device)
test_input, test_output = testing_input.to(device), testing_target.to(device)

dataset_training = TensorDataset(train_input, train_output)
dataset_testing = TensorDataset(test_input, test_output)

# # Calculate class weights
# class_counts = np.bincount(mapped_targets)
# weights = 1.0 / class_counts
# sample_weights = weights[mapped_targets]
# sample_weights = torch.from_numpy(sample_weights)
# sampler = torch.utils.data.WeightedRandomSampler(
#     weights=sample_weights,
#     num_samples=len(sample_weights),
#     replacement=True
# )

# # Use sampler in DataLoader
# loader_training = DataLoader(dataset_training, batch_size=256, sampler=sampler)

loader_training = DataLoader(dataset_training, batch_size=256, shuffle=True)
loader_testing = DataLoader(dataset_testing, batch_size=256, shuffle=True)

# Load in the model
model = CurrentModelV1().to(device)
print(model)

# Define the loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)

# Set number of epochs
epochs = 25

for epoch in tqdm(range(epochs), desc="Training ..."):
    ### Training Phase
    model.train()
    correct_predictions, total_samples = 0, 0
    epoch_loss = 0

    all_labels_train = []
    all_predictions_train = []

    for data in loader_training:
        inputs, labels = data[0].to(device), data[1].to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Store predictions and labels
        all_labels_train.extend(labels.cpu().numpy())
        all_predictions_train.extend(predicted.cpu().numpy())
    
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]


    epoch_loss_avg = epoch_loss / len(loader_training)
    epoch_acc = correct_predictions / total_samples

    ### Validation Phase
    model.eval()
    correct_predictions_test, total_samples_test = 0, 0
    epoch_loss_test = 0

    all_labels_test = []
    all_predictions_test = []

    with torch.no_grad():
        for data in loader_testing:
            inputs, labels = data[0].to(device), data[1].to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Accumulate metrics
            epoch_loss_test += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions_test += (predicted == labels).sum().item()
            total_samples_test += labels.size(0)

            all_labels_test.extend(labels.cpu().numpy())
            all_predictions_test.extend(predicted.cpu().numpy())


    epoch_loss_avg_test = epoch_loss_test / len(loader_testing)
    epoch_acc_test = correct_predictions_test / total_samples_test

    # Logging metrics
    print(f"Epoch: {epoch} | Loss: {epoch_loss_avg:.5f}, Acc: {epoch_acc:.6f} | "
            f"Test Loss: {epoch_loss_avg_test:.5f}, Test Acc: {epoch_acc_test:.6f} | "
            f"Current LR: {optimizer.param_groups[0]['lr']}")
        
    if epoch == epochs - 1:
        print("Classification Report (Training):\n", classification_report(all_labels_train, all_predictions_train))
        print("Classification Report (Testing):\n", classification_report(all_labels_test, all_predictions_test))

        # Create confusion matrices
        cm_train = confusion_matrix(all_labels_train, all_predictions_train)
        cm_test = confusion_matrix(all_labels_test, all_predictions_test)

        # Plot confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training confusion matrix
        sns.heatmap(cm_train, annot=True, fmt='d', ax=ax1, cmap='Blues')
        ax1.set_title('Training Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # Testing confusion matrix
        sns.heatmap(cm_test, annot=True, fmt='d', ax=ax2, cmap='Blues')
        ax2.set_title('Testing Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        
        plt.tight_layout()
        plt.show()
        