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
from sklearn.metrics import precision_recall_curve, average_precision_score


# Models
from MLModel import PIDModelv1 as CurrentModelV1
from MLModel import PIDModelv2 as CurrentModelV2
from MLModel import PIDModelv3 as CurrentModelV3
from MLModel import PIDModelv4 as CurrentModelV4
from MLModel import PIDModelv5 as CurrentModelV5
from MLModel import PIDModelv6 as CurrentModelV6

training_samples = np.load('training_samples.npy')
mapped_targets = np.load('mapped_targets.npy')

# Split the data
training_input, testing_input, training_target, testing_target = train_test_split(training_samples, 
                                                    mapped_targets, 
                                                    test_size=0.3, # 20% test, 80% train
                                                    stratify=mapped_targets, # Ensures similar ratio's in test and training 
                                                    random_state=42) # make the random split reproducible

# # Compute weight

class_weights = compute_class_weight('balanced', classes = np.unique(mapped_targets), y=mapped_targets)
class_weights = 1./np.bincount(training_target)
class_weights = torch.from_numpy(class_weights).type(torch.float)

print(class_weights)

# Calculate the scale pos weight as number of the negative class divided by the number of the positive class

scale_pos_weight =  np.sum(training_target == 0) / np.sum(training_target == 1)
scale_pos_weight = torch.tensor(scale_pos_weight).type(torch.float)

# Turn data into tensors
training_input = torch.from_numpy(training_input).type(torch.float)
training_target = torch.from_numpy(training_target).type(torch.float) # changed to float instead of long

testing_input = torch.from_numpy(testing_input).type(torch.float)
testing_target = torch.from_numpy(testing_target).type(torch.float) # changed to float instead of long

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the datasets and loaders
train_input, train_output = training_input.to(device), training_target.to(device)
test_input, test_output = testing_input.to(device), testing_target.to(device)

dataset_training = TensorDataset(train_input, train_output)
dataset_testing = TensorDataset(test_input, test_output)

loader_training = DataLoader(dataset_training, batch_size=512, shuffle=True)
loader_testing = DataLoader(dataset_testing, batch_size=512, shuffle=True)

# Load in the model
model = CurrentModelV5().to(device)
print(model)

# Define the loss and optimizer
loss_fn = nn.BCEWithLogitsLoss(pos_weight = scale_pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_ite
epochs = 10

# Initialize lists to store history outside the loop
training_history = {
    'labels': [],
    'predictions': [],
    'accuracy': [],
    'loss': [],
    'probs': []
}
testing_history = {
    'labels': [],
    'predictions': [],
    'accuracy': [],
    'loss': [],
    'probs': []
}

# Train the model
for epoch in tqdm(range(epochs)):
    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0

    all_labels_train = []
    all_predictions_train = []
    all_probs_train = []

    n_samples, n_correct, loss_values_train = 0, 0, 0
    for data in loader_training:
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.unsqueeze(1)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(outputs)
        predictions = (outputs > 0.9).float()
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)
        train_loss += loss.item()

        all_probs_train.extend(probs.detach().numpy())
        all_labels_train.extend(labels.cpu().numpy())
        all_predictions_train.extend(predictions.cpu().numpy())

    train_accuracy = train_correct / train_total
    train_loss = train_loss / len(loader_training)


    # # Store training results for this epoch
    # training_history['labels'].append(all_labels_train)
    # training_history['predictions'].append(all_predictions_train)
    # training_history['accuracy'].append(train_accuracy)
    # training_history['loss'].append(train_loss)
    # training_history['probs'].append(all_probs_train)

    # Testing Phase       
    model.eval()

    all_labels_test = []
    all_predictions_test = []

    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for data in loader_testing:
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            labels = labels.unsqueeze(1)
            loss = loss_fn(outputs, labels)

            predictions = (outputs > 0.9).float()
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
            val_loss += loss.item()

            all_labels_test.extend(labels.cpu().numpy())
            all_predictions_test.extend(predictions.cpu().numpy())
        
    val_accuracy = val_correct / val_total
    val_loss = val_loss / len(loader_testing)

    # # Store testing results for this epoch
    # testing_history['labels'].append(all_labels_test)
    # testing_history['predictions'].append(all_predictions_test)
    # testing_history['accuracy'].append(val_accuracy)
    # testing_history['loss'].append(val_loss)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Save the model 
torch.save(model.state_dict(), 'NNModel.pth')

print("Classification Report (Training):\n", classification_report(all_labels_train, all_predictions_train, digits=4))
print("Classification Report (Testing):\n", classification_report(all_labels_test, all_predictions_test, digits=4))
    
cm_train = confusion_matrix(all_labels_train, all_predictions_train)
cm_test = confusion_matrix(all_labels_test, all_predictions_test)

precision, recall, thresholds = precision_recall_curve(all_labels_train, all_probs_train)
average_precision = average_precision_score(all_labels_train, all_probs_train)

plt.figure(figsize=(10,6))
plt.plot(recall, precision, color='blue', lw=2, 
         label=f'Precision-Recall curve (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

prob_train = np.array(all_probs_train)
labels_train = np.array(all_labels_train)

print(prob_train, labels_train)

positive_probs = prob_train[labels_train==1]
negative_probs = prob_train[labels_train==0]
print(positive_probs)

# Plot the predictions distribution for the positive class in red for electrons en blue for background
plt.hist(positive_probs, bins=250, histtype='step', label='Electrons', color='red')
plt.hist(negative_probs, bins=250, histtype='step', label='Background', color='blue')
plt.yscale('log')
plt.grid()
plt.xlabel('Predictions')
plt.ylabel('Counts')
plt.title('Predictions Distribution')
plt.legend()
plt.show()

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





        