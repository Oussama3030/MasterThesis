import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  

# Import the models
from DANN import FeatureExtractor, ClassClassifier, DomainClassifier

# Load the MC data
training_samples_mc = np.load('/Users/oussamabenchikhi/o2workdir/PID/ML/Data/DANN_training_samples_mc.npy')
mapped_targets_mc = np.load('/Users/oussamabenchikhi/o2workdir/PID/ML/Data/DANN_mapped_targets_mc.npy')

# Load the real data
training_samples_real = np.load('/Users/oussamabenchikhi/o2workdir/PID/ML/Data/DANN_training_samples_raw.npy')

# Split the MC Data

training_input_mc, testing_input_mc, training_target_mc, testing_target_mc = train_test_split(training_samples_mc,
                                                                                            mapped_targets_mc,
                                                                                            test_size=0.3,
                                                                                            stratify=mapped_targets_mc,
                                                                                            random_state=42)

# Split the real data

training_input_real, testing_input_real = train_test_split(training_samples_real, 
                                                           test_size=0.3, 
                                                           random_state=42)


# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Turn the data into tensors
training_input_mc = torch.from_numpy(training_input_mc).type(torch.float).to(device)
training_target_mc = torch.from_numpy(training_target_mc).type(torch.long).to(device)

testing_input_mc = torch.from_numpy(testing_input_mc).type(torch.float).to(device)
testing_target_mc = torch.from_numpy(testing_target_mc).type(torch.long).to(device)

# Create a DataLoader for the MC data
train_dataset_mc = TensorDataset(training_input_mc, training_target_mc)
test_dataset_mc = TensorDataset(testing_input_mc, testing_target_mc)

train_loader_mc = DataLoader(train_dataset_mc, batch_size=32, shuffle=True)
test_loader_mc = DataLoader(test_dataset_mc, batch_size=32, shuffle=True)

# Create a DataLoader for the real data
training_input_real = torch.from_numpy(training_input_real).type(torch.float).to(device)

train_dataset_real = TensorDataset(training_input_real)

train_loader_real = DataLoader(train_dataset_real, batch_size=32, shuffle=True)

# Create the models

FeatureExtractor = FeatureExtractor().to(device)
ClassClassifier = ClassClassifier().to(device)
DomainClassifier = DomainClassifier().to(device)    

# Create the optimizers

optimizer = optim.Adam([
    {"params": FeatureExtractor.parameters()},
    {"params": ClassClassifier.parameters()},
    {"params": DomainClassifier.parameters()}
], lr=0.001)

classifier_criterion = nn.CrossEntropyLoss().cuda()
discriminator_criterion = nn.CrossEntropyLoss().cuda()

# optimizer = optim.Adam()

# Definition

def set_model_mode(mode='train', models=None):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()


epochs = 5
lambda_param = 0.1

# Set up metrics tracking
train_class_losses = []
train_domain_losses = []
test_accuracies = []

for epoch in tqdm(range(epochs), desc="Training ..."):
    set_model_mode('train', [FeatureExtractor, ClassClassifier, DomainClassifier])

    # Track losses per epoch
    epoch_class_loss = 0.0
    epoch_domain_loss = 0.0
    num_batches = 0

    real_iter = iter(train_loader_real)

    # Loop through MC data batches
    for batch_idx, (mc_data, mc_labels) in enumerate(train_loader_mc):
        try:
            real_data = next(real_iter)[0]
        except StopIteration:
            real_iter = iter(train_loader_real)
            real_data = next(real_iter)[0]

        mc_data, mc_labels, real_data = mc_data.to(device), mc_labels.to(device), real_data.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Create the labels for the domain classifier
        mc_labels_domain = torch.zeros(mc_data.size(0)).type(torch.long).to(device)
        real_labels_domain = torch.ones(real_data.size(0)).type(torch.long).to(device)

        # Forward pass for MC data
        mc_features = FeatureExtractor(mc_data)
        mc_preds = ClassClassifier(mc_features)
        mc_domain_output = DomainClassifier(mc_features)

        # Forward pass for real data
        real_features = FeatureExtractor(real_data)
        real_domain_output = DomainClassifier(real_features)

        # Calculate the losses

        # 1. Classification loss (only MC)
        class_loss = classifier_criterion(mc_preds, mc_labels)

        # 2. Domain loss
        mc_domain_loss = discriminator_criterion(mc_domain_output, mc_labels_domain)
        real_domain_loss = discriminator_criterion(real_domain_output, real_labels_domain)
        domain_loss = mc_domain_loss + real_domain_loss

        # Total loss: classification loss + lambda * domain loss
        total_loss = class_loss + lambda_param * domain_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Accumulate losses
        epoch_class_loss += class_loss.item()
        epoch_domain_loss += domain_loss.item()
        num_batches += 1

    # Calculate the average loss per epoch
    avg_class_loss = epoch_class_loss / num_batches
    avg_domain_loss = epoch_domain_loss / num_batches
    train_class_losses.append(avg_class_loss)
    train_domain_losses.append(avg_domain_loss)


    # Evaluate the model on the test set
    if epoch % 5 == 0 or epoch == epoch - 1:
        set_model_mode('eval', [FeatureExtractor, ClassClassifier, DomainClassifier])
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader_mc:
                features = FeatureExtractor(data)
                outputs = ClassClassifier(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        print(f"Epoch {epoch}/{epochs} - Classification Loss: {avg_class_loss:.4f}, Domain Loss: {avg_domain_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

# Save the models 

torch.save(FeatureExtractor.state_dict(), 'DANN_FeatureExtractor.pth')
torch.save(ClassClassifier.state_dict(), 'DANN_ClassClassifier.pth')
torch.save(DomainClassifier.state_dict(), 'DANN_DomainClassifier.pth')