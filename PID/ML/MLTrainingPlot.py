import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn import metrics
import plotly.graph_objects as go
import torch
from MLModel import PIDModelv1  # Import your model architecture
import torch.nn.functional as F

# Load and prepare data
training_samples = np.load('training_samples.npy')
mapped_targets = np.load('mapped_targets.npy')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PIDModelv1().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Split the data
training_input, testing_input, training_target, testing_target = train_test_split(training_samples, 
                                                    mapped_targets, 
                                                    test_size=0.3, # 20% test, 80% train
                                                    stratify=mapped_targets, # Ensures similar ratio's in test and training 
                                                    random_state=42) # make the random split reproducible

# Get predictions from the model 
test_input = torch.from_numpy(testing_input).type(torch.float)
test_target = torch.from_numpy(testing_target).type(torch.long)

# Get the predictions
with torch.no_grad():
    output = model(test_input)
    probabilities = torch.softmax(output, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

# Convert to numpy
all_test_labels = test_target.numpy()
all_test_predictions = predictions.numpy()
all_test_probabilities = probabilities.numpy()

# all_test_labels = np.load('all_test_labels.npy')
# all_test_predictions = np.load('all_test_predictions.npy')
# all_test_probabilities = np.load('all_test_probabilities.npy')

# print(all_test_labels)
# print(all_test_predictions)
# print(all_test_probabilities)

# Create DataFrame
features = ["dE/dx", "P", "tofBeta"]
            

#             # , "tpcNSigmaEl", "tofNSigmaEl", 
#             #      "tofNSigmaPr", "tpcNSigmaPr", "tofNSigmaPi", "tpcNSigmaPi", 
#             #     "tofNSigmaKa", "tpcNSigmaKa"]


y_true = all_test_labels
y_pred = all_test_predictions
y_score = all_test_probabilities

print(all_test_labels, testing_target)

# Print a dataframe with the number of samples for each label
print(pd.DataFrame({"label": np.unique(all_test_labels), "count": np.bincount(all_test_labels)}))

# Get probabilities for the electron class
electron_probs = y_score[:, 0]
electron_true = (y_true == 0).astype(int)

print("Classification Report:")
print(classification_report(y_true, y_pred))

def plot_precision_recall_vs_thresholds(precisions, recalls, thresholds):
    plt.figure(figsize=(10,6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.grid()
    plt.legend()
    plt.title("Precision and Recall for Electron Class vs Decision Threshold")

# Compute precision-recall curve
prec, recall, thresholds = precision_recall_curve(electron_true, electron_probs)

auc_precision_recall = auc(recall, prec)

# Precision-Recall Curve plot
plt.figure(figsize=(10,6))
plt.plot(recall, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve AUC = %.5f" % auc_precision_recall)
plt.grid()
plt.show()

# Plot
plot_precision_recall_vs_thresholds(prec, recall, thresholds)
plt.show()

electron_threshold = 0.9550
#0.9550
# 976990
# 0.957564
# 0.982826
electron_predictions = (y_score[:, 0] > electron_threshold).astype(int)

# print(y_true)
# print(y_score)    

# Create a DataFrame with the precision and recall
df = pd.DataFrame()
df['precision'] = prec
df['recall'] = recall
df['thresholds'] = np.append(thresholds, 1)

# Select specific range of precision
df = df[(df['precision'] > 0.998)]
print(df)

prec = metrics.precision_score(electron_true, electron_predictions)
recall = metrics.recall_score(electron_true, electron_predictions)

# Check performance with new threshold
print(classification_report(electron_true, electron_predictions, digits=6))

# # Plot the electron probabilities of all the actual electrons as a 2d histogram with momentum axis using plotly
fig = go.Figure(data=go.Scatter(x=testing_input[y_true == 0][:,1], y=electron_probs[y_true == 0], mode='markers', marker=dict(size=2), name='All Electrons'))
# # Now add the electron probabilities per each other class
# fig.add_trace(go.Scatter(x=testing_input[(y_true == 1)][:,1], y=electron_probs[(y_true == 1)], mode='markers', marker=dict(size=2), name='Muons', marker_color='purple'))
# fig.add_trace(go.Scatter(x=testing_input[(y_true == 2)][:,1], y=electron_probs[(y_true == 2)], mode='markers', marker=dict(size=2), name='Pions', marker_color='red'))
# fig.add_trace(go.Scatter(x=testing_input[(y_true == 3)][:,1], y=electron_probs[(y_true == 3)], mode='markers', marker=dict(size=2), name='Kaons', marker_color='black'))
# fig.add_trace(go.Scatter(x=testing_input[(y_true == 4)][:,1], y=electron_probs[(y_true == 4)], mode='markers', marker=dict(size=2), name='Protons', marker_color='cyan'))
#Add all the background classes as one class
fig.update_layout(title='Electron Probability vs p', xaxis_title='p', yaxis_title='Electron Probability')
fig.show()

# Plot with new threshold classification
plt.scatter(testing_input[:,1], testing_input[:,0], color='grey', label='All Classes', s=0.5)
plt.scatter(testing_input[y_true == 0][:,1], testing_input[y_true == 0][:,0], color='red', label='Electrons', s=0.75)
plt.scatter(testing_input[(electron_predictions == 1) & (y_true == 0)][:,1], testing_input[(electron_predictions == 1) & (y_true == 0)][:,0], color='green', label='True Positive (Electrons)', s=0.75)
plt.scatter(testing_input[(y_true == 1) & (electron_predictions == 1)][:,1], testing_input[(y_true == 1) & (electron_predictions == 1)][:,0], color='purple', label='Muons', s=5)
plt.scatter(testing_input[(y_true == 2) & (electron_predictions == 1)][:,1], testing_input[(y_true == 2) & (electron_predictions == 1)][:,0], color='blue', label='Pion', s=5)
plt.scatter(testing_input[(y_true == 3) & (electron_predictions == 1)][:,1], testing_input[(y_true == 3) & (electron_predictions == 1)][:,0], color='black', label='Kaon', s=5)
plt.scatter(testing_input[(y_true == 4) & (electron_predictions == 1)][:,1], testing_input[(y_true == 4) & (electron_predictions == 1)][:,0], color='cyan', label='Proton', s=5)
plt.grid()
plt.gcf().set_size_inches(15, 10)
plt.xlabel('p')
plt.ylabel(r'($dE/dx - dE/dx_{\text{exp}}^{\text{el}})/dE/dx_{\text{exp}}^{\text{el}}$')
plt.title(f'Threshold: {electron_threshold}, Recall: {recall:.4f}, Precision: {prec:.4f}')
plt.ylim(-0.5, 0.5)
plt.xlim(0, 3)
plt.legend()
plt.show()
