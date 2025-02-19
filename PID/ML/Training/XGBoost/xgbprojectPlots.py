import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier
import seaborn as sns
from sklearn import metrics 
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.utils import class_weight
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay
from xgboost import plot_tree
from supertree import SuperTree
from mpl_toolkits.mplot3d import Axes3D
import plotly.io as pio
from plotly.offline import plot
# import plotly.graph_objects as go
import plotly.graph_objs as go


# Load and prepare data
training_samples = np.load('/Users/oussamabenchikhi/o2workdir/PID/ML/Data/training_samples.npy')
mapped_targets = np.load('/Users/oussamabenchikhi/o2workdir/PID/ML/Data/mapped_targets.npy')

# Apply a P range

# pt_range = (0.5, 1.0)
# mask = (training_samples[:,1] > 0.75) 

# training_samples = training_samples[mask]
# mapped_targets = mapped_targets[mask]

# Create DataFrame
features = ["dE/dx", "P", "tofBeta"]
            

            # , "tpcNSigmaEl", "tofNSigmaEl", 
            #      "tofNSigmaPr", "tpcNSigmaPr", "tofNSigmaPi", "tpcNSigmaPi", 
            #     "tofNSigmaKa", "tpcNSigmaKa"]

# Split the data
training_input, testing_input, training_target, testing_target = train_test_split(training_samples, 
                                                    mapped_targets, 
                                                    test_size=0.3, # 20% test, 80% train
                                                    stratify=mapped_targets, # Ensures similar ratio's in test and training 
                                                    random_state=42) # make the random split reproducible


# Print the number of samples in the testing data that belong to the positive class and negative class in a dataframe
print(pd.DataFrame(testing_target, columns=['target'])['target'].value_counts())

# Load the model model
clf = xgb.XGBClassifier()
clf.load_model('/Users/oussamabenchikhi/o2workdir/PID/ML/Models/XGBModel.json')

##########################################################################

# MULTICLASS CLASSIFICATION CODE

########################################################################### 

y_true = testing_target
y_pred = clf.predict(testing_input)  # This directly gives class predictions
y_score = clf.predict_proba(testing_input)  # This gives probability scores for each class

# Get probabilities for the electron class
electron_probs = y_score[:, 0]
electron_true = (y_true == 0).astype(int)

print("Classification Report:")
print(classification_report(y_true, y_pred))


# Plot all the model predictions for the 5 classes in the dEdx vs p plane  
# plt.scatter(testing_input[:,1], testing_input[:,0], c=y_pred, s=0.5)
# plt.grid()
# plt.xlabel('p')
# plt.ylabel(r'($dE/dx - dE/dx_{\text{exp}}^{\text{el}})/dE/dx_{\text{exp}}^{\text{el}}$')
# plt.title('Model Predictions')
# plt.ylim(-0.5, 0.5)
# plt.xlim(0, 3)
# plt.colorbar()
# plt.show()

import plotly.express as px

# Create a DataFrame for easier plotting
df = pd.DataFrame({
    'p': testing_input[:,1],
    'dEdx': testing_input[:,0],
    'dim3': testing_input[:,2],
    'prediction': y_pred
})

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=df['p'],
    y=df['dEdx'],
    z=df['dim3'],
    mode='markers',
    marker=dict(
        size=20,
        color=df['prediction'],
        colorscale='Viridis',
        opacity=0.8
    )
)])

# Update the layout
fig.update_layout(
    title='Model Predictions',
    scene=dict(
        xaxis_title='p',
        yaxis_title='(dE/dx - dE/dx_exp^el)/dE/dx_exp^el',
        zaxis_title='t',
        xaxis=dict(range=[0, 3]),
        yaxis=dict(range=[-0.5, 0.5]),
        zaxis=dict(range=[-0.5, 0.5])
    ),
    width=1000,
    height=800
)

# Save the plot as an HTML file
fig.write_html("model_predictions_3d.html")

# # Plot the Pt vs Tpc Signal distribution for electrons vs other classes
# plt.scatter(testing_input[:,1], testing_input[:,0], color='grey', label='All Classes', s=0.5)
# plt.scatter(testing_input[y_true == 0][:,1], testing_input[y_true == 0][:,0], color='red', label='Electrons', s=0.75)
# plt.scatter(testing_input[(y_pred == 0) & (y_true == 0)][:,1], testing_input[(y_pred == 0) & (y_true == 0)][:,0], color='green', label='True Positive (Electrons)', s=0.75)
# plt.scatter(testing_input[(y_true != 0) & (y_pred == 0)][:,1], testing_input[(y_true != 0) & (y_pred == 0)][:,0], color='blue', label='False Positive (Misclassified as Electrons)', s=5)
# plt.grid()
# plt.gcf().set_size_inches(15, 10)
# plt.xlabel('p')
# plt.ylabel('dE/dx')
# plt.title(f'Multiclass Electron Classification')
# plt.ylim(-0.5, 0.5)
# plt.xlim(0, 3)
# plt.legend()
# plt.show()

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

print(y_true)
print(y_score)    

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

# Only add the probabilities for electron_probs > 0.8 using an if statement for the plot below

# # Plot the electron probabilities of all the actual electrons as a 2d histogram with momentum axis using plotly
fig = go.Figure(data=go.Scatter(x=testing_input[y_true == 0][:,1], y=electron_probs[y_true == 0], mode='markers', marker=dict(size=2), name='All Electrons'))
# Now add the electron probabilities per each other class
fig.add_trace(go.Scatter(x=testing_input[(y_true == 1)][:,1], y=electron_probs[(y_true == 1)], mode='markers', marker=dict(size=2), name='Muons', marker_color='purple'))
fig.add_trace(go.Scatter(x=testing_input[(y_true == 2)][:,1], y=electron_probs[(y_true == 2)], mode='markers', marker=dict(size=2), name='Pions', marker_color='red'))
fig.add_trace(go.Scatter(x=testing_input[(y_true == 3)][:,1], y=electron_probs[(y_true == 3)], mode='markers', marker=dict(size=2), name='Kaons', marker_color='black'))
fig.add_trace(go.Scatter(x=testing_input[(y_true == 4)][:,1], y=electron_probs[(y_true == 4)], mode='markers', marker=dict(size=2), name='Protons', marker_color='cyan'))
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

# Plot with new threshold classification
plt.scatter(testing_input[:,1], testing_input[:,2], color='grey', label='All Classes', s=0.5)
plt.scatter(testing_input[y_true == 0][:,1], testing_input[y_true == 0][:,2], color='red', label='Electrons', s=0.75)
plt.scatter(testing_input[(electron_predictions == 1) & (y_true == 0)][:,1], testing_input[(electron_predictions == 1) & (y_true == 0)][:,2], color='green', label='True Positive (Electrons)', s=0.75)
plt.scatter(testing_input[(y_true == 1) & (electron_predictions == 1)][:,1], testing_input[(y_true == 1) & (electron_predictions == 1)][:,2], color='purple', label='Muons', s=5)
plt.scatter(testing_input[(y_true == 2) & (electron_predictions == 1)][:,1], testing_input[(y_true == 2) & (electron_predictions == 1)][:,2], color='blue', label='Pion', s=5)
plt.scatter(testing_input[(y_true == 3) & (electron_predictions == 1)][:,1], testing_input[(y_true == 3) & (electron_predictions == 1)][:,2], color='black', label='Kaon', s=5)
plt.scatter(testing_input[(y_true == 4) & (electron_predictions == 1)][:,1], testing_input[(y_true == 4) & (electron_predictions == 1)][:,2], color='cyan', label='Proton', s=5)
plt.grid()
plt.gcf().set_size_inches(15, 10)
plt.xlabel('p')
plt.ylabel('TOF Signal')
plt.ylabel(r'$\frac{t - t_{\text{exp}}^{\text{el}}}{t_{\text{exp}}^{\text{el}}}$')
plt.title(f'Threshold: {electron_threshold}, Recall: {recall:.4f}, Precision: {prec:.4f}')
plt.ylim(-0.5, 0.5)
plt.xlim(0, 3)
plt.legend()
plt.show()

# Plot the Pt distribution of all the electrons (everything labeled 1) and the true positives
plt.hist(testing_input[y_true == 0][:,1], bins=400, histtype='step', color='red', label='All Electrons')
plt.hist(testing_input[(electron_predictions == 1) & (y_true == 0)][:,1], bins=100, histtype='step', color='green', label='True Positive')
plt.hist(testing_input[(y_true != 0) & (electron_predictions == 1)][:,1], bins=100, histtype='step', color='blue', label='False Positive')
plt.grid()
plt.xlim(0,3)
plt.xlabel('p')
plt.ylabel('Number of Events')
plt.title(f'True Positives: {np.sum((electron_predictions == 1) & (y_pred == 0))}, Total Electrons: {np.sum(y_true == 0)}')
plt.legend()
plt.show()

# Super Tree
st = SuperTree(clf, testing_input, testing_target, features)
st.save_html("tree")





# # Create 3D figure
# fig = plt.figure(figsize=(15, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Plot all points in grey first (background)
# ax.scatter(testing_input[:,1], testing_input[:,0], testing_input[:,2], 
#           color='grey', label='All Classes', s=0.5, alpha=0.3)

# # Plot all electrons in red
# ax.scatter(testing_input[y_true == 0][:,1], 
#           testing_input[y_true == 0][:,0], 
#           testing_input[y_true == 0][:,2], 
#           color='red', label='All Electrons', s=0.75)

# # Plot true positive electrons in green
# ax.scatter(testing_input[(electron_predictions == 1) & (y_true == 0)][:,1],
#           testing_input[(electron_predictions == 1) & (y_true == 0)][:,0],
#           testing_input[(electr on_predictions == 1) & (y_true == 0)][:,2],
#           color='green', label='True Positive Electrons', s=0.75)

# # Set labels and limits
# ax.set_xlabel('p')
# ax.set_ylabel('dE/dx')
# ax.set_zlabel('TOF Signal')

# ax.set_xlim(0, 3)
# ax.set_ylim(-0.5, 0.5)
# ax.set_zlim(-0.5, 0.5)

# # Add grid
# ax.grid(True)

# # Add legend
# ax.legend()

# # Adjust the viewing angle for better visualization
# ax.view_init(elev=20, azim=45)

# # plt.show()

# Calculate permutation feature importance
# testing_input_df = pd.DataFrame(testing_input, columns=features)

# # print(testing_input_df)
# result = permutation_importance(
#     clf, testing_input, testing_target, scoring='average_precision', n_repeats=5, random_state=42
# )

# feature_importance = pd.Series(result.importances_mean, index=features).sort_values(ascending=False)
# print(feature_importance)

# # Plot feature importance   
# sns.barplot(x=feature_importance, y=feature_importance.index)
# plt.xlabel('Feature Importance')
# plt.ylabel('Feature')
# plt.title('Feature Importance')
# plt.show()

# Add more intersting visualizations here

# Plot what the decision tree looks like with feature names 

# plot_tree(clf, num_trees=0, rankdir='LR', feature_names=features)
# plt.show()



