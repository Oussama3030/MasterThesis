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
clf.load_model('/Users/oussamabenchikhi/o2workdir/PID/ML/Models/XGBModel_Binary.json')

# Predict the testing data
threshold = 0.961122

#0.999994 
#0.999993 
#0.9999969
#999962

# Predict the testing data
y_true = testing_target
y_pred = (clf.predict_proba(testing_input)[:,1] >= threshold).astype(bool)
y_score = clf.predict_proba(testing_input)

print(y_pred, y_score)
# Calculate the precision and recall
precision = metrics.precision_score(y_true, y_pred)
recall = metrics.recall_score(y_true, y_pred)

# Number of false positives
fp = np.sum((y_true == 0) & (y_pred == 1))

# Plot the Pt vs Tpc Signal distribution for the positive class (signal) of the true positive and false positive using a scatter plot 
plt.scatter(testing_input[:,1], testing_input[:,0], color='grey', label='Everything', s=0.5)
plt.scatter(testing_input[y_true == 1][:,1], testing_input[y_true == 1][:,0], color='red', label='All Electrons', s = 0.75)
plt.scatter(testing_input[(y_pred == 1) & (y_true == 1)][:,1], testing_input[(y_pred == 1) & (y_true == 1)][:,0], color='green', label='True Positive', s=0.75)
plt.scatter(testing_input[(y_true == 0) & (y_pred == 1)][:,1], testing_input[(y_true == 0) & (y_pred == 1)][:,0], color='blue', label='False Positive', s=5)
plt.grid()
plt.gcf().set_size_inches(15, 10)
plt.xlabel('p')
plt.ylabel('dE/dx')
plt.title(f'Threshold: {threshold}, Recall: {recall:.4f}, Precision: {precision:.4f}, N Contamination: {fp}')
plt.ylim(-0.5, 0.5)
plt.xlim(0, 3)
plt.legend()
# plt.savefig(f'xgboostPlots/dedxvsp/p_vs_dedx_{threshold}.png')
# plt.savefig(f'xgboostPlots/dedxvsp/p_vs_dedx_cut75_{threshold}.png')
plt.show()

# Plot the Pt vs Tpc Signal distribution for the positive class (signal) of the true positive and false positive using a scatter plot 
plt.scatter(testing_input[:,1], testing_input[:,2], color='grey', label='Everything', s=0.5)
plt.scatter(testing_input[y_true == 1][:,1], testing_input[y_true == 1][:,2], color='red', label='All Electrons', s = 0.75)
plt.scatter(testing_input[(y_pred == 1) & (y_true == 1)][:,1], testing_input[(y_pred == 1) & (y_true == 1)][:,2], color='green', label='True Positive', s=0.75)
plt.scatter(testing_input[(y_true == 0) & (y_pred == 1)][:,1], testing_input[(y_true == 0) & (y_pred == 1)][:,2], color='blue', label='False Positive', s=5)
plt.grid()
plt.gcf().set_size_inches(15, 10)
plt.xlabel('p')
plt.ylabel('TOF Signal')
plt.title(f'Threshold: {threshold}, Recall: {recall:.4f}, Precision: {precision:.4f}, N Contamination: {fp}')
plt.ylim(-0.5, 0.5)
plt.xlim(0, 3)
plt.legend()
# plt.savefig(f'xgboostPlots/dedxvsp/p_vs_dedx_{threshold}.png')
# plt.savefig(f'xgboostPlots/dedxvsp/p_vs_dedx_cut75_{threshold}.png')
plt.show()

# Plot the Pt distribution of all the electrons (everything labeled 1) and the true positives
plt.hist(testing_input[y_true == 1][:,1], bins=400, histtype='step', color='red', label='All Electrons')
plt.hist(testing_input[(y_pred == 1) & (y_true == 1)][:,1], bins=100, histtype='step', color='green', label='True Positive')
plt.grid()
plt.xlabel('p')
plt.ylabel('Number of Events')
# Plot the title with the number of true positives and total number of electrons
plt.title(f'True Positives: {np.sum((y_true == 1) & (y_pred == 1))}, Total Electrons: {np.sum(y_true == 1)}')
plt.legend()
plt.show()

import plotly.graph_objects as go
import plotly.express as px

# Plot the electorn probabilities as a function of the momentum using plotly of only particles that ARE electrons
fig = go.Figure(data=go.Scatter(x=testing_input[y_true == 1][:,1], y=y_score[y_true == 1][:,1], mode='markers', marker=dict(color='red'), name='Electrons'))
fig.update_layout(title='Electron Probability vs p', xaxis_title='p', yaxis_title='Electron Probability')
fig.show()



# plt.hist(y_score[:,0], bins=50, histtype='step', color='blue', label='Signal')
# plt.show()

# Print the classification report for the testing data
print("Classification Report (Valid):\n", metrics.classification_report(y_true, y_pred, digits=4))

# Calculate the precision-recall curve
prec, recall, thresholds = precision_recall_curve(y_true, y_score[:,1])

# Create a DataFrame with the precision and recall
df = pd.DataFrame()
df['precision'] = prec
df['recall'] = recall
df['thresholds'] = np.append(thresholds, 1)

# Select specific range of precision
df = df[(df['precision'] > 0.99)]
print(df)

# plot the precision-recall curve and calculate the area under the curve
auc_precision_recall = auc(recall, prec)
print('AUC PR: %.5f' % auc_precision_recall)
plt.plot(recall, prec, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve, AUC = %.5f' % auc_precision_recall)
plt.legend()
plt.grid()
plt.show()

# Plot the predicted probabilities for class 1 with the ones that are classified as signal
plt.hist(y_score[y_true == 0][:,1], bins=250, histtype='step', color='blue', label='Background, N = %i' % np.sum(y_true == 0))
plt.hist(y_score[y_true == 1][:,1], bins=250, histtype='step', color='red', label='Electrons, N = %i' % np.sum(y_true == 1))
plt.yscale('log')
plt.grid()
plt.xlabel('Predicted Probability for Class 1')
plt.ylabel('Number of Events')
plt.title('Predicted Probabilities for Class 1')
plt.legend()
plt.show()

# plot the precision-recall curves with Threshold for class 1
def plot_precision_recall_vs_thresholds(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.grid()
    plt.legend()
    plt.title("Precision and Recall Scores as a function of the decision threshold")

plot_precision_recall_vs_thresholds(prec, recall, thresholds)
plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Super Tree
st = SuperTree(clf, testing_input, testing_target, features)
st.save_html("tree")

# Calculate permutation feature importance
testing_input_df = pd.DataFrame(testing_input, columns=features)

# print(testing_input_df)
result = permutation_importance(
    clf, testing_input, testing_target, scoring='average_precision', n_repeats=5, random_state=42
)

feature_importance = pd.Series(result.importances_mean, index=features).sort_values(ascending=False)
print(feature_importance)

# Plot feature importance   
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()
