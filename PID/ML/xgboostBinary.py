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
training_samples = np.load('training_samples.npy')
mapped_targets = np.load('mapped_targets.npy')

print(training_samples.shape)
print(mapped_targets.shape)
# Create DataFrame
features = ["dE/dx", "pT", "tofBeta"]

# , "tpcNSigmaEl", "tofNSigmaEl", 
#                 "tofNSigmaPr", "tpcNSigmaPr", "tofNSigmaPi", "tpcNSigmaPi", 
#                 "tofNSigmaKa", "tpcNSigmaKa"]

# Split the data
training_input, testing_input, training_target, testing_target = train_test_split(training_samples, 
                                                    mapped_targets, 
                                                    test_size=0.3, # 20% test, 80% train
                                                    stratify=mapped_targets, # Ensures similar ratio's in test and training 
                                                    random_state=42) # make the random split reproducible

# params = {'max_depth': 3, 'n_estimators': 390, 'tree_method': 'hist', 'min_child_weight': 6, 'subsample': 0.9670026595172778, 
#           'colsample_bytree': 0.8207917373372546, 'gamma': 4.508976811996854, 'learning_rate': 0.12623238845489873, 
#           'eval_metric': 'aucpr', 'objective': 'binary:logistic'}

# params = {'n_estimators': 837, 'max_depth': 9, 'min_child_weight': 1, 
#          'subsample': 0.7553736512887829, 'colsample_bytree': 0.7087055015743895, 
#          'gamma': 2.228857026602595, 'objective': 'binary:logistic', 
#          'eval_metric': 'aucpr', 'tree_method':'hist'}

params = {'n_estimators': 104, 'max_depth': 9, 'min_child_weight': 8, 'subsample': 0.8645035840204937, 
          'colsample_bytree': 0.8856351733429728, 'gamma': 0.7497060708235628, 'objective': 'binary:logistic', 
         'eval_metric': 'aucpr', 'tree_method':'hist'}


# 'scale_pos_weight': 11.091305274794989, 
# Define the scale pos weight
scale_pos_weight =  np.sum(training_target == 0) / np.sum(training_target == 1)

# Train the model
clf = xgb.XGBClassifier(**params, scale_pos_weights = scale_pos_weight)
clf.fit(training_input, training_target,  
        eval_set=[(training_input, training_target), (testing_input, testing_target)], 
        verbose=2)

# Save the model 
clf.save_model('BinaryModel.json')
