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

print(training_samples.shape)
print(mapped_targets.shape)
# Create DataFrame
features = ["dE/dx", "pT", "tofBeta"]

# , "tpcNSigmaEl", "tofNSigmaEl", 
#                 "tofNSigmaPr", "tpcNSigmaPr", "tofNSigmaPi", "tpcNSigmaPi", 
#                 "tofNSigmaKa", "tpcNSigmaKa"]


##########################################################################

# MODEL FOR ALL P RANGES

##########################################################################


# Split the data
training_input, testing_input, training_target, testing_target = train_test_split(training_samples, 
                                                    mapped_targets, 
                                                    test_size=0.3, # 20% test, 80% train
                                                    stratify=mapped_targets, # Ensures similar ratio's in test and training 
                                                    random_state=42) # make the random split reproducible


# params = {
#           'max_depth': 9, 
#           'n_estimators': 119, 
#           'objective': 'multi:softprob', 
#           'tree_method': 'approx', 
#           'eval_metric': 'auc'
#           }

# params = {'max_depth': 3, 'n_estimators': 390, 'tree_method': 'hist', 'min_child_weight': 6, 'subsample': 0.9670026595172778, 
#           'colsample_bytree': 0.8207917373372546, 'gamma': 4.508976811996854, 'learning_rate': 0.12623238845489873, 
#           'scale_pos_weight': 11.091305274794989, 'eval_metric': 'aucpr', 'objective': 'multi:softmax', 'num_class': 5}

# params = {'max_depth': 8, 'n_estimators': 114, 'tree_method': 'approx', 
#           'min_child_weight': 1, 'subsample': 0.8373789281965098, 'colsample_bytree': 0.8682734933639846, 
#           'gamma': 1.7933569494141572, 'learning_rate': 0.2913448240677106, 'objective': 'multi:softprob', 'eval_metric': 'auc'
# }

params = {'n_estimators': 175, 'max_depth': 11, 'min_child_weight': 5, 
          'subsample': 0.8384450785184981, 'colsample_bytree': 0.808193537251804, 
          'gamma': 1.4444987173198711, 'learning_rate': 0.129286878127211,
          'objective': 'multi:softprob', 'eval_metric': 'auc', 'tree_method': 'approx'}


# Compute class weights based on target distribution
sample_weights = class_weight.compute_sample_weight(
    class_weight='balanced', 
    y=training_target
)

# sample_weights = 1./sample_weights

# print(1./sample_weights)

# Define the scale pos weight
# scale_pos_weight =  np.sum(training_target == 0) / np.sum(training_target == 1)

# Train the model
clf = xgb.XGBClassifier(**params)
clf.fit(training_input, training_target,

        eval_set=[(training_input, training_target), (testing_input, testing_target)], 
        verbose=2)
        # sample_weight=1./sample_weights,  


import os
# Define the absolute path

save_dir = "~/o2workdir/PID/ML/Models"
# os.makedirs(save_dir, exist_ok=True)

# Save the model
clf.save_model(os.path.join(save_dir, 'XGBModel.json'))


##########################################################################

# MODEL FOR 0.5 < P < 1.0

##########################################################################


# # Prep the data by selecting the pt range
# pt_range = (0.5, 1.0)

# # Load and prepare data by creating a mask first 

# mask = (training_samples[:,1] > 0.75) #& (training_samples[:,1] < pt_range[1])

# training_samples = training_samples[mask]
# mapped_targets = mapped_targets[mask]

# # Split the data
# training_input, testing_input, training_target, testing_target = train_test_split(training_samples, 
#                                                     mapped_targets, 
#                                                     test_size=0.3, # 20% test, 80% train
#                                                     stratify=mapped_targets, # Ensures similar ratio's in test and training 
#                                                     random_state=42) # make the random split reproducible

# print(training_input.shape)

# # Define the scale pos weight
# scale_pos_weight =  np.sum(training_target == 0) / np.sum(training_target == 1)

# # params = {'max_depth': 10, 'n_estimators': 186, 'tree_method': 'hist', 'objective': 'binary:logistic', 'scale_pos_weight': scale_pos_weight, 'eval_metric': 'aucpr'}
# # params = {'max_depth': 5, 'n_estimators': 164, 'tree_method': 'auto', 'min_child_weight': 3, 'subsample': 0.9751577927516398, 'colsample_bytree': 0.8172671049132658, 'gamma': 2.589999580778448, 'scale_pos_weight': scale_pos_weight, 'objective': 'binary:logistic', 'eval_metric': 'aucpr'}


# params = {'max_depth': 10, 'n_estimators': 500, 'tree_method': 'hist', 'min_child_weight': 6, 'subsample': 0.8797613972542115, 'colsample_bytree': 0.6768097633093875, 'gamma': 4.467235705931428, 'learning_rate': 0.25715408450511723, 'scale_pos_weight':15.29225877167869, 'eval_metric': 'aucpr', 'objective': 'binary:logistic'}

# #

# clf2 = xgb.XGBClassifier(**params)
# clf2.fit(training_input, training_target,
#         eval_set=[(training_input, training_target), (testing_input, testing_target)], 
#         verbose=2)

# clf2.save_model('model2.json')












