import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.utils import class_weight
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from functools import partial
from skopt import space
from skopt import gp_minimize
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    average_precision_score  # Add this import
)

# Load and prepare data
training_samples = np.load('/Users/oussamabenchikhi/o2workdir/PID/ML/Data/training_samples.npy')
mapped_targets = np.load('/Users/oussamabenchikhi/o2workdir/PID/ML/Data/mapped_targets.npy')

# mask = (training_samples[:,1] > 0.75) 

# training_samples = training_samples[mask]
# mapped_targets = mapped_targets[mask]

##########################################################################

# MULTICLASS CLASSIFICATION CODE

########################################################################### 
def optimize_multiclass_metric(params, param_names, x, y):
    # Convert params to dictionary
    params = dict(zip(param_names, params))
    
    # Initialize multiclass XGBoost classifier
    model = xgb.XGBClassifier(**params)
    
    # Use StratifiedKFold for multiclass
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Metrics for multiclass
    macro_f1_scores = []
    macro_ap_scores = []
    
    for train_idx, test_idx in kf.split(x, y):
        xtrain, ytrain = x[train_idx], y[train_idx]
        xtest, ytest = x[test_idx], y[test_idx]
        
        # Fit the model
        model.fit(xtrain, ytrain)
        
        # Predictions
        y_pred = model.predict(xtest)
        y_pred_proba = model.predict_proba(xtest)
        
        # Calculate macro F1 score
        macro_f1 = f1_score(ytest, y_pred, average='macro')
        macro_f1_scores.append(macro_f1)
        
        # Calculate macro Average Precision
        macro_ap = average_precision_score(
            pd.get_dummies(ytest).values, 
            y_pred_proba, 
            average='macro'
        )
        macro_ap_scores.append(macro_ap)
    
    # We'll minimize the negative of these metrics
    combined_metric = -np.mean(macro_f1_scores) - np.mean(macro_ap_scores)
    
    return combined_metric

def run_optimization(X, y):
    # Parameter space for multiclass
    param_space = [
        space.Integer(2, 10, name="max_depth"),
        space.Integer(50, 500, name="n_estimators"),
        space.Categorical(["hist"], name="tree_method"),
        space.Integer(1, 10, name="min_child_weight"),
        space.Real(0.5, 1, prior="uniform", name="subsample"),
        space.Real(0.5, 1, prior="uniform", name="colsample_bytree"),
        space.Real(0.01, 10, prior="uniform", name="gamma"),
        space.Real(0.01, 1, prior="uniform", name="learning_rate"),
        space.Categorical(['multiclass', 'multi:softmax', 'multi:softprob'], name="objective")
    ]
    
    param_names = [
        "max_depth",
        "n_estimators",
        "tree_method",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "gamma",
        "learning_rate",
        "objective"
    ]
    
    optimization_function = partial(
        optimize_multiclass_metric,
        param_names=param_names,
        x=X,
        y=y
    )
    
    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=25,
        n_random_starts=15,
        verbose=10
    )
    
    best_params = dict(zip(param_names, result.x))
    best_score = -result.fun
    
    return best_params, best_score

if __name__ == "__main__":
    # Run optimization
    best_params, best_score = run_optimization(training_samples, mapped_targets)
    
    print(f"Best Parameters: {best_params}")
    print(f"Best Combined Metric (Macro F1 + Macro AP): {best_score:.4f}")
    


##########################################################################

# BINARY CLASSIFICATION CODE

########################################################################### 

# def optimize_auprc(params, param_names, x, y):
#     params = dict(zip(param_names, params))
#     model = xgb.XGBClassifier(**params)
    
#     kf = StratifiedKFold(n_splits=5)
#     auprc_scores = []
    
#     for train_idx, test_idx in kf.split(X=x, y=y):
#         xtrain, ytrain = x[train_idx], y[train_idx]
#         xtest, ytest = x[test_idx], y[test_idx]
        
#         model.fit(xtrain, ytrain)
#         y_pred_proba = model.predict_proba(xtest)[:, 1]
        
#         precision, recall, _ = metrics.precision_recall_curve(ytest, y_pred_proba)
#         auprc = metrics.auc(recall, precision)
#         auprc_scores.append(auprc)
    
#     return -1.0 * np.mean(auprc_scores)

# def run_optimization(X, y):
#     param_space = [
#         space.Integer(2, 10, name="max_depth"),
#         space.Integer(50, 500, name="n_estimators"),
#         space.Categorical(["hist"], name="tree_method"),
#         space.Integer(1, 10, name="min_child_weight"),
#         space.Real(0.5, 1, prior="uniform", name="subsample"),
#         space.Real(0.5, 1, prior="uniform", name="colsample_bytree"),
#         space.Real(0.01, 10, prior="uniform", name="gamma"),
#         space.Real(0.01, 1, prior="uniform", name="learning_rate"),
#         space.Real(0.1, 1000.0, prior="log-uniform", name="scale_pos_weight"),
#         space.Categorical(['aucpr', 'auc', 'error', 'logloss'], name="eval_metric"),
#         space.Categorical(['binary:logistic', 'binary:logitraw', 'binary:hinge'], name="objective")
#     ]

#     param_names = [
#         "max_depth",
#         "n_estimators",
#         "tree_method",
#         "min_child_weight",
#         "subsample",
#         "colsample_bytree",
#         "gamma",
#         "learning_rate",
#         "scale_pos_weight",
#         "eval_metric",
#         "objective"
#     ]

#     optimization_function = partial(
#         optimize_auprc,
#         param_names=param_names,
#         x=X,
#         y=y
#     )
    
#     result = gp_minimize(
#         optimization_function,
#         dimensions=param_space,
#         n_calls=50,
#         n_random_starts=15,
#         verbose=10
#     )
    
#     best_params = dict(zip(param_names, result.x))
#     best_auprc = -result.fun
    
#     return best_params, best_auprc

# if __name__ == "__main__":
#     best_params, best_auprc = run_optimization(training_samples, mapped_targets)
#     print(f"Best Parameters: {best_params}")
#     print(f"Best AUPRC: {best_auprc:.4f}")