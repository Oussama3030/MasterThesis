import seaborn as sns
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import cross_val_score


training_samples = np.load('training_samples.npy')
mapped_targets = np.load('mapped_targets.npy')

features = ["dE/dx", "pT", "tofBeta"]

X_train, X_test, y_train, y_test = train_test_split(training_samples, 
                                                    mapped_targets, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    stratify=mapped_targets, # Ensures similar ratio's in test and training 
                                                    random_state=42) # make the random split reproducible

cf = XGBClassifier()

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.01, 10.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1)
            }

    model = XGBClassifier(**param)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()

    return score

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=42))

study.optimize(objective, n_trials=25)

best_params = study.best_params

