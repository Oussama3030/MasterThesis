import optuna
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
import optuna.visualization as vis

training_samples = np.load('training_samples.npy')
mapped_targets = np.load('mapped_targets.npy')

X_train, X_test, y_train, y_test = train_test_split(training_samples, 
                                                    mapped_targets, 
                                                    test_size=0.2, 
                                                    stratify=mapped_targets, 
                                                    random_state=42)

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.01, 5.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'objective': 'multi:softprob',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'tree_method': 'hist' 
    }

    model = XGBClassifier(**param)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)  

best_params = study.best_params
print(best_params)


fig1 = vis.plot_optimization_history(study)
fig1.show()

fig2 = vis.plot_parallel_coordinate(study)
fig2.show()

fig3 = vis.plot_param_importances(study)
fig3.show()

fig4 = vis.plot_slice(study, params=['n_estimators', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree', 'gamma', 'learning_rate'])
fig4.show()



# optuna.visualization.plot_optimization_history(study)
# plt.show()

# optuna.visualization.plot_parallel_coordinate(study)
# plt.show()

# optuna.visualization.plot_param_importances(study)
# plt.show()

# optuna.visualization.plot_slice(study, params=['n_estimators', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree', 'gamma', 'learning_rate'])
# plt.show()

