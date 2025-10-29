import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from ddh.ddh_api import DynamicDeepHit
from sklearn.model_selection import ParameterGrid
import warnings
import random
from auton_survival.models.dsm import datasets

# Load dataset
x, t, e = datasets.load_dataset('PBC', sequential = True)

# Compute time horizons
horizons = [0.25, 0.5, 0.75]
times = np.quantile([t_[-1] for t_, e_ in zip(t, e) if e_[-1] == 1], horizons).tolist()

# Split dataset
n = len(x)

tr_size = int(n*0.70)
vl_size = int(n*0.10)
te_size = int(n*0.20)

x_train, x_test, x_val = np.array(x[:tr_size], dtype = object), np.array(x[-te_size:], dtype = object), np.array(x[tr_size:tr_size+vl_size], dtype = object)
t_train, t_test, t_val = np.array(t[:tr_size], dtype = object), np.array(t[-te_size:], dtype = object), np.array(t[tr_size:tr_size+vl_size], dtype = object)
e_train, e_test, e_val = np.array(e[:tr_size], dtype = object), np.array(e[-te_size:], dtype = object), np.array(e[tr_size:tr_size+vl_size], dtype = object)

# Set model parameters
layers = [[], [100], [100, 100], [100, 100, 100]]

param_grid = {
              'layers_rnn': [2, 3],
              'hidden_long': layers,
              'hidden_rnn': [50, 100],
              'hidden_att': layers,
              'hidden_cs': layers,
              'sigma': [0.1, 1, 3],
              'learning_rate' : [1e-3],
             }
params = ParameterGrid(param_grid)

# Train and select best model
models = []
for param in params:
    model = DynamicDeepHit(
                layers_rnn = param['layers_rnn'],
                hidden_rnn = param['hidden_rnn'], 
                long_param = {'layers': param['hidden_long'], 'dropout': 0.3}, 
                att_param = {'layers': param['hidden_att'], 'dropout': 0.3}, 
                cs_param = {'layers': param['hidden_cs'], 'dropout': 0.3},
                sigma = param['sigma'],
                split = [0] + times + [np.max([t_.max() for t_ in t])])
    # The fit method is called to train the model
    model.fit(x_train, t_train, e_train, iters = 10, 
              learning_rate = param['learning_rate'])
    models.append([[model.compute_nll(x_val, t_val, e_val), model]])
best_model = min(models)
model = best_model[0][1]

best_model = min(models)
model = best_model[0][1]

# Inference
out_risk = model.predict_risk(x_test, times, all_step = True)
out_survival = model.predict_survival(x_test, times, all_step = True)

# Evaluation
cis = []
brs = []

et_train = np.array([(e_train[i][j], t_train[i][j]) for i in range(len(e_train)) for j in range(len(e_train[i]))],
                 dtype = [('e', bool), ('t', float)])
et_test = np.array([(e_test[i][j], t_test[i][j]) for i in range(len(e_test)) for j in range(len(e_test[i]))],
                 dtype = [('e', bool), ('t', float)])
et_val = np.array([(e_val[i][j], t_val[i][j]) for i in range(len(e_val)) for j in range(len(e_val[i]))],
                 dtype = [('e', bool), ('t', float)])

for i, _ in enumerate(times):
    cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])
brs.append(brier_score(et_train, et_test, out_survival, times)[1])
roc_auc = []

for i, _ in enumerate(times):
    roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0])

for horizon in enumerate(horizons):
    print(f"For {horizon[1]} quantile,")
    print("TD Concordance Index:", cis[horizon[0]])
    print("Brier Score:", brs[0][horizon[0]])
    print("ROC AUC ", roc_auc[horizon[0]][0], "\n")