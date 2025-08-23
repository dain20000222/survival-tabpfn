import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sksurv.util import Surv
from sklearn.pipeline import make_pipeline
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored, integrated_brier_score, cumulative_dynamic_auc
import torch
from pycox.models import DeepHitSingle, CoxPH
import torchtuples as tt
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from itertools import product
import random
import warnings
warnings.filterwarnings("ignore")

# --- Pandas 2.x compatibility shim for scikit-survival ---
if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items
# ---------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Directory containing the datasets
data_dir = os.path.join("test")

# List all CSV files in the directory
dataset_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

# Grid search hyperparameters
cph_grid = {
    'alpha': [1e-6, 1e-3, 1e-1]
}

rsf_grid = {
    'n_estimators': [50, 100, 200],
    'min_samples_split': [2, 5, 10, 20]
}

deephit_grid = {
    'dropout': [0, 0.1],
    'first_layer_size': [32, 128, 256, 512],
    'second_layer_size': [32],
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'num_bins': [5, 10, 15, 20]
}

deepsurv_grid = {
    'dropout': [0, 0.1],
    'first_layer_size': [32, 128, 256, 512],
    'second_layer_size': [32],
    'learning_rate': [1e-3, 1e-4, 1e-5]
}

def censored_c_index(model, x_test, y_test, is_deephit=False, is_deepsurv=False):
    """
    Compute concordance index.
    """
    if is_deephit:
        # Use expected time as risk proxy (lower expected time => higher risk)
        deephit_pmf = model.predict_pmf(x_test)
        time_bins = np.arange(1, deephit_pmf.shape[1] + 1)
        expected_time = np.sum(deephit_pmf * time_bins, axis=1)
        risk_scores = -expected_time
    elif is_deepsurv:
        risk_scores = model.predict(x_test).flatten()
    else:
        risk_scores = model.predict(x_test)

    c_index, *_ = concordance_index_censored(
        y_test["event"], y_test["time"], risk_scores
    )
    return float(c_index)


def compute_ibs(model, x_test, y_test, y_train, times, is_deephit=False, is_deepsurv=False, labtrans=None):
    """
    Compute IBS at fixed `times`.
    """
    if is_deephit:
        deephit_surv_probs = model.predict_surv_df(x_test).T.values
        surv_probs = np.array([np.interp(times, labtrans.cuts, sp) for sp in deephit_surv_probs])

    elif is_deepsurv:
        model.compute_baseline_hazards()
        deepsurv_surv = model.predict_surv_df(x_test)
        deepsurv_surv_probs = deepsurv_surv.T.values
        surv_probs = np.array([np.interp(times, deepsurv_surv.index, sp) for sp in deepsurv_surv_probs])

    else:
        surv_funcs = model.predict_survival_function(x_test)
        surv_probs = np.row_stack([sf(times) for sf in surv_funcs])

    ibs = integrated_brier_score(y_train, y_test, surv_probs, times)
    return float(ibs)


def compute_mean_auc(model, x_test, y_test, y_train, times, is_deephit=False, is_deepsurv=False):
    """
    Compute mean time-dependent AUC across `times`.
    """
    if is_deephit:
        deephit_pmf = model.predict_pmf(x_test).detach().cpu().numpy()
        time_bins = np.arange(1, deephit_pmf.shape[1] + 1)  
        expected_time = np.sum(deephit_pmf * time_bins, axis=1)
        risk_scores = -expected_time
    elif is_deepsurv:
        risk_scores = model.predict(x_test).flatten().cpu().numpy()
    else:
        risk_scores = model.predict(x_test)

    _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
    return mean_auc

def evaluate_model_fold(model, x_test, y_test, y_train, times, model_type='standard', **kwargs):
    """
    Evaluate a single model on a single fold .
    """
    results = {}

    if model_type == 'deephit':
        c_index = censored_c_index(model, x_test.cpu().numpy(), y_test, is_deephit=True)
        ibs = compute_ibs(model, x_test, y_test, y_train, times, is_deephit=True, labtrans=kwargs.get('labtrans'))
        mean_auc = compute_mean_auc(model, x_test, y_test, y_train, times, is_deephit=True)
    elif model_type == 'deepsurv':
        c_index = censored_c_index(model, x_test.cpu().numpy(), y_test, is_deepsurv=True)
        ibs = compute_ibs(model, x_test, y_test, y_train, times, is_deepsurv=True)
        mean_auc = compute_mean_auc(model, x_test, y_test, y_train, times, is_deepsurv=True)
    else:
        c_index = censored_c_index(model, x_test, y_test)
        ibs = compute_ibs(model, x_test, y_test, y_train, times)
        mean_auc = compute_mean_auc(model, x_test, y_test, y_train, times)

    results['c_index'] = c_index
    results['ibs'] = ibs
    results['mean_auc'] = mean_auc
    return results

def grid_search_cph(x_train, y_train, x_val, y_val, param_grid):
    best_score = -np.inf
    best_params = None
    
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        try:
            model = make_pipeline(
                StandardScaler(),
                CoxPHSurvivalAnalysis(alpha=param_dict['alpha'])
            )
            model.fit(x_train, y_train)
            risk_scores = model.predict(x_val)
            c_index, *_ = concordance_index_censored(y_val["event"], y_val["time"], risk_scores)
            
            if c_index > best_score:
                best_score = c_index
                best_params = param_dict
        except Exception as e:
            print(f"CoxPH trial failed with params {param_dict}: {e}")
            continue
    
    return best_params, best_score

def grid_search_rsf(x_train, y_train, x_val, y_val, param_grid):
    best_score = -np.inf
    best_params = None
    
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        try:
            model = make_pipeline(
                StandardScaler(),
                RandomSurvivalForest(
                    n_estimators=param_dict['n_estimators'],
                    min_samples_split=param_dict['min_samples_split'],
                    random_state=SEED
                )
            )
            model.fit(x_train, y_train)
            risk_scores = model.predict(x_val)
            c_index, *_ = concordance_index_censored(y_val["event"], y_val["time"], risk_scores)
            
            if c_index > best_score:
                best_score = c_index
                best_params = param_dict
        except Exception as e:
            print(f"RSF trial failed with params {param_dict}: {e}")
            continue
    
    return best_params, best_score

def grid_search_deephit(x_train, y_train, x_val, y_val, times, param_grid):
    best_score = -np.inf
    best_params = None
    
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        try:
            # Preprocess
            scaler = StandardScaler()
            X_train_arr = scaler.fit_transform(x_train)
            X_val_arr = scaler.transform(x_val)
            
            # Label transform
            labtrans = LabTransDiscreteTime(param_dict['num_bins'])
            y_train_dh = labtrans.fit_transform(y_train["time"], y_train["event"])
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_train_dh = torch.tensor(np.asarray(X_train_arr), dtype=torch.float32, device=device)
            X_val_dh = torch.tensor(np.asarray(X_val_arr), dtype=torch.float32, device=device)
            y_train_time = torch.tensor(y_train_dh[0], dtype=torch.long, device=device)
            y_train_event = torch.tensor(y_train_dh[1], dtype=torch.float32, device=device)
            
            # Model
            in_features = X_train_dh.shape[1]
            out_features = labtrans.out_features
            num_nodes = [param_dict['first_layer_size'], param_dict['second_layer_size']]
            net = tt.practical.MLPVanilla(in_features, num_nodes, out_features,
                                          batch_norm=True, dropout=param_dict['dropout'])
            
            model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
            model.optimizer.set_lr(param_dict['learning_rate'])
            model.fit(X_train_dh, (y_train_time, y_train_event), batch_size=256, epochs=100, verbose=False)
            
            # Evaluate
            deephit_pmf = model.predict_pmf(X_val_dh)
            if isinstance(deephit_pmf, torch.Tensor):
                deephit_pmf = deephit_pmf.detach().cpu().numpy()
            time_bins = np.arange(1, deephit_pmf.shape[1] + 1)
            expected_time = np.sum(deephit_pmf * time_bins, axis=1)
            risk_scores = -expected_time
            c_index, *_ = concordance_index_censored(y_val["event"], y_val["time"], risk_scores)
            
            if c_index > best_score:
                best_score = c_index
                best_params = param_dict
        except Exception as e:
            print(f"DeepHit trial failed with params {param_dict}: {e}")
            continue
    
    return best_params, best_score

def grid_search_deepsurv(x_train, y_train, x_val, y_val, param_grid):
    best_score = -np.inf
    best_params = None
    
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        try:
            # Preprocess
            scaler = StandardScaler()
            X_train_arr = scaler.fit_transform(x_train)
            X_val_arr = scaler.transform(x_val)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_train_ds = torch.tensor(np.asarray(X_train_arr), dtype=torch.float32, device=device)
            X_val_ds = torch.tensor(np.asarray(X_val_arr), dtype=torch.float32, device=device)
            
            y_train_time = torch.tensor(y_train["time"].copy(), dtype=torch.float32, device=device)
            y_train_event = torch.tensor(y_train["event"].astype(int).copy(), dtype=torch.float32, device=device)
            
            # For numerical stability
            epsilon = 1e-6
            y_train_event[y_train_event == 0] = epsilon
            
            # Model
            in_features = X_train_ds.shape[1]
            num_nodes = [param_dict['first_layer_size'], param_dict['second_layer_size']]
            net = tt.practical.MLPVanilla(in_features, num_nodes, 1,
                                          batch_norm=True, dropout=param_dict['dropout'])
            
            model = CoxPH(net, tt.optim.Adam)
            model.optimizer.set_lr(param_dict['learning_rate'])
            model.fit(X_train_ds, (y_train_time, y_train_event), batch_size=256, epochs=100, verbose=False)
            model.compute_baseline_hazards()
            
            # Evaluate
            risk_scores = model.predict(X_val_ds).flatten()
            if isinstance(risk_scores, torch.Tensor):
                risk_scores = risk_scores.detach().cpu().numpy()
            c_index, *_ = concordance_index_censored(y_val["event"], y_val["time"], risk_scores)
            
            if c_index > best_score:
                best_score = c_index
                best_params = param_dict
        except Exception as e:
            print(f"DeepSurv trial failed with params {param_dict}: {e}")
            continue
    
    return best_params, best_score

for file_name in dataset_files:
    try:
        dataset_name = file_name.replace(".csv", "")
        file_path = os.path.join(data_dir, file_name)
        print("="*50)
        print(f"\nProcessing dataset: {dataset_name}")

        # Load datasets
        df = pd.read_csv(file_path)
        df.drop(columns=['pid'], inplace=True)
        shape = df.shape
        print(f"Dataset shape: {shape}")

        # Define columns
        time_col = "time"
        event_col = "event"
        censored = (df[event_col] == 0).sum()
        censored_percent = (censored)/len(df)*100
        print(f"Percentage of censored data: {censored_percent}%")
        covariates = df.columns.difference([time_col, event_col])

        # Define covariates and target variable
        x = df[covariates].copy()
        y = Surv.from_arrays(event=df[event_col].astype(bool), time=df[time_col])

        # --------------- 70-15-15 Split ---------------
        x_trainval, x_test, y_trainval, y_test = train_test_split(
            x, y, test_size=0.15, stratify=y["event"], random_state=SEED
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval, y_trainval, test_size=0.1765, stratify=y_trainval["event"], random_state=SEED
        )
        # 0.1765 * 0.85 ≈ 0.15, so test/val are both 15%

        # One-hot encode using get_dummies (fit on TRAIN only!)
        x_train_ohe = pd.get_dummies(x_train, drop_first=True)
        x_val_ohe   = pd.get_dummies(x_val, drop_first=True)
        x_test_ohe  = pd.get_dummies(x_test, drop_first=True)

        # Align columns so train/val/test match
        x_train_ohe, x_val_ohe = x_train_ohe.align(x_val_ohe, join="left", axis=1, fill_value=0)
        x_train_ohe, x_test_ohe = x_train_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)

        covariates_ohe = x_train_ohe.columns  # all columns are covariates after OHE

        # Impute missing values 
        imputer = SimpleImputer().fit(x_train_ohe.loc[:, covariates_ohe.tolist()])
        x_train_imputed = imputer.transform(x_train_ohe.loc[:, covariates_ohe.tolist()])
        x_train_imputed = pd.DataFrame(x_train_imputed, columns=covariates_ohe, index=x_train.index)
        x_val_imputed = imputer.transform(x_val_ohe.loc[:, covariates_ohe.tolist()])
        x_val_imputed = pd.DataFrame(x_val_imputed, columns=covariates_ohe, index=x_val.index)
        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)

        # Evaluation time points (use train+val)
        times = np.percentile(y_trainval["time"], np.arange(10, 100, 10))
        times = np.unique(times)
        max_trainval_time = y_trainval["time"].max()
        times = times[times < max_trainval_time]
        test_mask = y_test["time"] < max_trainval_time
        y_test_filtered = y_test[test_mask]
        x_test_filtered = x_test_imputed[test_mask]
        times = times[(times > y_test_filtered["time"].min()) & (times < y_test_filtered["time"].max())]

        # --------------- Hyperparameter Tuning using Grid Search ---------------
        # Grid search for CoxPH
        print("Running Grid Search for CoxPH...")
        best_params_cph, best_score_cph = grid_search_cph(x_train_imputed, y_train, x_val_imputed, y_val, cph_grid)
        
        # Fallback for CoxPH if grid search fails
        if best_params_cph is None:
            print("CoxPH grid search failed, using default parameters")
            best_params_cph = {'alpha': 1e-4}
            best_score_cph = 0.0
        print(f"Best CoxPH params: {best_params_cph}, Score: {best_score_cph:.4f}")

        # Grid search for RSF
        print("Running Grid Search for RSF...")
        best_params_rsf, best_score_rsf = grid_search_rsf(x_train_imputed, y_train, x_val_imputed, y_val, rsf_grid)
        
        # Fallback for RSF if grid search fails
        if best_params_rsf is None:
            print("RSF grid search failed, using default parameters")
            best_params_rsf = {'n_estimators': 100, 'min_samples_split': 5}
            best_score_rsf = 0.0
        print(f"Best RSF params: {best_params_rsf}, Score: {best_score_rsf:.4f}")

        # Grid search for DeepHit
        print("Running Grid Search for DeepHit...")
        best_params_dh, best_score_dh = grid_search_deephit(x_train_imputed, y_train, x_val_imputed, y_val, times, deephit_grid)
        
        # Fallback for DeepHit if grid search fails
        if best_params_dh is None:
            print("DeepHit grid search failed, using default parameters")
            best_params_dh = {
                'dropout': 0.1,
                'first_layer_size': 128,
                'second_layer_size': 32,
                'learning_rate': 1e-3,
                'num_bins': 10
            }
            best_score_dh = 0.0
        print(f"Best DeepHit params: {best_params_dh}, Score: {best_score_dh:.4f}")

        # Grid search for DeepSurv
        print("Running Grid Search for DeepSurv...")
        best_params_ds, best_score_ds = grid_search_deepsurv(x_train_imputed, y_train, x_val_imputed, y_val, deepsurv_grid)
        
        # Fallback for DeepSurv if grid search fails
        if best_params_ds is None:
            print("DeepSurv grid search failed, using default parameters")
            best_params_ds = {
                'dropout': 0.1,
                'first_layer_size': 128,
                'second_layer_size': 32,
                'learning_rate': 1e-3
            }
            best_score_ds = 0.0
        print(f"Best DeepSurv params: {best_params_ds}, Score: {best_score_ds:.4f}")

        # --- Save best hyperparameters for this dataset ---
        hp_out = "baseline_hyperparameter.csv"
        write_hp_header = not os.path.exists(hp_out) or os.stat(hp_out).st_size == 0

        with open(hp_out, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_hp_header:
                writer.writerow([
                    "dataset_name",
                    # CoxPH
                    "cph_alpha", "cph_score",
                    # RSF
                    "rsf_n_estimators", "rsf_min_samples_split", "rsf_score",
                    # DeepHit
                    "dh_dropout", "dh_first_layer_size", "dh_second_layer_size", "dh_learning_rate", "dh_num_bins", "dh_score",
                    # DeepSurv
                    "ds_dropout", "ds_first_layer_size", "ds_second_layer_size", "ds_learning_rate", "ds_score"
                ])
            writer.writerow([
                dataset_name,
                # CoxPH
                best_params_cph["alpha"], best_score_cph,
                # RSF
                best_params_rsf["n_estimators"], best_params_rsf["min_samples_split"], best_score_rsf,
                # DeepHit
                best_params_dh["dropout"], best_params_dh["first_layer_size"], best_params_dh["second_layer_size"],
                best_params_dh["learning_rate"], best_params_dh["num_bins"], best_score_dh,
                # DeepSurv
                best_params_ds["dropout"], best_params_ds["first_layer_size"], best_params_ds["second_layer_size"],
                best_params_ds["learning_rate"], best_score_ds
            ])

        # ========== Retrain on train+val, evaluate on test ==========
        # OHE and impute trainval/test
        x_trainval_ohe = pd.get_dummies(x_trainval, drop_first=True)
        x_test_ohe = pd.get_dummies(x_test, drop_first=True)
        x_trainval_ohe, x_test_ohe = x_trainval_ohe.align(x_test_ohe, join="left", axis=1, fill_value=0)
        covariates_ohe = x_trainval_ohe.columns
        imputer = SimpleImputer().fit(x_trainval_ohe.loc[:, covariates_ohe.tolist()])
        x_trainval_imputed = imputer.transform(x_trainval_ohe.loc[:, covariates_ohe.tolist()])
        x_trainval_imputed = pd.DataFrame(x_trainval_imputed, columns=covariates_ohe, index=x_trainval.index)
        x_test_imputed = imputer.transform(x_test_ohe.loc[:, covariates_ohe.tolist()])
        x_test_imputed = pd.DataFrame(x_test_imputed, columns=covariates_ohe, index=x_test.index)
        test_mask = y_test["time"] < max_trainval_time
        y_test_filtered = y_test[test_mask]
        x_test_filtered = x_test_imputed[test_mask]
        times = times[(times > y_test_filtered["time"].min()) & (times < y_test_filtered["time"].max())]

        # ============== RSF ==============
        print("Training RSF...")
        rsf = make_pipeline(
            StandardScaler(),
            RandomSurvivalForest(
                n_estimators=best_params_rsf["n_estimators"],
                min_samples_split=best_params_rsf["min_samples_split"],
                random_state=SEED
            )
        )
        rsf.fit(x_trainval_imputed, y_trainval)

        # Additional time clipping for RSF
        if hasattr(rsf, "named_steps"):
            for step_name in reversed(rsf.named_steps):
                step = rsf.named_steps[step_name]
                if hasattr(step, "predict_survival_function"):
                    estimator = step
                    break
            else:
                raise ValueError("No estimator with predict_survival_function found in pipeline.")
        else:
            estimator = rsf

        event_times = estimator.event_times_
        times = times[(times > event_times[0]) & (times < event_times[-1])] 

        rsf_results = evaluate_model_fold(rsf, x_test_filtered, y_test_filtered, y_trainval, times)

        # ============== CoxPH ==============
        print("Training CoxPH...")
        cph = make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis(alpha=best_params_cph["alpha"]))
        cph.fit(x_trainval_imputed, y_trainval)
        cph_results = evaluate_model_fold(cph, x_test_filtered, y_test_filtered, y_trainval, times)

        # ============== DeepHit ===============
        print("Training DeepHit...")

        # Preprocess via pipeline: StandardScaler (fit on trainval only)
        dh_prep = make_pipeline(StandardScaler())
        X_train_dh_arr = dh_prep.fit_transform(x_trainval_imputed)
        X_test_dh_arr = dh_prep.transform(x_test_filtered)

        # Label transform
        labtrans = LabTransDiscreteTime(best_params_dh["num_bins"])
        y_trainval_dh = labtrans.fit_transform(y_trainval["time"], y_trainval["event"])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train_dh = torch.tensor(np.asarray(X_train_dh_arr), dtype=torch.float32, device=device)
        X_test_dh = torch.tensor(np.asarray(X_test_dh_arr), dtype=torch.float32, device=device)
        y_trainval_time = torch.tensor(y_trainval_dh[0], dtype=torch.long, device=device)
        y_trainval_event = torch.tensor(y_trainval_dh[1], dtype=torch.float32, device=device)

        # Model architecture using best grid search params
        in_features = X_train_dh.shape[1]
        out_features = labtrans.out_features
        num_nodes = [best_params_dh["first_layer_size"], best_params_dh["second_layer_size"]]
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features,
                                      batch_norm=True, dropout=best_params_dh["dropout"])

        deephit = DeepHitSingle(net, tt.optim.Adam,
                                alpha=0.2,
                                sigma=0.1,
                                duration_index=labtrans.cuts)
        deephit.optimizer.set_lr(best_params_dh["learning_rate"])
        deephit.fit(X_train_dh, (y_trainval_time, y_trainval_event),
                    batch_size=256, epochs=300, verbose=False)

        dh_results = evaluate_model_fold(deephit, X_test_dh, y_test_filtered, y_trainval, times, 
                                         model_type='deephit', labtrans=labtrans)

        # ============== DeepSurv ==============
        print("Training DeepSurv...")

        # Preprocess via pipeline: StandardScaler (fit on trainval only)
        ds_prep = make_pipeline(StandardScaler())
        X_train_ds_arr = ds_prep.fit_transform(x_trainval_imputed)
        X_test_ds_arr = ds_prep.transform(x_test_filtered)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train_ds = torch.tensor(np.asarray(X_train_ds_arr), dtype=torch.float32, device=device)
        X_test_ds = torch.tensor(np.asarray(X_test_ds_arr), dtype=torch.float32, device=device)

        y_trainval_time = torch.tensor(y_trainval["time"].copy(), dtype=torch.float32, device=device)
        y_trainval_event = torch.tensor(y_trainval["event"].astype(int).copy(), dtype=torch.float32, device=device)

        # For numerical stability
        epsilon = 1e-6
        y_trainval_event[y_trainval_event == 0] = epsilon

        # Model architecture
        num_nodes = [best_params_ds["first_layer_size"], best_params_ds["second_layer_size"]]
        net = tt.practical.MLPVanilla(in_features, num_nodes, 1,
                                      batch_norm=True, dropout=best_params_ds["dropout"])

        deepsurv = CoxPH(net, tt.optim.Adam)
        deepsurv.optimizer.set_lr(best_params_ds["learning_rate"])
        deepsurv.fit(X_train_ds, (y_trainval_time, y_trainval_event),
                     batch_size=256, epochs=300, verbose=False)
        deepsurv.compute_baseline_hazards()

        ds_results = evaluate_model_fold(deepsurv, X_test_ds, y_test_filtered, y_trainval, times, 
                                         model_type='deepsurv')

        # Print results 
        print("\n" + "="*50)
        print("HOLDOUT TEST SET RESULTS")
        print("="*50)
        
        print(f"\nRandom Survival Forest:")
        print(f"C-index: {rsf_results['c_index']:.4f}")
        print(f"IBS: {rsf_results['ibs']:.4f}")
        print(f"Mean AUC: {rsf_results['mean_auc']:.4f}")

        print(f"\nCoxPH:")
        print(f"C-index: {cph_results['c_index']:.4f}")
        print(f"IBS: {cph_results['ibs']:.4f}")
        print(f"Mean AUC: {cph_results['mean_auc']:.4f}")

        print(f"\nDeepHit:")
        print(f"C-index: {dh_results['c_index']:.4f}")
        print(f"IBS: {dh_results['ibs']:.4f}")
        print(f"Mean AUC: {dh_results['mean_auc']:.4f}")

        print(f"\nDeepSurv:")
        print(f"C-index: {ds_results['c_index']:.4f}")
        print(f"IBS: {ds_results['ibs']:.4f}")
        print(f"Mean AUC: {ds_results['mean_auc']:.4f}")

        # Prepare results for CSV
        results = [dataset_name]
        
        # Add RSF results
        results.extend([
            rsf_results['c_index'],
            rsf_results['ibs'],
            rsf_results['mean_auc']
        ])
        
        # Add CoxPH results
        results.extend([
            cph_results['c_index'],
            cph_results['ibs'],
            cph_results['mean_auc']
        ])
        
        # Add DeepHit results
        results.extend([
            dh_results['c_index'],
            dh_results['ibs'],
            dh_results['mean_auc']
        ])
        
        # Add DeepSurv results
        results.extend([
            ds_results['c_index'],
            ds_results['ibs'],
            ds_results['mean_auc']
        ])

        # Write the results into a file
        output_file = "baseline_evaluation.csv"

        # Check if file is empty or doesn't exist
        write_header = not os.path.exists(output_file) or os.stat(output_file).st_size == 0

        with open(output_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "dataset_name",
                    "rsf_c_index", "rsf_ibs", "rsf_mean_auc",
                    "cph_c_index", "cph_ibs", "cph_mean_auc",
                    "dh_c_index", "dh_ibs", "dh_mean_auc",
                    "ds_c_index", "ds_ibs", "ds_mean_auc"
                ])
            writer.writerow(results)
    except Exception as e:
        print(f" Skipping {file_name}: {e}")
        continue
