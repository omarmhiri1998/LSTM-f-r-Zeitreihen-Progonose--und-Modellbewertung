# -*- coding: utf-8 -*-

"""
Models implemented in this pipeline:

- LSTM1 (per country forecast model)

Meta-Models:
- META_LSTM
- META_LSTM_STAT
- META_ATT
- META_ATT_BILSTM
- META_ATT_RESID
- META_ATT_BILSTM_RESID
"""

# ======================================================
# IMPORTS
# ======================================================
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, explained_variance_score
from scipy.stats import skew, kurtosis, spearmanr

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ======================================================
# CONFIG
# ======================================================
DATA_PATH = "daily_weather_formatted_Date (1).csv"

OUT_DIR = "plots_v13_all_models_fixed46_bilstm_residlstm"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

WINDOW = 180 # FIXED as requested

# Window splits (on windows)
P35 = 0.35
P50 = 0.50
P75 = 0.75
P90 = 0.90

META_METRICS = ["sMAPE", "MAE", "MASE"]

# Budgets
N_TRIALS_L1 = 8
L1_MAX_EPOCHS = 50

N_TRIALS_META = 20
META_MAX_EPOCHS = 50

# Save Optuna trials CSV
SAVE_OPTUNA_TRIALS = True

# --- ONLY CHANGE #1 (Seasonal MASE period) ---
SEASONAL_PERIOD = 365


# ======================================================
# METRICS
# ======================================================
def smape_scalar(y_true, y_pred):
    y_true = float(y_true)
    y_pred = float(y_pred)
    return float(2.0 * abs(y_pred - y_true) / (abs(y_true) + abs(y_pred) + 1e-8))

# --- ONLY CHANGE #2: seasonal baseline for MASE ---
def mase_scalar(y_true, y_pred, y_train):
    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) <= SEASONAL_PERIOD:
        denom = np.mean(np.abs(np.diff(y_train))) + 1e-8
    else:
        diffs = np.abs(
            y_train[SEASONAL_PERIOD:] -
            y_train[:-SEASONAL_PERIOD]
        )
        denom = np.mean(diffs) + 1e-8

    return float(abs(float(y_true) - float(y_pred)) / denom)

def skill_score_mae(mae_model, mae_baseline):
    return float(1.0 - (mae_model / (mae_baseline + 1e-8)))

def safe_spearman(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) < 2 or len(p) < 2:
        return np.nan
    try:
        return float(spearmanr(y, p).correlation)
    except Exception:
        return np.nan


# ======================================================
# WINDOW STATISTICS (statische Eigenschaften)
# ======================================================
def window_stats(arr):
    arr = np.array(arr, dtype=float)
    return np.array([
        np.mean(arr),
        np.std(arr),
        np.min(arr),
        np.max(arr),
        arr[-1] - arr[0],                                # trend
        skew(arr),
        kurtosis(arr),
        np.percentile(arr, 75) - np.percentile(arr, 25)  # IQR
    ], dtype=float)


# ======================================================
# ATTENTION LAYER
# ======================================================
class Attention(layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

    def call(self, x):
        score = tf.matmul(x, self.W) + self.b
        w = tf.nn.softmax(tf.nn.tanh(score), axis=1)
        return tf.reduce_sum(x * w, axis=1)


# ======================================================
# OPTUNA HELPERS
# ======================================================
def save_optuna_trials(study, fname):
    if not SAVE_OPTUNA_TRIALS:
        return
    try:
        df_trials = study.trials_dataframe()
        df_trials.to_csv(os.path.join(OUT_DIR, fname), index=False)
    except Exception as e:
        print("  [warn] Could not save optuna trials:", fname, "->", str(e))

def pack_best_row(stage, metric, best_value, best_params, extra=None):
    row = {
        "Stage": stage,
        "Metric": metric,
        "BestValue": float(best_value) if best_value is not None else np.nan,
        "BestParamsJSON": json.dumps(best_params, ensure_ascii=False)
    }
    if extra:
        for k, v in extra.items():
            row[k] = v
    return row

def select_best_pareto_trial(study):
    """
    Select one trial from a multi-objective study:
    - Take only COMPLETE trials
    - Compute ideal point (min per objective)
    - Choose trial with minimal normalized L2 distance to ideal
    """
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
    if len(trials) == 0:
        return None

    vals = np.array([t.values for t in trials], dtype=float)  # shape (n, k)
    mins = np.min(vals, axis=0)
    maxs = np.max(vals, axis=0)
    denom = (maxs - mins)
    denom[denom == 0] = 1.0
    norm = (vals - mins) / denom
    d = np.sqrt(np.sum(norm ** 2, axis=1))
    best_idx = int(np.argmin(d))
    return trials[best_idx]


# ======================================================
# LSTM1 (per country) - مع Early Stopping
# ======================================================
def build_lstm1(trial, window, input_dim):
    units = trial.suggest_int("units_l1", 32, 128)
    lr = trial.suggest_float("lr_l1", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("patience_l1", 3, 10)

    model = tf.keras.Sequential([
        layers.LSTM(units, input_shape=(window, input_dim)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer=Adam(lr), loss="mse")
    return model, patience

def objective_lstm1(trial, X, y, window):
    """
    PARETO objective (2 أهداف):
    - MAE على جزء validation
    - MSE على جزء validation
    """
    m, patience = build_lstm1(trial, window, X.shape[2])

    n = len(X)
    cut = int(n * 0.8)
    if cut < 10 or (n - cut) < 10:
        cut = max(1, n - 1)

    X_tr, y_tr = X[:cut], y[:cut]
    X_va, y_va = X[cut:], y[cut:]

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = m.fit(
        X_tr, y_tr,
        epochs=L1_MAX_EPOCHS,
        batch_size=32,
        validation_data=(X_va, y_va),
        callbacks=[early_stopping],
        verbose=0
    )

    final_epoch = len(history.history['loss'])
    print(f"    LSTM1 - Trial {trial.number}: Final epoch = {final_epoch}")
    trial.set_user_attr("effective_epochs", final_epoch)

    pred_va = m.predict(X_va, verbose=0).flatten()
    mae = float(mean_absolute_error(y_va, pred_va)) if len(y_va) else np.nan
    mse = float(np.mean((y_va - pred_va) ** 2)) if len(y_va) else np.nan
    return mae, mse


# ======================================================
# META MODELS (base architectures) - مع Early Stopping
# ======================================================
# 1) META LSTM (seq)
def build_meta_lstm(trial, window):
    units = trial.suggest_int("meta_lstm_units", 32, 128)
    dense_units = trial.suggest_int("meta_lstm_dense", 16, 64)
    lr = trial.suggest_float("meta_lstm_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("meta_lstm_patience", 3, 10)

    inp = layers.Input(shape=(window, 1))
    x = layers.LSTM(units)(inp)
    x = layers.Dense(dense_units, activation="relu")(x)
    out = layers.Dense(1)(x)

    m = Model(inp, out)
    m.compile(optimizer=Adam(lr), loss="mse")
    return m, patience

def objective_meta_lstm(trial, Xs, y, window):
    m, patience = build_meta_lstm(trial, window)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = m.fit(
        Xs, y,
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    final_epoch = len(history.history['loss'])
    print(f"    META LSTM - Trial {trial.number}: Final epoch = {final_epoch}")
    trial.set_user_attr("effective_epochs", final_epoch)

    pred = m.predict(Xs, verbose=0).flatten()
    return mean_absolute_error(y, pred)


# 2) META ATT (seq + attention)
def build_meta_att(trial, window):
    units = trial.suggest_int("meta_att_units", 32, 128)
    dense_units = trial.suggest_int("meta_att_dense", 16, 64)
    lr = trial.suggest_float("meta_att_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("meta_att_patience", 3, 10)

    seq_in = layers.Input(shape=(window, 1))
    x = layers.LSTM(units, return_sequences=True)(seq_in)
    x = Attention()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    out = layers.Dense(1)(x)

    m = Model(seq_in, out)
    m.compile(optimizer=Adam(lr), loss="mse")
    return m, patience

def objective_meta_att(trial, Xs, y, window):
    m, patience = build_meta_att(trial, window)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = m.fit(
        Xs, y,
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    final_epoch = len(history.history['loss'])
    print(f"    META ATT-LSTM - Trial {trial.number}: Final epoch = {final_epoch}")
    trial.set_user_attr("effective_epochs", final_epoch)

    pred = m.predict(Xs, verbose=0).flatten()
    return mean_absolute_error(y, pred)


# 3) META ATT-BiLSTM (seq + attention + bidirectional)
def build_meta_att_bilstm(trial, window):
    units = trial.suggest_int("meta_att_bilstm_units", 32, 128)
    dense_units = trial.suggest_int("meta_att_bilstm_dense", 16, 64)
    lr = trial.suggest_float("meta_att_bilstm_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("meta_att_bilstm_patience", 3, 10)

    seq_in = layers.Input(shape=(window, 1))
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(seq_in)
    x = Attention()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    out = layers.Dense(1)(x)

    m = Model(seq_in, out)
    m.compile(optimizer=Adam(lr), loss="mse")
    return m, patience

def objective_meta_att_bilstm(trial, Xs, y, window):
    m, patience = build_meta_att_bilstm(trial, window)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = m.fit(
        Xs, y,
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    final_epoch = len(history.history['loss'])
    print(f"    META ATT-BiLSTM - Trial {trial.number}: Final epoch = {final_epoch}")
    trial.set_user_attr("effective_epochs", final_epoch)

    pred = m.predict(Xs, verbose=0).flatten()
    return mean_absolute_error(y, pred)


# 4) META LSTM+STAT (seq + static window stats)
def build_meta_lstm_stat(trial, window, stats_dim):
    units = trial.suggest_int("meta_lstm_stat_units", 32, 128)
    dense_units = trial.suggest_int("meta_lstm_stat_dense", 16, 64)
    lr = trial.suggest_float("meta_lstm_stat_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("meta_lstm_stat_patience", 3, 10)

    seq_in = layers.Input(shape=(window, 1))
    stat_in = layers.Input(shape=(stats_dim,))

    x_seq = layers.LSTM(units)(seq_in)
    x_stat = layers.Dense(32, activation="relu")(stat_in)

    x = layers.Concatenate()([x_seq, x_stat])
    x = layers.Dense(dense_units, activation="relu")(x)
    out = layers.Dense(1)(x)

    m = Model([seq_in, stat_in], out)
    m.compile(optimizer=Adam(lr), loss="mse")
    return m, patience

def objective_meta_lstm_stat(trial, Xs, Xstat, y, window, stats_dim):
    m, patience = build_meta_lstm_stat(trial, window, stats_dim)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = m.fit(
        [Xs, Xstat], y,
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    final_epoch = len(history.history['loss'])
    print(f"    META LSTM+STAT - Trial {trial.number}: Final epoch = {final_epoch}")
    trial.set_user_attr("effective_epochs", final_epoch)

    pred = m.predict([Xs, Xstat], verbose=0).flatten()
    return mean_absolute_error(y, pred)


# 5) RESIDUAL MODEL for ATT-BiLSTM (predict residual = y_true - base_pred)
def build_meta_residual_lstm_for_attbilstm(trial, window):
    units = trial.suggest_int("meta_att_bilstm_resid_units", 32, 128)
    dense_units = trial.suggest_int("meta_att_bilstm_resid_dense", 16, 64)
    lr = trial.suggest_float("meta_att_bilstm_resid_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("meta_att_bilstm_resid_patience", 3, 10)

    inp = layers.Input(shape=(window, 1))
    x = layers.LSTM(units)(inp)
    x = layers.Dense(dense_units, activation="relu")(x)
    out = layers.Dense(1)(x)

    m = Model(inp, out)
    m.compile(optimizer=Adam(lr), loss="mse")
    return m, patience

def objective_meta_attbilstm_residual(trial, Xs, y_resid, window):
    m, patience = build_meta_residual_lstm_for_attbilstm(trial, window)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = m.fit(
        Xs, y_resid,
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    final_epoch = len(history.history['loss'])
    print(f"    META ATT-BiLSTM RESID - Trial {trial.number}: Final epoch = {final_epoch}")
    trial.set_user_attr("effective_epochs", final_epoch)

    pred = m.predict(Xs, verbose=0).flatten()
    return mean_absolute_error(y_resid, pred)


# ======================================================
# LOAD DATA
# ======================================================
print(">>> Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["country"])
df["country"] = df["country"].astype(str).str.strip()
df = df[df["country"] != ""].dropna()

if "Date" not in df.columns:
    raise RuntimeError("Column 'Date' is required in dataset.")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

if "tavg" not in df.columns:
    raise RuntimeError("Column 'tavg' is required as target but not found in dataset.")

FEATURES = [
    "tavg","tmin","tmax",
    "Temp_Max","Temp_Mean","Temp_Min",
    "wspd","wgust","Windspeed_Max","Windgusts_Max",
    "sunshine","Sunshine_Duration",
    "prcp","Precipitation_Sum"
]
FEATURES = [c for c in FEATURES if c in df.columns]
print(">>> Using FEATURES:", FEATURES)

countries = df["country"].unique().tolist()
print(f">>> Countries found: {len(countries)}")
print(f">>> WINDOW fixed = {WINDOW}")


# ======================================================
# STORAGE / POOLS
# ======================================================
X_pool_seq = []   # pooled (WINDOW,1) from tnorm
X_pool_stat = []  # pooled stats (8,)
y_pool = {m: [] for m in META_METRICS}

splits = {}  # per country: 75–90 and 90–100 sets (seq+stat + y)
global_true_errors_35_90 = {m: [] for m in META_METRICS}

# Optuna result tables
optuna_rows_meta = []
lstm1_best_rows = []

# LSTM1 plots (optional): store small subset
plot_lstm1 = []
first_lstm1_history = None
first_lstm1_country = None


# ======================================================
# STEP A: TRAIN LSTM1 PER COUNTRY + COLLECT TRUE ERRORS
# ======================================================
print("\n=== STEP A: LSTM1 PER COUNTRY + TRUE ERRORS ===")

for country in countries:
    dfc = df[df["country"] == country].reset_index(drop=True)
    if dfc.shape[0] < WINDOW + 60:
        continue

    data = dfc[FEATURES].values.astype(float)
    tavg = dfc["tavg"].values.astype(float)
    N = len(data)

    idx35 = int(N * P35)
    if idx35 < WINDOW + 5:
        continue

    scaler = StandardScaler()
    scaler.fit(data[:idx35])
    norm = scaler.transform(data)

    # Build windows for LSTM1
    X1, y1 = [], []
    for i in range(len(norm) - WINDOW):
        X1.append(norm[i:i+WINDOW])
        y1.append(norm[i+WINDOW, 0])  # predicting normalized tavg
    X1 = np.array(X1, dtype=float)
    y1 = np.array(y1, dtype=float)
    nW = len(X1)
    if nW < 60:
        continue

    w35 = int(nW * P35)
    w50 = int(nW * P50)
    w75 = int(nW * P75)
    w90 = int(nW * P90)

    # Tune LSTM1 on 0–35 (PARETO)
    print(f"  Tuning LSTM1 for {country} (PARETO)...")
    st1 = optuna.create_study(directions=["minimize", "minimize"])
    st1.optimize(lambda tr: objective_lstm1(tr, X1[:w35], y1[:w35], WINDOW), n_trials=N_TRIALS_L1)
    save_optuna_trials(st1, f"optuna_trials_LSTM1_{country}.csv")

    best_trial = select_best_pareto_trial(st1)
    if best_trial is None:
        continue

    lstm1_best_rows.append({
        "Country": country,
        "BestValues_PARETO_[MAE,MSE]": json.dumps([float(best_trial.values[0]), float(best_trial.values[1])]),
        "BestParamsJSON": json.dumps(best_trial.params, ensure_ascii=False),
        "WINDOW": WINDOW,
        "EffectiveEpochs": best_trial.user_attrs.get("effective_epochs", "N/A")
    })

    # Fit best LSTM1 with early stopping
    print(f"  Training best LSTM1 for {country}...")
    m1, patience = build_lstm1(best_trial, WINDOW, X1.shape[2])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    hist = m1.fit(
        X1[:w35], y1[:w35],
        epochs=L1_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    final_epoch = len(hist.history['loss'])
    print(f"  Final epoch for {country}: {final_epoch}")

    if first_lstm1_history is None:
        first_lstm1_history = hist.history.get("loss", None)
        first_lstm1_country = country

    # quick plot data for 35–50 (optional)
    try:
        pn = m1.predict(X1[w35:w50], verbose=0).flatten()
        rp = pn * scaler.scale_[0] + scaler.mean_[0]
        rt = tavg[WINDOW+w35:WINDOW+w50]
        if len(rt) == len(rp) and len(rt) > 0:
            plot_lstm1.append((country, rt, rp))
    except Exception:
        pass

    # Predict all windows
    all_p = m1.predict(X1, verbose=0).flatten()
    all_raw = all_p * scaler.scale_[0] + scaler.mean_[0]
    all_true = tavg[WINDOW:WINDOW+nW]  # aligned

    # True errors per window (targets for meta)
    err_smape = np.array([smape_scalar(all_true[i], all_raw[i]) for i in range(nW)], dtype=float)
    err_mae   = np.array([abs(all_true[i] - all_raw[i]) for i in range(nW)], dtype=float)
    err_mase  = np.array([mase_scalar(all_true[i], all_raw[i], tavg[:WINDOW+i]) for i in range(nW)], dtype=float)

    # collect supervisor baseline values from 35–90
    global_true_errors_35_90["sMAPE"].extend(err_smape[w35:w90].tolist())
    global_true_errors_35_90["MAE"].extend(err_mae[w35:w90].tolist())
    global_true_errors_35_90["MASE"].extend(err_mase[w35:w90].tolist())

    # meta input series: tnorm = normalized tavg
    tnorm = norm[:, 0].astype(float)

    # pooled meta training (35–75)
    for i in range(w35, w75):
        X_pool_seq.append(tnorm[i:i+WINDOW].reshape(WINDOW, 1))
        X_pool_stat.append(window_stats(tnorm[i:i+WINDOW]))
        y_pool["sMAPE"].append(err_smape[i])
        y_pool["MAE"].append(err_mae[i])
        y_pool["MASE"].append(err_mase[i])

    # save splits for later prediction
    splits[country] = {
        "75_90_seq": np.array([tnorm[i:i+WINDOW].reshape(WINDOW, 1) for i in range(w75, w90)], dtype=float),
        "75_90_stat": np.array([window_stats(tnorm[i:i+WINDOW]) for i in range(w75, w90)], dtype=float),
        "90_100_seq": np.array([tnorm[i:i+WINDOW].reshape(WINDOW, 1) for i in range(w90, nW)], dtype=float),
        "90_100_stat": np.array([window_stats(tnorm[i:i+WINDOW]) for i in range(w90, nW)], dtype=float),
        "75_90_y": {
            "sMAPE": err_smape[w75:w90],
            "MAE": err_mae[w75:w90],
            "MASE": err_mase[w75:w90],
        },
        "90_100_y": {
            "sMAPE": err_smape[w90:nW],
            "MAE": err_mae[w90:nW],
            "MASE": err_mase[w90:nW],
        },
        "nW": nW
    }

# finalize pools
X_pool_seq = np.array(X_pool_seq, dtype=float)
X_pool_stat = np.array(X_pool_stat, dtype=float)
for m in META_METRICS:
    y_pool[m] = np.array(y_pool[m], dtype=float)

if X_pool_seq.shape[0] < 50:
    raise RuntimeError("Meta pool too small. Check dataset size / country filtering.")

stats_dim = X_pool_stat.shape[1]

print(f"=== LSTM1 DONE ===")
print(f"Pooled meta windows: {X_pool_seq.shape[0]}")
print(f"Stats dim: {stats_dim}")
print(f"Countries kept for splits: {len(splits)}")

# Save LSTM1 best table
df_lstm1_best = pd.DataFrame(lstm1_best_rows)
df_lstm1_best_path = os.path.join(OUT_DIR, "OPTUNA_BEST_LSTM1_PER_COUNTRY.csv")
df_lstm1_best.to_csv(df_lstm1_best_path, index=False)
print("Saved:", df_lstm1_best_path)


# ======================================================
# STEP B: SUPERVISOR GLOBAL CONSTANT BASELINE (per metric)
# ======================================================
GLOBAL_BASELINE_VALUE = {}
for m in META_METRICS:
    vals = np.array(global_true_errors_35_90[m], dtype=float)
    if len(vals) == 0:
        raise RuntimeError(f"No baseline values collected for metric {m}.")
    GLOBAL_BASELINE_VALUE[m] = float(np.mean(vals))

print("\n=== SUPERVISOR GLOBAL CONSTANT BASELINE ===")
for m in META_METRICS:
    print(f"Baseline[{m}] = {GLOBAL_BASELINE_VALUE[m]:.6f}")


# ======================================================
# STEP C: TRAIN META MODELS (NOW 5) PER METRIC + PREDICT 75–90 / 90–100
# ======================================================
print("\n=== STEP C: TRAIN META MODELS (5) PER METRIC ===")

# Store predictions:
# res_meta[arch][metric][country] = (y75, p75, y100, p100)
ARCHS = ["LSTM", "ATT", "ATT_BILSTM", "LSTM_STAT", "ATT_BILSTM_RESID"]
res_meta = {a: {m: {} for m in META_METRICS} for a in ARCHS}

# Store trained models
meta_models = {a: {} for a in ARCHS}

# For residual: keep both base and residual per metric
meta_models["ATT_BILSTM_RESID"] = {m: {"base": None, "resid": None} for m in META_METRICS}

for metric in META_METRICS:
    print(f"\n--- Metric: {metric} ---")

    # 1) META LSTM (seq)
    print(f"  Tuning META LSTM for {metric}...")
    st_lstm = optuna.create_study(direction="minimize")
    st_lstm.optimize(lambda tr: objective_meta_lstm(tr, X_pool_seq, y_pool[metric], WINDOW),
                     n_trials=N_TRIALS_META)
    save_optuna_trials(st_lstm, f"optuna_trials_META_LSTM_{metric}.csv")
    optuna_rows_meta.append(pack_best_row("META_LSTM(seq)", metric, st_lstm.best_value, st_lstm.best_trial.params, {
        "WINDOW": WINDOW,
        "EffectiveEpochs": st_lstm.best_trial.user_attrs.get("effective_epochs", "N/A")
    }))

    print(f"  Training best META LSTM for {metric}...")
    m_lstm, patience = build_meta_lstm(st_lstm.best_trial, WINDOW)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = m_lstm.fit(
        X_pool_seq, y_pool[metric],
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    final_epoch = len(history.history['loss'])
    print(f"  Final epoch for META LSTM ({metric}): {final_epoch}")

    meta_models["LSTM"][metric] = m_lstm

    # 2) META ATT (seq + attention)
    print(f"  Tuning META ATT-LSTM for {metric}...")
    st_att = optuna.create_study(direction="minimize")
    st_att.optimize(lambda tr: objective_meta_att(tr, X_pool_seq, y_pool[metric], WINDOW),
                    n_trials=N_TRIALS_META)
    save_optuna_trials(st_att, f"optuna_trials_META_ATT_{metric}.csv")
    optuna_rows_meta.append(pack_best_row("META_ATT(seq+att)", metric, st_att.best_value, st_att.best_trial.params, {
        "WINDOW": WINDOW,
        "EffectiveEpochs": st_att.best_trial.user_attrs.get("effective_epochs", "N/A")
    }))

    print(f"  Training best META ATT-LSTM for {metric}...")
    m_att, patience = build_meta_att(st_att.best_trial, WINDOW)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = m_att.fit(
        X_pool_seq, y_pool[metric],
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    final_epoch = len(history.history['loss'])
    print(f"  Final epoch for META ATT-LSTM ({metric}): {final_epoch}")

    meta_models["ATT"][metric] = m_att

    # 3) META ATT-BiLSTM (seq + attention + bidirectional)
    print(f"  Tuning META ATT-BiLSTM for {metric}...")
    st_att_bilstm = optuna.create_study(direction="minimize")
    st_att_bilstm.optimize(lambda tr: objective_meta_att_bilstm(tr, X_pool_seq, y_pool[metric], WINDOW),
                           n_trials=N_TRIALS_META)
    save_optuna_trials(st_att_bilstm, f"optuna_trials_META_ATT_BILSTM_{metric}.csv")
    optuna_rows_meta.append(pack_best_row("META_ATT_BILSTM(seq+att+bi)", metric, st_att_bilstm.best_value, st_att_bilstm.best_trial.params, {
        "WINDOW": WINDOW,
        "EffectiveEpochs": st_att_bilstm.best_trial.user_attrs.get("effective_epochs", "N/A")
    }))

    print(f"  Training best META ATT-BiLSTM for {metric}...")
    m_att_bilstm, patience = build_meta_att_bilstm(st_att_bilstm.best_trial, WINDOW)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = m_att_bilstm.fit(
        X_pool_seq, y_pool[metric],
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    final_epoch = len(history.history['loss'])
    print(f"  Final epoch for META ATT-BiLSTM ({metric}): {final_epoch}")

    meta_models["ATT_BILSTM"][metric] = m_att_bilstm

    # 4) META LSTM+STAT
    print(f"  Tuning META LSTM+STAT for {metric}...")
    st_lstm_stat = optuna.create_study(direction="minimize")
    st_lstm_stat.optimize(lambda tr: objective_meta_lstm_stat(tr, X_pool_seq, X_pool_stat, y_pool[metric], WINDOW, stats_dim),
                          n_trials=N_TRIALS_META)
    save_optuna_trials(st_lstm_stat, f"optuna_trials_META_LSTM_STAT_{metric}.csv")
    optuna_rows_meta.append(pack_best_row("META_LSTM_STAT(seq+stat)", metric, st_lstm_stat.best_value, st_lstm_stat.best_trial.params, {
        "WINDOW": WINDOW,
        "EffectiveEpochs": st_lstm_stat.best_trial.user_attrs.get("effective_epochs", "N/A")
    }))

    print(f"  Training best META LSTM+STAT for {metric}...")
    m_lstm_stat, patience = build_meta_lstm_stat(st_lstm_stat.best_trial, WINDOW, stats_dim)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = m_lstm_stat.fit(
        [X_pool_seq, X_pool_stat], y_pool[metric],
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    final_epoch = len(history.history['loss'])
    print(f"  Final epoch for META LSTM+STAT ({metric}): {final_epoch}")

    meta_models["LSTM_STAT"][metric] = m_lstm_stat

    # 5) META ATT-BiLSTM + RESIDUAL
    print(f"  Building residual targets for ATT-BiLSTM ({metric})...")
    base_pred_pool = m_att_bilstm.predict(X_pool_seq, verbose=0).flatten()
    y_resid_pool = (y_pool[metric] - base_pred_pool).astype(float)

    print(f"  Tuning META ATT-BiLSTM+RESID for {metric}...")
    st_resid = optuna.create_study(direction="minimize")
    st_resid.optimize(lambda tr: objective_meta_attbilstm_residual(tr, X_pool_seq, y_resid_pool, WINDOW),
                      n_trials=N_TRIALS_META)
    save_optuna_trials(st_resid, f"optuna_trials_META_ATT_BILSTM_RESID_{metric}.csv")
    optuna_rows_meta.append(pack_best_row("META_ATT_BILSTM_RESID(base+resid)", metric, st_resid.best_value, st_resid.best_trial.params, {
        "WINDOW": WINDOW,
        "EffectiveEpochs": st_resid.best_trial.user_attrs.get("effective_epochs", "N/A")
    }))

    print(f"  Training best RESIDUAL model for ATT-BiLSTM ({metric})...")
    m_resid, patience = build_meta_residual_lstm_for_attbilstm(st_resid.best_trial, WINDOW)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = m_resid.fit(
        X_pool_seq, y_resid_pool,
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    final_epoch = len(history.history['loss'])
    print(f"  Final epoch for META ATT-BiLSTM RESID ({metric}): {final_epoch}")

    meta_models["ATT_BILSTM_RESID"][metric]["base"] = m_att_bilstm
    meta_models["ATT_BILSTM_RESID"][metric]["resid"] = m_resid

    # Predict per country 75–90 and 90–100 for all 5 archs
    for c in splits:
        s75 = splits[c]["75_90_seq"]
        st75 = splits[c]["75_90_stat"]
        y75 = splits[c]["75_90_y"][metric]

        s100 = splits[c]["90_100_seq"]
        st100 = splits[c]["90_100_stat"]
        y100 = splits[c]["90_100_y"][metric]

        # LSTM
        p75 = m_lstm.predict(s75, verbose=0).flatten() if len(s75) else np.array([])
        p100 = m_lstm.predict(s100, verbose=0).flatten() if len(s100) else np.array([])
        res_meta["LSTM"][metric][c] = (y75, p75, y100, p100)

        # ATT
        p75a = m_att.predict(s75, verbose=0).flatten() if len(s75) else np.array([])
        p100a = m_att.predict(s100, verbose=0).flatten() if len(s100) else np.array([])
        res_meta["ATT"][metric][c] = (y75, p75a, y100, p100a)

        # ATT-BiLSTM
        p75ab = m_att_bilstm.predict(s75, verbose=0).flatten() if len(s75) else np.array([])
        p100ab = m_att_bilstm.predict(s100, verbose=0).flatten() if len(s100) else np.array([])
        res_meta["ATT_BILSTM"][metric][c] = (y75, p75ab, y100, p100ab)

        # LSTM+STAT
        p75_ls = m_lstm_stat.predict([s75, st75], verbose=0).flatten() if len(s75) else np.array([])
        p100_ls = m_lstm_stat.predict([s100, st100], verbose=0).flatten() if len(s100) else np.array([])
        res_meta["LSTM_STAT"][metric][c] = (y75, p75_ls, y100, p100_ls)

        # ATT-BiLSTM + RESID (final = base + resid)
        base75 = m_att_bilstm.predict(s75, verbose=0).flatten() if len(s75) else np.array([])
        base100 = m_att_bilstm.predict(s100, verbose=0).flatten() if len(s100) else np.array([])
        r75 = m_resid.predict(s75, verbose=0).flatten() if len(s75) else np.array([])
        r100 = m_resid.predict(s100, verbose=0).flatten() if len(s100) else np.array([])
        p75_resid = (base75 + r75).astype(float) if len(base75) else np.array([])
        p100_resid = (base100 + r100).astype(float) if len(base100) else np.array([])
        res_meta["ATT_BILSTM_RESID"][metric][c] = (y75, p75_resid, y100, p100_resid)

# Save Optuna best params for meta models
df_optuna_meta = pd.DataFrame(optuna_rows_meta)
optuna_meta_path = os.path.join(OUT_DIR, "OPTUNA_BEST_PARAMS_META_MODELS.csv")
df_optuna_meta.to_csv(optuna_meta_path, index=False)
print("\nSaved:", optuna_meta_path)


# ======================================================
# STEP D: EVALUATION ON 90–100 (pooled across all countries)
# Models:
# 1 META LSTM
# 2 META ATT
# 3 META ATT-BiLSTM
# 4 META LSTM+STAT
# 5 META ATT-BiLSTM+RESID
# plus the Supervisor baseline constant for SkillScore reference
# ======================================================
print("\n=== STEP D: EVALUATION ON 90–100 ===")

def eval_model(metric, model_name, y_true_all, y_pred_all, baseline_value):
    y_true_all = np.array(y_true_all, dtype=float)
    y_pred_all = np.array(y_pred_all, dtype=float)

    y_base = np.full_like(y_true_all, baseline_value, dtype=float)

    mae_m = mean_absolute_error(y_true_all, y_pred_all) if len(y_true_all) else np.nan
    mae_b = mean_absolute_error(y_true_all, y_base) if len(y_true_all) else np.nan
    evs = explained_variance_score(y_true_all, y_pred_all) if len(y_true_all) else np.nan
    sp = safe_spearman(y_true_all, y_pred_all)
    ss = skill_score_mae(mae_m, mae_b) if (not np.isnan(mae_m) and not np.isnan(mae_b)) else np.nan

    return {
        "Metric": metric,
        "Model": model_name,
        "MAE": mae_m,
        "EVS": evs,
        "Spearman": sp,
        "SkillScore_MAE_vs_GlobalBaseline": ss,
        "GlobalBaselineValue": float(baseline_value),
        "WINDOW": WINDOW
    }

summary_rows = []

for metric in META_METRICS:
    base_val = GLOBAL_BASELINE_VALUE[metric]

    y_all = []
    pred_baseline = []

    preds = {
        "META_LSTM": [],
        "META_ATT": [],
        "META_ATT_BILSTM": [],
        "META_LSTM_STAT": [],
        "META_ATT_BILSTM_RESID": [],
    }

    for c in splits:
        y100 = splits[c]["90_100_y"][metric]
        if len(y100) == 0:
            continue

        y_all.extend(y100.tolist())
        pred_baseline.extend([base_val] * len(y100))

        # Base meta preds (90–100)
        preds["META_LSTM"].extend(res_meta["LSTM"][metric][c][3].tolist())
        preds["META_ATT"].extend(res_meta["ATT"][metric][c][3].tolist())
        preds["META_ATT_BILSTM"].extend(res_meta["ATT_BILSTM"][metric][c][3].tolist())

        preds["META_LSTM_STAT"].extend(res_meta["LSTM_STAT"][metric][c][3].tolist())
        preds["META_ATT_BILSTM_RESID"].extend(res_meta["ATT_BILSTM_RESID"][metric][c][3].tolist())

    # baseline row
    summary_rows.append(eval_model(metric, "Supervisor_GlobalConstantBaseline", y_all, pred_baseline, base_val))

    # model rows
    summary_rows.append(eval_model(metric, "META_LSTM", y_all, preds["META_LSTM"], base_val))
    summary_rows.append(eval_model(metric, "META_ATT", y_all, preds["META_ATT"], base_val))
    summary_rows.append(eval_model(metric, "META_ATT_BILSTM", y_all, preds["META_ATT_BILSTM"], base_val))
    summary_rows.append(eval_model(metric, "META_LSTM_STAT", y_all, preds["META_LSTM_STAT"], base_val))
    summary_rows.append(eval_model(metric, "META_ATT_BILSTM_RESID", y_all, preds["META_ATT_BILSTM_RESID"], base_val))

df_summary = pd.DataFrame(summary_rows)

summary_path = os.path.join(OUT_DIR, "ALL_META_MODEL_SUMMARY_MULTI_METRIC.csv")
df_summary.to_csv(summary_path, index=False)
print("Saved:", summary_path)
print(df_summary)


# ======================================================
# STEP E: EXPORT FULL TIMESERIES (75–100) PER COUNTRY + METRIC
# Include: baseline constant + 5 meta model predictions
# ======================================================
print("\n=== STEP E: EXPORTING FULL TIMESERIES (75–100) ===")

rows = []
for c in splits:
    for metric in META_METRICS:
        y75 = splits[c]["75_90_y"][metric]
        y100 = splits[c]["90_100_y"][metric]
        true_full = np.concatenate([y75, y100]).astype(float)
        L = len(true_full)

        base_val = GLOBAL_BASELINE_VALUE[metric]
        baseline_full = np.full(L, base_val, dtype=float)

        # meta model full predictions (75–100)
        p75_lstm = res_meta["LSTM"][metric][c][1]
        p100_lstm = res_meta["LSTM"][metric][c][3]
        lstm_full = np.concatenate([p75_lstm, p100_lstm]).astype(float)

        p75_att = res_meta["ATT"][metric][c][1]
        p100_att = res_meta["ATT"][metric][c][3]
        att_full = np.concatenate([p75_att, p100_att]).astype(float)

        p75_att_bilstm = res_meta["ATT_BILSTM"][metric][c][1]
        p100_att_bilstm = res_meta["ATT_BILSTM"][metric][c][3]
        att_bilstm_full = np.concatenate([p75_att_bilstm, p100_att_bilstm]).astype(float)

        p75_lstm_stat = res_meta["LSTM_STAT"][metric][c][1]
        p100_lstm_stat = res_meta["LSTM_STAT"][metric][c][3]
        lstm_stat_full = np.concatenate([p75_lstm_stat, p100_lstm_stat]).astype(float)

        p75_att_bilstm_resid = res_meta["ATT_BILSTM_RESID"][metric][c][1]
        p100_att_bilstm_resid = res_meta["ATT_BILSTM_RESID"][metric][c][3]
        att_bilstm_resid_full = np.concatenate([p75_att_bilstm_resid, p100_att_bilstm_resid]).astype(float)

        # align safety
        baseline_full = baseline_full[:L]
        lstm_full = lstm_full[:L]
        att_full = att_full[:L]
        att_bilstm_full = att_bilstm_full[:L]
        lstm_stat_full = lstm_stat_full[:L]
        att_bilstm_resid_full = att_bilstm_resid_full[:L]

        for i in range(L):
            rows.append({
                "country": c,
                "metric": metric,
                "index_75_100": i,
                "true_error": float(true_full[i]),
                "baseline_global_constant": float(baseline_full[i]),
                "meta_lstm": float(lstm_full[i]),
                "meta_att": float(att_full[i]),
                "meta_att_bilstm": float(att_bilstm_full[i]),
                "meta_lstm_stat": float(lstm_stat_full[i]),
                "meta_att_bilstm_resid": float(att_bilstm_resid_full[i]),
                "window": WINDOW
            })

df_ts = pd.DataFrame(rows)
ts_path = os.path.join(OUT_DIR, "MODEL_FULL_TIMESERIES_MULTI_METRIC.csv")
df_ts.to_csv(ts_path, index=False)
print("Saved:", ts_path)


# ======================================================
# STEP F: SAVE OPTUNA TABLES (combined)
# ======================================================
df_optuna_all = pd.DataFrame(optuna_rows_meta)
optuna_all_path = os.path.join(OUT_DIR, "OPTUNA_BEST_PARAMS_ALL_META_MODELS.csv")
df_optuna_all.to_csv(optuna_all_path, index=False)
print("\nSaved:", optuna_all_path)


# ======================================================
# OPTIONAL: SAVE A COUPLE LSTM1 PLOTS
# ======================================================
try:
    import matplotlib.pyplot as plt

    # Save loss plot (first country)
    if first_lstm1_history is not None and first_lstm1_country is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(first_lstm1_history)
        plt.title(f"LSTM1 Training Loss (0–35) — {first_lstm1_country}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "LSTM1_training_loss_example.png"))
        plt.close()

    # Save up to 5 country plots (35–50)
    for k, (c, yt, yp) in enumerate(plot_lstm1[:5]):
        plt.figure(figsize=(10, 5))
        plt.plot(yt, label="True")
        plt.plot(yp, label="Pred")
        plt.title(f"LSTM1 — {c} (35–50)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"LSTM1_{c}.png"))
        plt.close()

except Exception as e:
    print("[warn] Could not save plots:", str(e))

print("\n=== ALL DONE ✓ V13 (LSTM1 PARETO + 5 Meta Models: LSTM, ATT, ATT-BiLSTM, LSTM+STAT, ATT-BiLSTM+RESID) ===")
