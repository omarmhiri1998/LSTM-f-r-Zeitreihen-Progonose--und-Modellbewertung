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
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

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
from tensorflow.keras.regularizers import l2

# ======================================================
# CONFIG
# ======================================================
# DATA_PATH = "daily_weather_formatted_Date (1).csv"   # (REMOVED) -> upload instead
OUT_DIR = "plots_v13_all_models_fixed46_bilstm_residlstm"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# WINDOW will be defined by user input

# Window splits (on windows)
P35 = 0.35
P50 = 0.50
P75 = 0.75
P90 = 0.90

META_METRICS = ["sMAPE", "MAE", "MASE"]

# Budgets
N_TRIALS_L1 = 20
L1_MAX_EPOCHS = 50

N_TRIALS_META = 20
META_MAX_EPOCHS = 50

# Save Optuna trials CSV
SAVE_OPTUNA_TRIALS = True

# --- ONLY CHANGE #1 (Seasonal MASE period) ---
SEASONAL_PERIOD = 365

# Parallel processing settings
MAX_WORKERS = 4
USE_THREADS = True


def setup_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                )
            print(f"GPU memory growth enabled for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")

setup_gpu_memory()

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
        diffs = np.abs(y_train[SEASONAL_PERIOD:] - y_train[:-SEASONAL_PERIOD])
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
# OPTUNA HELPERS
# ======================================================
def save_optuna_trials(study, fname):
    if not SAVE_OPTUNA_TRIALS:
        return
    try:
        df_trials = study.trials_dataframe()
        df_trials.to_csv(os.path.join(OUT_DIR, fname), index=False)
    except Exception as e:
        print(" [warn] Could not save optuna trials:", fname, "->", str(e))

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

# ======================================================
# LSTM1 (per country)
# ======================================================
def build_lstm1(trial, window, input_dim):
    units = trial.suggest_int("units_l1", 32, 128)
    lr = trial.suggest_float("lr_l1", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("patience_l1", 3, 10)

    dropout = trial.suggest_float("dropout_l1", 0.0, 0.5)
    recurrent_dropout = trial.suggest_float("recurrent_dropout_l1", 0.0, 0.5)
    l2_reg = trial.suggest_float("l2_reg_l1", 1e-6, 1e-2, log=True)

    model = tf.keras.Sequential([
        layers.LSTM(
            units,
            input_shape=(window, input_dim),
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg)
        ),
        layers.Dropout(dropout),
        layers.Dense(64, activation="relu", kernel_regularizer=l2(l2_reg)),
        layers.Dense(1, kernel_regularizer=l2(l2_reg))
    ])
    model.compile(optimizer=Adam(lr), loss="mse")
    return model, patience

# ======================================================
# LSTM1 Optuna uses Pareto (multi-objective)
# Objectives: MAE, sMAPE, MASE (on normalized target y)
# ======================================================
def objective_lstm1(trial, X, y, window):
    m, patience = build_lstm1(trial, window, X.shape[2])
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )
    history = m.fit(
        X, y,
        epochs=L1_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    trial.set_user_attr("effective_epochs", len(history.history['loss']))
    pred = m.predict(X, verbose=0).flatten()

    mae = mean_absolute_error(y, pred)
    smape = float(np.mean([smape_scalar(y[i], pred[i]) for i in range(len(y))]))

    denom = float(np.mean(np.abs(np.diff(y))) + 1e-8)
    mase = float(np.mean(np.abs(y - pred) / denom))

    return mae, smape, mase

# ======================================================
# META MODELS (base architectures)
# ======================================================
# 1) META LSTM (seq)
def build_meta_lstm(trial, window):
    units = trial.suggest_int("meta_lstm_units", 32, 128)
    dense_units = trial.suggest_int("meta_lstm_dense", 16, 64)
    lr = trial.suggest_float("meta_lstm_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("meta_lstm_patience", 3, 10)

    dropout = trial.suggest_float("meta_lstm_dropout", 0.0, 0.5)
    recurrent_dropout = trial.suggest_float("meta_lstm_recurrent_dropout", 0.0, 0.5)
    l2_reg = trial.suggest_float("meta_lstm_l2_reg", 1e-6, 1e-2, log=True)

    inp = layers.Input(shape=(window, 1))
    x = layers.LSTM(
        units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    )(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    out = layers.Dense(1, kernel_regularizer=l2(l2_reg))(x)

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
    trial.set_user_attr("effective_epochs", len(history.history['loss']))
    pred = m.predict(Xs, verbose=0).flatten()
    return mean_absolute_error(y, pred)

# 2) META LSTM + STAT (seq + window stats)
def window_stats(arr):
    arr = np.array(arr, dtype=float)
    return np.array([
        np.mean(arr),
        np.std(arr),
        np.min(arr),
        np.max(arr),
        arr[-1] - arr[0],             # trend
        skew(arr),
        kurtosis(arr),
        np.percentile(arr, 75) - np.percentile(arr, 25)  # IQR
    ], dtype=float)

def build_meta_lstm_stat(trial, window, stats_dim):
    units = trial.suggest_int("meta_lstm_stat_units", 32, 128)
    dense_units = trial.suggest_int("meta_lstm_stat_dense", 16, 64)
    lr = trial.suggest_float("meta_lstm_stat_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("meta_lstm_stat_patience", 3, 10)

    dropout = trial.suggest_float("meta_lstm_stat_dropout", 0.0, 0.5)
    recurrent_dropout = trial.suggest_float("meta_lstm_stat_recurrent_dropout", 0.0, 0.5)
    l2_reg = trial.suggest_float("meta_lstm_stat_l2_reg", 1e-6, 1e-2, log=True)

    seq_in = layers.Input(shape=(window, 1), name="seq_in")
    stat_in = layers.Input(shape=(stats_dim,), name="stat_in")

    x = layers.LSTM(
        units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    )(seq_in)
    x = layers.Dropout(dropout)(x)

    z = layers.Dense(32, activation="relu", kernel_regularizer=l2(l2_reg))(stat_in)
    h = layers.Concatenate()([x, z])
    h = layers.Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_reg))(h)
    out = layers.Dense(1, kernel_regularizer=l2(l2_reg))(h)

    m = Model([seq_in, stat_in], out)
    m.compile(optimizer=Adam(lr), loss="mse")
    return m, patience

def objective_meta_lstm_stat(trial, Xs_seq, Xs_stat, y, window, stats_dim):
    m, patience = build_meta_lstm_stat(trial, window, stats_dim)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )
    history = m.fit(
        [Xs_seq, Xs_stat], y,
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    trial.set_user_attr("effective_epochs", len(history.history['loss']))
    pred = m.predict([Xs_seq, Xs_stat], verbose=0).flatten()
    return mean_absolute_error(y, pred)

# 3) META ATT (seq + attention)
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

def build_meta_att(trial, window):
    units = trial.suggest_int("meta_att_units", 32, 128)
    dense_units = trial.suggest_int("meta_att_dense", 16, 64)
    lr = trial.suggest_float("meta_att_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("meta_att_patience", 3, 10)

    dropout = trial.suggest_float("meta_att_dropout", 0.0, 0.5)
    recurrent_dropout = trial.suggest_float("meta_att_recurrent_dropout", 0.0, 0.5)
    l2_reg = trial.suggest_float("meta_att_l2_reg", 1e-6, 1e-2, log=True)

    seq_in = layers.Input(shape=(window, 1))
    x = layers.LSTM(
        units,
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    )(seq_in)
    x = Attention()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    out = layers.Dense(1, kernel_regularizer=l2(l2_reg))(x)

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
    trial.set_user_attr("effective_epochs", len(history.history['loss']))
    pred = m.predict(Xs, verbose=0).flatten()
    return mean_absolute_error(y, pred)

# 4) META ATT-BiLSTM (seq + attention + bidirectional)
def build_meta_att_bilstm(trial, window):
    units = trial.suggest_int("meta_att_bilstm_units", 32, 128)
    dense_units = trial.suggest_int("meta_att_bilstm_dense", 16, 64)
    lr = trial.suggest_float("meta_att_bilstm_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("meta_att_bilstm_patience", 3, 10)

    dropout = trial.suggest_float("meta_att_bilstm_dropout", 0.0, 0.5)
    recurrent_dropout = trial.suggest_float("meta_att_bilstm_recurrent_dropout", 0.0, 0.5)
    l2_reg = trial.suggest_float("meta_att_bilstm_l2_reg", 1e-6, 1e-2, log=True)

    seq_in = layers.Input(shape=(window, 1))
    x = layers.Bidirectional(
        layers.LSTM(
            units,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg)
        )
    )(seq_in)
    x = Attention()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    out = layers.Dense(1, kernel_regularizer=l2(l2_reg))(x)

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
    trial.set_user_attr("effective_epochs", len(history.history['loss']))
    pred = m.predict(Xs, verbose=0).flatten()
    return mean_absolute_error(y, pred)

# ======================================================
# RESIDUAL-LSTM = SAME STYLE AS META_LSTM
# Target residual = y_true - base_prediction
# ======================================================
def build_meta_residual_lstm(trial, window, prefix="resid_lstm"):
    units = trial.suggest_int(f"{prefix}_units", 32, 128)
    dense_units = trial.suggest_int(f"{prefix}_dense", 16, 64)
    lr = trial.suggest_float(f"{prefix}_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int(f"{prefix}_patience", 3, 10)

    dropout = trial.suggest_float(f"{prefix}_dropout", 0.0, 0.5)
    recurrent_dropout = trial.suggest_float(f"{prefix}_recurrent_dropout", 0.0, 0.5)
    l2_reg = trial.suggest_float(f"{prefix}_l2_reg", 1e-6, 1e-2, log=True)

    inp = layers.Input(shape=(window, 1))
    x = layers.LSTM(
        units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    )(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    out = layers.Dense(1, kernel_regularizer=l2(l2_reg))(x)

    m = Model(inp, out)
    m.compile(optimizer=Adam(lr), loss="mse")
    return m, patience

def objective_meta_residual_lstm(trial, Xs, resid_y, window, prefix="resid_lstm"):
    m, patience = build_meta_residual_lstm(trial, window, prefix=prefix)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )
    history = m.fit(
        Xs, resid_y,
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    trial.set_user_attr("effective_epochs", len(history.history['loss']))
    pred = m.predict(Xs, verbose=0).flatten()
    return mean_absolute_error(resid_y, pred)

# ==========================LOAD DATA============================

import tkinter as tk
from tkinter import filedialog

# hide main window
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select your CSV file",
    filetypes=[("CSV files", "*.csv")]
)

if not file_path:
    raise RuntimeError("No file selected.")

print(">>> Loading:", file_path)
df = pd.read_csv(file_path)

print("\n>>> Columns found:", df.columns.tolist())

# ======= التغيير اللي اتفقنا عليه: GROUP_COL بدل country ثابت =======
GROUP_COL = input(">>> Enter GROUP column (country/city/firma/...): ").strip()
if GROUP_COL not in df.columns:
    raise RuntimeError(f"Group column '{GROUP_COL}' not found in dataset.")

# تنظيف الـ GROUP_COL
df = df.dropna(subset=[GROUP_COL])
df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip()
df = df[df[GROUP_COL] != ""].dropna()

if "Date" not in df.columns:
    raise RuntimeError("Column 'Date' is required in dataset.")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

TARGET_COL = input(">>> Enter TARGET column (example: tavg): ").strip()
if TARGET_COL not in df.columns:
    raise RuntimeError(f"Target column '{TARGET_COL}' not found in dataset.")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET_COL not in numeric_cols:
    raise RuntimeError(f"Target column '{TARGET_COL}' must be numeric.")

FEATURES = [TARGET_COL] + [c for c in numeric_cols if c != TARGET_COL]

print(">>> GROUP_COL:", GROUP_COL)
print(">>> TARGET_COL:", TARGET_COL)
print(">>> Using FEATURES (target first):", FEATURES)

countries = df[GROUP_COL].unique().tolist()
# ===== Safe USER-DEFINED WINDOW =====
WINDOW_INPUT = input(">>> Enter WINDOW size (e.g., 180): ").strip()

if not WINDOW_INPUT.isdigit():
    raise RuntimeError("WINDOW must be a positive integer.")

WINDOW = int(WINDOW_INPUT)

if WINDOW < 30:
    raise RuntimeError("WINDOW too small. Use at least 30.")

# Compute smallest group length
min_group_length = min(len(df[df[GROUP_COL] == c]) for c in countries)

if WINDOW >= min_group_length:
    raise RuntimeError(
        f"WINDOW={WINDOW} is too large. "
        f"Smallest group has only {min_group_length} rows."
    )

print(f">>> Using WINDOW = {WINDOW}")
print(f">>> Smallest group length = {min_group_length}")

print(f">>> Groups found: {len(countries)}")


# ======================================================
# STORAGE / POOLS
# ======================================================
X_pool_seq = []
X_pool_stat = []
y_pool = {m: [] for m in META_METRICS}
splits = {}
global_true_errors_35_90 = {m: [] for m in META_METRICS}

optuna_rows_meta = []
lstm1_best_rows = []

plot_lstm1 = []
first_lstm1_history = None
first_lstm1_country = None

# ===========================STEP A: TRAIN LSTM1 PER COUNTRY + COLLECT TRUE ERRORS===========================

print(f"\n=== STEP A: LSTM1 PER GROUP + TRUE ERRORS (Parallel - Max Workers: {MAX_WORKERS}) ===")

def process_country_lstm1(country):
    print(f" Starting LSTM1 for {country}...")
    dfc = df[df[GROUP_COL] == country].reset_index(drop=True)  # <-- التغيير هنا فقط
    if dfc.shape[0] < WINDOW + 60:
        print(f" Skipping {country}: insufficient data")
        return None, [], [], [], None, None, None

    data = dfc[FEATURES].values.astype(float)
    target_raw = dfc[TARGET_COL].values.astype(float)

    N = len(data)

    idx35 = int(N * P35)
    if idx35 < WINDOW + 5:
        print(f" Skipping {country}: insufficient data after split")
        return None, [], [], [], None, None, None

    scaler = StandardScaler()
    scaler.fit(data[:idx35])
    norm = scaler.transform(data)

    X1, y1 = [], []
    for i in range(len(norm) - WINDOW):
        X1.append(norm[i:i+WINDOW])
        y1.append(norm[i+WINDOW, 0])  # target must be col 0
    X1 = np.array(X1, dtype=float)
    y1 = np.array(y1, dtype=float)

    nW = len(X1)
    if nW < 60:
        print(f" Skipping {country}: insufficient windows")
        return None, [], [], [], None, None, None

    w35 = int(nW * P35)
    w50 = int(nW * P50)
    w75 = int(nW * P75)
    w90 = int(nW * P90)

    print(f" Tuning LSTM1 (PARETO) for {country}...")
    st1 = optuna.create_study(directions=["minimize", "minimize", "minimize"])
    st1.optimize(lambda tr: objective_lstm1(tr, X1[:w35], y1[:w35], WINDOW), n_trials=N_TRIALS_L1)
    save_optuna_trials(st1, f"optuna_trials_LSTM1_{country}.csv")

    pareto_trials = st1.best_trials
    mae_min = min(t.values[0] for t in pareto_trials)
    smape_min = min(t.values[1] for t in pareto_trials)
    mase_min = min(t.values[2] for t in pareto_trials)

    def distance_to_ideal(t):
        return (
            (t.values[0] - mae_min) ** 2 +
            (t.values[1] - smape_min) ** 2 +
            (t.values[2] - mase_min) ** 2
        )

    best_trial = min(pareto_trials, key=distance_to_ideal)

    row_data = {
        "Country": country,  # (نترك الاسم كما هو حتى ما نغير exports)
        "BestValue_MAE_train": float(best_trial.values[0]),
        "BestValue_sMAPE_train": float(best_trial.values[1]),
        "BestValue_MASE_train": float(best_trial.values[2]),
        "BestParamsJSON": json.dumps(best_trial.params, ensure_ascii=False),
        "WINDOW": WINDOW,
        "EffectiveEpochs": best_trial.user_attrs.get("effective_epochs", "N/A")
    }

    print(f" Training best LSTM1 for {country}...")
    m1, patience = build_lstm1(best_trial, WINDOW, X1.shape[2])
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0)
    hist = m1.fit(
        X1[:w35], y1[:w35],
        epochs=L1_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    print(f" Final epoch for {country}: {len(hist.history['loss'])}")

    plot_data = None
    try:
        pn = m1.predict(X1[w35:w50], verbose=0).flatten()
        rp = pn * scaler.scale_[0] + scaler.mean_[0]
        rt = target_raw[WINDOW+w35:WINDOW+w50]
        if len(rt) == len(rp) and len(rt) > 0:
            plot_data = (country, rt, rp)
    except Exception as e:
        print(f" Warning: Could not generate plot data for {country}: {e}")

    all_p = m1.predict(X1, verbose=0).flatten()
    all_raw = all_p * scaler.scale_[0] + scaler.mean_[0]
    all_true = target_raw[WINDOW:WINDOW+nW]

    err_smape = np.array([smape_scalar(all_true[i], all_raw[i]) for i in range(nW)], dtype=float)
    err_mae = np.array([abs(all_true[i] - all_raw[i]) for i in range(nW)], dtype=float)
    err_mase = np.array([mase_scalar(all_true[i], all_raw[i], target_raw[:WINDOW+i]) for i in range(nW)], dtype=float)

    tnorm = norm[:, 0].astype(float)

    seq_data = []
    stat_data = []
    y_data = {m: [] for m in META_METRICS}
    for i in range(w35, w75):
        seq_data.append(tnorm[i:i+WINDOW].reshape(WINDOW, 1))
        stat_data.append(window_stats(tnorm[i:i+WINDOW]))
        y_data["sMAPE"].append(err_smape[i])
        y_data["MAE"].append(err_mae[i])
        y_data["MASE"].append(err_mase[i])

    split_data = {
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

    baseline_errors = {
        "sMAPE": err_smape[w35:w90].tolist(),
        "MAE": err_mae[w35:w90].tolist(),
        "MASE": err_mase[w35:w90].tolist()
    }

    print(f" ✓ Completed LSTM1 for {country}")
    return row_data, seq_data, stat_data, y_data, split_data, baseline_errors, plot_data

print(f"Processing {len(countries)} groups in parallel (max {MAX_WORKERS} at a time)...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_country_lstm1, c): c for c in countries}

    completed = 0
    for future in concurrent.futures.as_completed(futures):
        country = futures[future]
        completed += 1
        try:
            result = future.result()
            if result[0] is None:
                print(f" [{completed}/{len(countries)}] Skipped {country}")
                continue

            row_data, seq_data, stat_data, y_data, split_data, baseline_errors, plot_data = result

            lstm1_best_rows.append(row_data)
            if plot_data:
                plot_lstm1.append(plot_data)

            cname = row_data["Country"]
            splits[cname] = split_data

            X_pool_seq.extend(seq_data)
            X_pool_stat.extend(stat_data)
            for m in META_METRICS:
                y_pool[m].extend(y_data[m])
                global_true_errors_35_90[m].extend(baseline_errors[m])

            print(f" [{completed}/{len(countries)}] Successfully processed {cname}")
        except Exception as e:
            print(f" [{completed}/{len(countries)}] Error processing {country}: {str(e)}")

X_pool_seq = np.array(X_pool_seq, dtype=float)
X_pool_stat = np.array(X_pool_stat, dtype=float)
for m in META_METRICS:
    y_pool[m] = np.array(y_pool[m], dtype=float)

if X_pool_seq.shape[0] < 50:
    raise RuntimeError("Meta pool too small. Check dataset size / filtering.")

stats_dim = X_pool_stat.shape[1]
print(f"=== LSTM1 DONE ===")
print(f"Pooled meta windows: {X_pool_seq.shape[0]}")
print(f"Stats dim: {stats_dim}")
print(f"Groups kept for splits: {len(splits)}")

df_lstm1_best = pd.DataFrame(lstm1_best_rows)
df_lstm1_best_path = os.path.join(OUT_DIR, "OPTUNA_BEST_LSTM1_PER_COUNTRY.csv")
df_lstm1_best.to_csv(df_lstm1_best_path, index=False)
print("Saved:", df_lstm1_best_path)

# ========================STEP B: SUPERVISOR GLOBAL CONSTANT BASELINE==============================

GLOBAL_BASELINE_VALUE = {}
for m in META_METRICS:
    vals = np.array(global_true_errors_35_90[m], dtype=float)
    if len(vals) == 0:
        raise RuntimeError(f"No baseline values collected for metric {m}.")
    GLOBAL_BASELINE_VALUE[m] = float(np.mean(vals))

print("\n=== SUPERVISOR GLOBAL CONSTANT BASELINE ===")
for m in META_METRICS:
    print(f"Baseline[{m}] = {GLOBAL_BASELINE_VALUE[m]:.6f}")

# ===========================STEP C: TRAIN META MODELS==========================

print(f"\n=== STEP C: TRAIN META MODELS (Parallel - Max Workers: {MAX_WORKERS}) ===")

ARCHS = ["LSTM", "LSTM_STAT", "ATT", "ATT_BILSTM", "ATT_RESID", "ATT_BILSTM_RESID"]
res_meta = {a: {m: {} for m in META_METRICS} for a in ARCHS}
meta_models = {a: {} for a in ARCHS}

def train_meta_model_for_metric_and_arch(metric, arch_name):
    print(f" Starting {arch_name} for {metric}...")

    if arch_name == "LSTM":
        st_model = optuna.create_study(direction="minimize")
        st_model.optimize(lambda tr: objective_meta_lstm(tr, X_pool_seq, y_pool[metric], WINDOW), n_trials=N_TRIALS_META)
        save_optuna_trials(st_model, f"optuna_trials_META_{arch_name}_{metric}.csv")
        model, patience = build_meta_lstm(st_model.best_trial, WINDOW)
        row = pack_best_row(f"META_{arch_name}(seq)", metric, st_model.best_value, st_model.best_trial.params,
                            {"WINDOW": WINDOW, "EffectiveEpochs": st_model.best_trial.user_attrs.get("effective_epochs", "N/A")})
        X_train = X_pool_seq

    elif arch_name == "ATT":
        st_model = optuna.create_study(direction="minimize")
        st_model.optimize(lambda tr: objective_meta_att(tr, X_pool_seq, y_pool[metric], WINDOW), n_trials=N_TRIALS_META)
        save_optuna_trials(st_model, f"optuna_trials_META_{arch_name}_{metric}.csv")
        model, patience = build_meta_att(st_model.best_trial, WINDOW)
        row = pack_best_row(f"META_{arch_name}(seq+att)", metric, st_model.best_value, st_model.best_trial.params,
                            {"WINDOW": WINDOW, "EffectiveEpochs": st_model.best_trial.user_attrs.get("effective_epochs", "N/A")})
        X_train = X_pool_seq

    elif arch_name == "ATT_BILSTM":
        st_model = optuna.create_study(direction="minimize")
        st_model.optimize(lambda tr: objective_meta_att_bilstm(tr, X_pool_seq, y_pool[metric], WINDOW), n_trials=N_TRIALS_META)
        save_optuna_trials(st_model, f"optuna_trials_META_{arch_name}_{metric}.csv")
        model, patience = build_meta_att_bilstm(st_model.best_trial, WINDOW)
        row = pack_best_row(f"META_{arch_name}(seq+att+bi)", metric, st_model.best_value, st_model.best_trial.params,
                            {"WINDOW": WINDOW, "EffectiveEpochs": st_model.best_trial.user_attrs.get("effective_epochs", "N/A")})
        X_train = X_pool_seq

    elif arch_name == "LSTM_STAT":
        st_model = optuna.create_study(direction="minimize")
        st_model.optimize(lambda tr: objective_meta_lstm_stat(tr, X_pool_seq, X_pool_stat, y_pool[metric], WINDOW, stats_dim),
                          n_trials=N_TRIALS_META)
        save_optuna_trials(st_model, f"optuna_trials_META_{arch_name}_{metric}.csv")
        model, patience = build_meta_lstm_stat(st_model.best_trial, WINDOW, stats_dim)
        row = pack_best_row(f"META_{arch_name}(seq+stat)", metric, st_model.best_value, st_model.best_trial.params,
                            {"WINDOW": WINDOW, "EffectiveEpochs": st_model.best_trial.user_attrs.get("effective_epochs", "N/A")})
        X_train = [X_pool_seq, X_pool_stat]

    elif arch_name in ["ATT_RESID", "ATT_BILSTM_RESID"]:
        if arch_name == "ATT_RESID":
            st_base = optuna.create_study(direction="minimize")
            st_base.optimize(lambda tr: objective_meta_att(tr, X_pool_seq, y_pool[metric], WINDOW), n_trials=N_TRIALS_META)
            save_optuna_trials(st_base, f"optuna_trials_META_ATT_BASE_for_RESID_{metric}.csv")
            base_model, base_pat = build_meta_att(st_base.best_trial, WINDOW)
            base_row = pack_best_row("META_ATT_BASE(for_residual)", metric, st_base.best_value, st_base.best_trial.params,
                                    {"WINDOW": WINDOW, "EffectiveEpochs": st_base.best_trial.user_attrs.get("effective_epochs", "N/A")})
            resid_prefix = "resid_att_lstm"
        else:
            st_base = optuna.create_study(direction="minimize")
            st_base.optimize(lambda tr: objective_meta_att_bilstm(tr, X_pool_seq, y_pool[metric], WINDOW), n_trials=N_TRIALS_META)
            save_optuna_trials(st_base, f"optuna_trials_META_ATT_BILSTM_BASE_for_RESID_{metric}.csv")
            base_model, base_pat = build_meta_att_bilstm(st_base.best_trial, WINDOW)
            base_row = pack_best_row("META_ATT_BILSTM_BASE(for_residual)", metric, st_base.best_value, st_base.best_trial.params,
                                    {"WINDOW": WINDOW, "EffectiveEpochs": st_base.best_trial.user_attrs.get("effective_epochs", "N/A")})
            resid_prefix = "resid_att_bilstm_lstm"

        base_es = EarlyStopping(monitor='val_loss', patience=base_pat, restore_best_weights=True, verbose=0)
        base_model.fit(
            X_pool_seq, y_pool[metric],
            epochs=META_MAX_EPOCHS,
            batch_size=32,
            validation_split=0.2,
            callbacks=[base_es],
            verbose=0
        )

        base_pred_train = base_model.predict(X_pool_seq, verbose=0).flatten()
        resid_target = (y_pool[metric].astype(float) - base_pred_train).astype(float)

        st_resid = optuna.create_study(direction="minimize")
        st_resid.optimize(lambda tr: objective_meta_residual_lstm(tr, X_pool_seq, resid_target, WINDOW, prefix=resid_prefix),
                          n_trials=N_TRIALS_META)
        save_optuna_trials(st_resid, f"optuna_trials_META_{arch_name}_RESID_{metric}.csv")

        resid_model, resid_pat = build_meta_residual_lstm(st_resid.best_trial, WINDOW, prefix=resid_prefix)
        resid_es = EarlyStopping(monitor='val_loss', patience=resid_pat, restore_best_weights=True, verbose=0)
        resid_model.fit(
            X_pool_seq, resid_target,
            epochs=META_MAX_EPOCHS,
            batch_size=32,
            validation_split=0.2,
            callbacks=[resid_es],
            verbose=0
        )

        optuna_rows_meta.append(base_row)
        resid_row = pack_best_row(
            f"META_{arch_name}(residual_lstm_optuna_same_style)",
            metric,
            st_resid.best_value,
            st_resid.best_trial.params,
            {"WINDOW": WINDOW, "EffectiveEpochs": st_resid.best_trial.user_attrs.get("effective_epochs", "N/A"),
             "BaseModel": "ATT" if arch_name == "ATT_RESID" else "ATT_BILSTM"}
        )

        country_predictions = {}
        for c in splits:
            s75 = splits[c]["75_90_seq"]
            s100 = splits[c]["90_100_seq"]
            y75 = splits[c]["75_90_y"][metric]
            y100 = splits[c]["90_100_y"][metric]

            if len(s75):
                p75_base = base_model.predict(s75, verbose=0).flatten()
                p75_res = resid_model.predict(s75, verbose=0).flatten()
                p75 = p75_base + p75_res
            else:
                p75 = np.array([])

            if len(s100):
                p100_base = base_model.predict(s100, verbose=0).flatten()
                p100_res = resid_model.predict(s100, verbose=0).flatten()
                p100 = p100_base + p100_res
            else:
                p100 = np.array([])

            country_predictions[c] = (y75, p75, y100, p100)

        model = {"base": base_model, "residual": resid_model}
        print(f" ✓ Completed {arch_name} for {metric}")
        return arch_name, metric, model, resid_row, country_predictions

    else:
        raise RuntimeError(f"Unknown arch_name: {arch_name}")

    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0)
    model.fit(
        X_train, y_pool[metric],
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    country_predictions = {}
    for c in splits:
        s75_seq = splits[c]["75_90_seq"]
        s100_seq = splits[c]["90_100_seq"]
        s75_stat = splits[c]["75_90_stat"]
        s100_stat = splits[c]["90_100_stat"]
        y75 = splits[c]["75_90_y"][metric]
        y100 = splits[c]["90_100_y"][metric]

        if arch_name == "LSTM_STAT":
            p75 = model.predict([s75_seq, s75_stat], verbose=0).flatten() if len(s75_seq) else np.array([])
            p100 = model.predict([s100_seq, s100_stat], verbose=0).flatten() if len(s100_seq) else np.array([])
        else:
            p75 = model.predict(s75_seq, verbose=0).flatten() if len(s75_seq) else np.array([])
            p100 = model.predict(s100_seq, verbose=0).flatten() if len(s100_seq) else np.array([])

        country_predictions[c] = (y75, p75, y100, p100)

    print(f" ✓ Completed {arch_name} for {metric}")
    return arch_name, metric, model, row, country_predictions

print(f"Training {len(META_METRICS)} metrics × {len(ARCHS)} models = {len(META_METRICS)*len(ARCHS)} total models")

meta_tasks = [(metric, arch) for metric in META_METRICS for arch in ARCHS]
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(train_meta_model_for_metric_and_arch, metric, arch): (metric, arch)
               for metric, arch in meta_tasks}

    completed = 0
    total_tasks = len(meta_tasks)
    for future in concurrent.futures.as_completed(futures):
        metric, arch = futures[future]
        completed += 1
        try:
            arch_name, metric_name, model, row, country_predictions = future.result()
            meta_models[arch_name][metric_name] = model
            optuna_rows_meta.append(row)
            for country in country_predictions:
                res_meta[arch_name][metric_name][country] = country_predictions[country]
            print(f" [{completed}/{total_tasks}] Successfully trained {arch_name} for {metric_name}")
        except Exception as e:
            print(f" [{completed}/{total_tasks}] Error training {arch} for {metric}: {str(e)}")

df_optuna_meta = pd.DataFrame(optuna_rows_meta)
optuna_meta_path = os.path.join(OUT_DIR, "OPTUNA_BEST_PARAMS_META_MODELS.csv")
df_optuna_meta.to_csv(optuna_meta_path, index=False)
print("\nSaved:", optuna_meta_path)

# ============================STEP D: EVALUATION ON 90–100==========================

print("\n=== STEP D: EVALUATION ON 90–100 ===")

def eval_model(metric, model_name, y_true_all, y_pred_all, baseline_value):
    y_true_all = np.array(y_true_all, dtype=float)
    y_pred_all = np.array(y_pred_all, dtype=float)
    y_base = np.full_like(y_true_all, baseline_value, dtype=float)

    mae_m = mean_absolute_error(y_true_all, y_pred_all) if len(y_true_all) else np.nan
    mae_b = mean_absolute_error(y_true_all, y_base) if len(y_true_all) else np.nan

    sp = safe_spearman(y_true_all, y_pred_all)
    ss = skill_score_mae(mae_m, mae_b) if (not np.isnan(mae_m) and not np.isnan(mae_b)) else np.nan

    return {
        "Metric": metric,
        "Model": model_name,
        "MAE": mae_m,
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
        "META_LSTM_STAT": [],
        "META_ATT": [],
        "META_ATT_BILSTM": [],
        "META_ATT_RESID": [],
        "META_ATT_BILSTM_RESID": [],
    }

    for c in splits:
        y100 = splits[c]["90_100_y"][metric]
        if len(y100) == 0:
            continue
        y_all.extend(y100.tolist())
        pred_baseline.extend([base_val] * len(y100))

        preds["META_LSTM"].extend(res_meta["LSTM"][metric][c][3].tolist())
        preds["META_LSTM_STAT"].extend(res_meta["LSTM_STAT"][metric][c][3].tolist())
        preds["META_ATT"].extend(res_meta["ATT"][metric][c][3].tolist())
        preds["META_ATT_BILSTM"].extend(res_meta["ATT_BILSTM"][metric][c][3].tolist())
        preds["META_ATT_RESID"].extend(res_meta["ATT_RESID"][metric][c][3].tolist())
        preds["META_ATT_BILSTM_RESID"].extend(res_meta["ATT_BILSTM_RESID"][metric][c][3].tolist())

    summary_rows.append(eval_model(metric, "Supervisor_GlobalConstantBaseline", y_all, pred_baseline, base_val))
    summary_rows.append(eval_model(metric, "META_LSTM", y_all, preds["META_LSTM"], base_val))
    summary_rows.append(eval_model(metric, "META_LSTM_STAT", y_all, preds["META_LSTM_STAT"], base_val))
    summary_rows.append(eval_model(metric, "META_ATT", y_all, preds["META_ATT"], base_val))
    summary_rows.append(eval_model(metric, "META_ATT_BILSTM", y_all, preds["META_ATT_BILSTM"], base_val))
    summary_rows.append(eval_model(metric, "META_ATT_RESID", y_all, preds["META_ATT_RESID"], base_val))
    summary_rows.append(eval_model(metric, "META_ATT_BILSTM_RESID", y_all, preds["META_ATT_BILSTM_RESID"], base_val))

df_summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, "ALL_META_MODEL_SUMMARY_MULTI_METRIC.csv")
df_summary.to_csv(summary_path, index=False)
print("Saved:", summary_path)
print(df_summary)

# ============================STEP E: EXPORT FULL TIMESERIES (75–100)==========================

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

        def concat_full(arch_key):
            p75 = res_meta[arch_key][metric][c][1]
            p100 = res_meta[arch_key][metric][c][3]
            return np.concatenate([p75, p100]).astype(float)

        lstm_full = concat_full("LSTM")[:L]
        lstm_stat_full = concat_full("LSTM_STAT")[:L]
        att_full = concat_full("ATT")[:L]
        att_bilstm_full = concat_full("ATT_BILSTM")[:L]
        att_resid_full = concat_full("ATT_RESID")[:L]
        att_bilstm_resid_full = concat_full("ATT_BILSTM_RESID")[:L]

        baseline_full = baseline_full[:L]

        for i in range(L):
            rows.append({
                "country": c,  # نترك اسم العمود كما هو حتى ما نكسر أي تحليل لاحق عندك
                "metric": metric,
                "index_75_100": i,
                "true_error": float(true_full[i]),
                "baseline_global_constant": float(baseline_full[i]),
                "meta_lstm": float(lstm_full[i]),
                "meta_lstm_stat": float(lstm_stat_full[i]),
                "meta_att": float(att_full[i]),
                "meta_att_bilstm": float(att_bilstm_full[i]),
                "meta_att_residual": float(att_resid_full[i]),
                "meta_att_bilstm_residual": float(att_bilstm_resid_full[i]),
                "window": WINDOW,
                "group_col_used": GROUP_COL,   # <-- إضافة صغيرة مفيدة للتوثيق (ما تمس النماذج)
                "target_col_used": TARGET_COL  # <-- إضافة صغيرة مفيدة للتوثيق
            })

df_ts = pd.DataFrame(rows)
ts_path = os.path.join(OUT_DIR, "MODEL_FULL_TIMESERIES_MULTI_METRIC.csv")
df_ts.to_csv(ts_path, index=False)
print("Saved:", ts_path)

# ==========================STEP F: SAVE OPTUNA TABLES (combined)==========================

df_optuna_all = pd.DataFrame(optuna_rows_meta)
optuna_all_path = os.path.join(OUT_DIR, "OPTUNA_BEST_PARAMS_ALL_META_MODELS.csv")
df_optuna_all.to_csv(optuna_all_path, index=False)
print("\nSaved:", optuna_all_path)

# ===============OPTIONAL: SAVE A COUPLE LSTM1 PLOTS======================

try:
    import matplotlib.pyplot as plt

    for k, (c, yt, yp) in enumerate(plot_lstm1[:5]):
        plt.figure(figsize=(10, 5))
        plt.plot(yt, label="True")
        plt.plot(yp, label="Pred")
        plt.title(f"LSTM1 — {c} (35–50) | target={TARGET_COL}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"LSTM1_{c}.png"))
        plt.close()
except Exception as e:
    print("[warn] Could not save plots:", str(e))

print(f"\n=== ALL DONE ✓ V13 (LSTM1 + 6 Meta Models, WINDOW fixed) ===")
print(f"=== Parallel processing: {MAX_WORKERS} countries/models at a time ===")
print(f"=== Total groups processed: {len(splits)} ===")
print(f"=== GROUP_COL used: {GROUP_COL} ===")
print(f"=== TARGET_COL used: {TARGET_COL} ===")
