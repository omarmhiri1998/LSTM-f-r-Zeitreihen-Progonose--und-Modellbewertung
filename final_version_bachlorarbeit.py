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

# Pfad zur Eingabe-CSV-Datei (muss mindestens country, Date, tavg enthalten)
DATA_PATH = "daily_weather_formatted_Date (1).csv"

# Ausgabeordner für alle Ergebnisse (CSV-Dateien, Plots, Optuna-Logs)
OUT_DIR = "plots_v13_all_models_fixed46_bilstm_residlstm"
os.makedirs(OUT_DIR, exist_ok=True)

# Reproduzierbarkeit durch festen Zufalls-Seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Feste Fensterlänge (Anzahl vergangener Zeitschritte pro Sample)
WINDOW = 180  # FIXED as requested

# ------------------------------------------------------
# Window-Splits (bezogen auf erzeugte Fenster, nicht Rohdaten)
# ------------------------------------------------------
# 0–35%   → Training LSTM1
# 35–50%  → interne Validierung / Diagnose
# 35–75%  → Training Meta-Modelle
# 75–90%  → Holdout-Bereich
# 90–100% → Finale Evaluation
P35 = 0.35
P50 = 0.50
P75 = 0.75
P90 = 0.90

# Fehlermetriken, die von den Meta-Modellen vorhergesagt werden
META_METRICS = ["sMAPE", "MAE", "MASE"]

# ------------------------------------------------------
# Optuna-Budgets
# ------------------------------------------------------
# Anzahl Trials und maximale Epochen für LSTM1 (Stufe 1)
N_TRIALS_L1 = 20
L1_MAX_EPOCHS = 50

# Anzahl Trials und maximale Epochen für Meta-Modelle (Stufe 2)
N_TRIALS_META = 20
META_MAX_EPOCHS = 50

# Speichern aller Optuna-Trials als CSV aktivieren/deaktivieren
SAVE_OPTUNA_TRIALS = True

# ------------------------------------------------------
# Saisonale Periode für MASE-Berechnung
# ------------------------------------------------------
# z.B. 365 für tägliche Daten mit jährlicher Saisonalität
SEASONAL_PERIOD = 365

# ------------------------------------------------------
# Parallelisierungseinstellungen
# ------------------------------------------------------
# Maximale Anzahl paralleler Worker (Länder / Modelle)
MAX_WORKERS = 4

# Nutzung von Threads (empfohlen bei GPU + TensorFlow)
USE_THREADS = True  


# ------------------------------------------------------
# GPU-Speicherkonfiguration
# ------------------------------------------------------
# Aktiviert Memory Growth, damit TensorFlow nicht den gesamten GPU-Speicher blockiert
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

# ------------------------------------------------------
# sMAPE (Symmetric Mean Absolute Percentage Error)
# ------------------------------------------------------
# Wird sowohl als Evaluationsmetrik als auch als Meta-Target genutzt
def smape_scalar(y_true, y_pred):
    y_true = float(y_true)
    y_pred = float(y_pred)
    return float(2.0 * abs(y_pred - y_true) / (abs(y_true) + abs(y_pred) + 1e-8))


# ------------------------------------------------------
# MASE (Mean Absolute Scaled Error)
# ------------------------------------------------------
# Skaliert den Fehler anhand einer saisonalen Naiv-Baseline
def mase_scalar(y_true, y_pred, y_train):
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= SEASONAL_PERIOD:
        denom = np.mean(np.abs(np.diff(y_train))) + 1e-8
    else:
        diffs = np.abs(y_train[SEASONAL_PERIOD:] - y_train[:-SEASONAL_PERIOD])
        denom = np.mean(diffs) + 1e-8
    return float(abs(float(y_true) - float(y_pred)) / denom)


# ------------------------------------------------------
# Skill-Score basierend auf MAE
# ------------------------------------------------------
# Misst die Verbesserung gegenüber der Global-Constant-Baseline
def skill_score_mae(mae_model, mae_baseline):
    return float(1.0 - (mae_model / (mae_baseline + 1e-8)))


# ------------------------------------------------------
# Sichere Spearman-Korrelation
# ------------------------------------------------------
# Verhindert Abstürze bei sehr kleinen oder degenerierten Stichproben
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

# ------------------------------------------------------
# Speichert alle Optuna-Trials als CSV-Datei
# ------------------------------------------------------
# study  : Optuna-Study-Objekt
# fname  : Dateiname für den Export
# Wird nur ausgeführt, wenn SAVE_OPTUNA_TRIALS = True
def save_optuna_trials(study, fname):
    if not SAVE_OPTUNA_TRIALS:
        return
    try:
        df_trials = study.trials_dataframe()
        df_trials.to_csv(os.path.join(OUT_DIR, fname), index=False)
    except Exception as e:
        print(" [warn] Could not save optuna trials:", fname, "->", str(e))


# ------------------------------------------------------
# Verpackt die besten Hyperparameter in ein einheitliches Dict-Format
# ------------------------------------------------------
# stage        : Modellname / Phase (z.B. LSTM1, META_ATT, etc.)
# metric       : Zielmetrik (sMAPE, MAE, MASE)
# best_value   : Bester Validierungswert
# best_params  : Hyperparameter-Dictionary von Optuna
# extra        : Optional zusätzliche Informationen (z.B. WINDOW)
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

# ------------------------------------------------------
# Baut das LSTM1-Prognosemodell
# ------------------------------------------------------
# trial      : Optuna-Trial (liefert Hyperparameter)
# window     : Fensterlänge (Anzahl Zeitschritte)
# input_dim  : Anzahl Features pro Zeitschritt
#
# Architektur:
# - LSTM
# - Dropout
# - Dense (ReLU)
# - Output Dense (Regression)
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
# CHANGE REQUESTED: LSTM1 Optuna uses Pareto (multi-objective)
# Objectives: MAE, sMAPE, MASE (on normalized target y)
# ======================================================

# ------------------------------------------------------
# Multi-Objective-Optimierungsfunktion für LSTM1
# ------------------------------------------------------
# Optimiert gleichzeitig:
# - MAE
# - sMAPE
# - MASE
#
# Rückgabe: Tupel (mae, smape, mase)
# → Optuna erzeugt daraus eine Pareto-Front
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

    # Speichert die effektive Anzahl trainierter Epochen
    trial.set_user_attr("effective_epochs", len(history.history['loss']))

    pred = m.predict(X, verbose=0).flatten()

    # MAE
    mae = mean_absolute_error(y, pred)

    # sMAPE
    smape = float(np.mean([smape_scalar(y[i], pred[i]) for i in range(len(y))]))

    # MASE (hier skaliert mit einfacher Differenz-Baseline auf normiertem y)
    denom = float(np.mean(np.abs(np.diff(y))) + 1e-8)
    mase = float(np.mean(np.abs(y - pred) / denom))

    return mae, smape, mase


# ======================================================
# META MODELS (base architectures) 
# ======================================================
# ======================================================
# 1) META_LSTM (nur Sequenzmodell)
# ======================================================
# Dieses Modell dient als Basis-Meta-Architektur.
#
# Ziel:
# Vorhersage der erwarteten Prognosefehlerhöhe
# (z.B. sMAPE, MAE oder MASE) ausschließlich
# basierend auf einem historischen Eingabefenster.
#
# Struktur:
# LSTM → Dropout → Dense(ReLU) → Dense(1)
#
# Keine Attention.
# Keine statischen Zusatzfeatures.
# Reines Sequenzmodell.


def build_meta_lstm(trial, window):

    # --------------------------------------------------
    # Hyperparameter-Suche mit Optuna
    # --------------------------------------------------

    # Anzahl der LSTM-Neuronen (Modellkapazität)
    units = trial.suggest_int("meta_lstm_units", 32, 128)

    # Größe der Dense-Zwischenschicht
    dense_units = trial.suggest_int("meta_lstm_dense", 16, 64)

    # Lernrate (logarithmisch gesucht)
    lr = trial.suggest_float("meta_lstm_lr", 1e-4, 1e-2, log=True)

    # Geduld für Early Stopping
    patience = trial.suggest_int("meta_lstm_patience", 3, 10)

    # Dropout zur Regularisierung
    dropout = trial.suggest_float("meta_lstm_dropout", 0.0, 0.5)

    # Recurrent Dropout (auf rekurrente Verbindungen)
    recurrent_dropout = trial.suggest_float("meta_lstm_recurrent_dropout", 0.0, 0.5)

    # L2-Regularisierung gegen Overfitting
    l2_reg = trial.suggest_float("meta_lstm_l2_reg", 1e-6, 1e-2, log=True)


    # --------------------------------------------------
    # Modellarchitektur
    # --------------------------------------------------

    # Eingabe: (Fensterlänge, 1 Feature)
    inp = layers.Input(shape=(window, 1))

    # LSTM extrahiert zeitliche Muster
    # return_sequences=False (Standard),
    # da nur Endzustand verwendet wird
    x = layers.LSTM(
        units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    )(inp)

    # Zusätzliche Regularisierung nach LSTM
    x = layers.Dropout(dropout)(x)

    # Nichtlineare Projektion
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=l2(l2_reg)
    )(x)

    # Lineare Regressionsausgabe
    # Ziel: kontinuierlicher Fehlerwert
    out = layers.Dense(
        1,
        kernel_regularizer=l2(l2_reg)
    )(x)

    # Modellobjekt erzeugen
    m = Model(inp, out)

    # Optimierung mit Adam
    # Verlustfunktion: MSE
    m.compile(
        optimizer=Adam(lr),
        loss="mse"
    )

    return m, patience



# ======================================================
# Optuna-Objective für META_LSTM
# ======================================================
# Trainiert das Modell auf gepoolten Fenstern
# (35–75%) und gibt MAE als Optimierungsziel zurück.


def objective_meta_lstm(trial, Xs, y, window):

    # Modell auf Basis der Trial-Hyperparameter erzeugen
    m, patience = build_meta_lstm(trial, window)

    # Early Stopping zur Vermeidung von Overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    # Training mit 20% interner Validierung
    history = m.fit(
        Xs, y,
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    # Speichern der tatsächlich verwendeten Epochen
    trial.set_user_attr(
        "effective_epochs",
        len(history.history['loss'])
    )

    # Vorhersage auf Trainingsdaten
    pred = m.predict(Xs, verbose=0).flatten()

    # Optimierungsziel:
    # Minimierung des mittleren absoluten Fehlers
    return mean_absolute_error(y, pred)

# ======================================================
# 2) META_LSTM + STAT
# ======================================================
# Erweiterte Meta-Architektur:
# Kombination aus Sequenzinformation (LSTM)
# und statischen Fenstermerkmalen.
#
# Idee:
# Reines LSTM sieht nur die zeitliche Dynamik.
# Durch zusätzliche statistische Kennzahlen
# erhält das Modell explizite Strukturinformationen
# über Niveau, Streuung, Schiefe, Trend usw.
#
# Architektur:
#   Sequenzzweig: LSTM → Dropout
#   Statistikzweig: Dense(32)
#   Fusion: Concatenate → Dense → Output
#
# Ziel:
# Verbesserung der Fehlervorhersage durch
# Kombination impliziter und expliziter Merkmale.
# ======================================================



# ------------------------------------------------------
# Berechnung statischer Fenstermerkmale
# ------------------------------------------------------
# Diese Funktion extrahiert feste statistische Kennzahlen
# aus einem Eingabefenster.
#
# Diese Features repräsentieren:
# - Lage (mean)
# - Streuung (std)
# - Extremwerte (min, max)
# - Trend (Differenz letzter − erster Wert)
# - Form der Verteilung (Schiefe, Kurtosis)
# - Robustmaß für Streuung (IQR)
#
# Ausgabe:
# Vektor fester Länge (stats_dim)
# ------------------------------------------------------

def window_stats(arr):

    # Sicherstellen, dass numerisches Array verwendet wird
    arr = np.array(arr, dtype=float)

    return np.array([

        # Mittelwert (Niveau des Fensters)
        np.mean(arr),

        # Standardabweichung (Volatilität)
        np.std(arr),

        # Minimum im Fenster
        np.min(arr),

        # Maximum im Fenster
        np.max(arr),

        # Einfacher Trendindikator
        # Differenz letzter minus erster Wert
        arr[-1] - arr[0],

        # Schiefe der Verteilung
        skew(arr),

        # Kurtosis (Wölbung)
        kurtosis(arr),

        # Interquartilsabstand (robustes Streuungsmaß)
        np.percentile(arr, 75) - np.percentile(arr, 25)

    ], dtype=float)



# ------------------------------------------------------
# Modellaufbau: META_LSTM_STAT
# ------------------------------------------------------
# Zwei Eingaben:
#   1) Sequenzinput (window × 1)
#   2) Statistikinput (stats_dim)
#
# Ziel:
# Fusion beider Informationsquellen
# zur präziseren Fehlerschätzung
# ------------------------------------------------------

def build_meta_lstm_stat(trial, window, stats_dim):

    # ------------------------------
    # Hyperparameter-Suche
    # ------------------------------

    units = trial.suggest_int("meta_lstm_stat_units", 32, 128)
    dense_units = trial.suggest_int("meta_lstm_stat_dense", 16, 64)
    lr = trial.suggest_float("meta_lstm_stat_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("meta_lstm_stat_patience", 3, 10)

    dropout = trial.suggest_float("meta_lstm_stat_dropout", 0.0, 0.5)
    recurrent_dropout = trial.suggest_float("meta_lstm_stat_recurrent_dropout", 0.0, 0.5)
    l2_reg = trial.suggest_float("meta_lstm_stat_l2_reg", 1e-6, 1e-2, log=True)


    # ------------------------------
    # Sequenzzweig
    # ------------------------------

    seq_in = layers.Input(
        shape=(window, 1),
        name="seq_in"
    )

    x = layers.LSTM(
        units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    )(seq_in)

    x = layers.Dropout(dropout)(x)


    # ------------------------------
    # Statistikzweig
    # ------------------------------

    stat_in = layers.Input(
        shape=(stats_dim,),
        name="stat_in"
    )

    # Projektion der statistischen Merkmale
    z = layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=l2(l2_reg)
    )(stat_in)


    # ------------------------------
    # Fusion beider Zweige
    # ------------------------------

    h = layers.Concatenate()([x, z])

    h = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=l2(l2_reg)
    )(h)

    # Regressionsausgabe (Fehlerwert)
    out = layers.Dense(
        1,
        kernel_regularizer=l2(l2_reg)
    )(h)


    # ------------------------------
    # Modellkompilierung
    # ------------------------------

    m = Model([seq_in, stat_in], out)

    m.compile(
        optimizer=Adam(lr),
        loss="mse"
    )

    return m, patience



# ------------------------------------------------------
# Optuna-Objective für META_LSTM_STAT
# ------------------------------------------------------
# Trainiert das kombinierte Modell
# auf gepoolten Fenstern (35–75%)
# und minimiert den MAE.
# ------------------------------------------------------

def objective_meta_lstm_stat(trial, Xs_seq, Xs_stat, y, window, stats_dim):

    # Modell anhand Trial-Hyperparameter erzeugen
    m, patience = build_meta_lstm_stat(trial, window, stats_dim)

    # Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    # Training mit interner Validierung
    history = m.fit(
        [Xs_seq, Xs_stat],
        y,
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    # Dokumentation der effektiven Epochenzahl
    trial.set_user_attr(
        "effective_epochs",
        len(history.history['loss'])
    )

    # Vorhersage
    pred = m.predict(
        [Xs_seq, Xs_stat],
        verbose=0
    ).flatten()

    # Optimierungsziel: MAE minimieren
    return mean_absolute_error(y, pred)


# ======================================================
# 3) META_ATT (Sequenz + Attention)
# ======================================================
# Motivation:
# Ein reines LSTM komprimiert die gesamte Sequenz
# in einen einzigen versteckten Zustand.
# Dabei gehen Informationen darüber verloren,
# welche Zeitpunkte im Fenster besonders wichtig waren.
#
# Die Attention-Mechanismus löst dieses Problem,
# indem er lernbare Gewichte über alle Zeitpunkte verteilt.
#
# Vorteil:
# Das Modell lernt explizit,
# welche Teile des Fensters für die Fehlerschätzung
# besonders relevant sind.
# ======================================================



# ------------------------------------------------------
# Attention Layer
# ------------------------------------------------------
# Diese Klasse implementiert eine einfache additive
# Attention-Mechanik über die Zeitdimension.
#
# Eingabe:
#   Tensor der Form (Batch, TimeSteps, Features)
#
# Ablauf:
#   1) Lineare Projektion jedes Zeitschritts
#   2) Nichtlineare Transformation (tanh)
#   3) Softmax-Normalisierung über Zeit
#   4) Gewichtete Summation
#
# Ausgabe:
#   Kontextvektor (Batch, Features)
# ------------------------------------------------------

class Attention(layers.Layer):

    def build(self, input_shape):

        # Gewichtsmatrix für Feature-Projektion
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True
        )

        # Bias pro Zeitschritt
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )


    def call(self, x):

        # Lineare Projektion
        score = tf.matmul(x, self.W) + self.b

        # Nichtlineare Aktivierung
        score = tf.nn.tanh(score)

        # Softmax über Zeitdimension
        w = tf.nn.softmax(score, axis=1)

        # Gewichtete Aggregation
        context = tf.reduce_sum(x * w, axis=1)

        return context



# ------------------------------------------------------
# Modellaufbau: META_ATT
# ------------------------------------------------------
# Architektur:
#   LSTM (return_sequences=True)
#   → Attention
#   → Dense
#   → Output
#
# Ziel:
# Relevante Zeitpunkte im Fenster
# explizit gewichten.
# ------------------------------------------------------

def build_meta_att(trial, window):

    # ------------------------------
    # Hyperparameter-Suche
    # ------------------------------

    units = trial.suggest_int("meta_att_units", 32, 128)
    dense_units = trial.suggest_int("meta_att_dense", 16, 64)
    lr = trial.suggest_float("meta_att_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("meta_att_patience", 3, 10)

    dropout = trial.suggest_float("meta_att_dropout", 0.0, 0.5)
    recurrent_dropout = trial.suggest_float("meta_att_recurrent_dropout", 0.0, 0.5)
    l2_reg = trial.suggest_float("meta_att_l2_reg", 1e-6, 1e-2, log=True)


    # ------------------------------
    # Eingabeschicht
    # ------------------------------

    seq_in = layers.Input(shape=(window, 1))


    # ------------------------------
    # LSTM mit Sequenzrückgabe
    # ------------------------------
    # return_sequences=True ist notwendig,
    # damit Attention Zugriff auf alle Zeitschritte erhält.

    x = layers.LSTM(
        units,
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    )(seq_in)


    # ------------------------------
    # Attention
    # ------------------------------

    x = Attention()(x)


    # ------------------------------
    # Dense-Projektion
    # ------------------------------

    x = layers.Dropout(dropout)(x)

    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=l2(l2_reg)
    )(x)


    # Regressionsausgabe
    out = layers.Dense(
        1,
        kernel_regularizer=l2(l2_reg)
    )(x)


    # ------------------------------
    # Modellkompilierung
    # ------------------------------

    m = Model(seq_in, out)

    m.compile(
        optimizer=Adam(lr),
        loss="mse"
    )

    return m, patience



# ------------------------------------------------------
# Optuna-Objective für META_ATT
# ------------------------------------------------------
# Trainiert das Modell
# und minimiert den MAE.
# ------------------------------------------------------

def objective_meta_att(trial, Xs, y, window):

    m, patience = build_meta_att(trial, window)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = m.fit(
        Xs,
        y,
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    trial.set_user_attr(
        "effective_epochs",
        len(history.history['loss'])
    )

    pred = m.predict(Xs, verbose=0).flatten()

    return mean_absolute_error(y, pred)



# ======================================================
# 4) META_ATT_BILSTM
# ======================================================
# Erweiterung des Attention-Modells
# durch bidirektionale Verarbeitung.
#
# Motivation:
# Fehlerstrukturen können sowohl von
# frühen als auch späten Sequenzanteilen abhängen.
#
# Bidirectional-LSTM verarbeitet:
#   → Vorwärtsrichtung
#   → Rückwärtsrichtung
#
# Dadurch erhält Attention einen reicheren Kontext.
# ======================================================



# ------------------------------------------------------
# Modellaufbau: META_ATT_BILSTM
# ------------------------------------------------------

def build_meta_att_bilstm(trial, window):

    # ------------------------------
    # Hyperparameter
    # ------------------------------

    units = trial.suggest_int("meta_att_bilstm_units", 32, 128)
    dense_units = trial.suggest_int("meta_att_bilstm_dense", 16, 64)
    lr = trial.suggest_float("meta_att_bilstm_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int("meta_att_bilstm_patience", 3, 10)

    dropout = trial.suggest_float("meta_att_bilstm_dropout", 0.0, 0.5)
    recurrent_dropout = trial.suggest_float("meta_att_bilstm_recurrent_dropout", 0.0, 0.5)
    l2_reg = trial.suggest_float("meta_att_bilstm_l2_reg", 1e-6, 1e-2, log=True)


    # ------------------------------
    # Eingabe
    # ------------------------------

    seq_in = layers.Input(shape=(window, 1))


    # ------------------------------
    # Bidirectional LSTM
    # ------------------------------

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


    # ------------------------------
    # Attention
    # ------------------------------

    x = Attention()(x)


    # ------------------------------
    # Dense + Output
    # ------------------------------

    x = layers.Dropout(dropout)(x)

    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=l2(l2_reg)
    )(x)

    out = layers.Dense(
        1,
        kernel_regularizer=l2(l2_reg)
    )(x)


    # ------------------------------
    # Kompilierung
    # ------------------------------

    m = Model(seq_in, out)

    m.compile(
        optimizer=Adam(lr),
        loss="mse"
    )

    return m, patience



# ------------------------------------------------------
# Optuna-Objective für META_ATT_BILSTM
# ------------------------------------------------------

def objective_meta_att_bilstm(trial, Xs, y, window):

    m, patience = build_meta_att_bilstm(trial, window)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = m.fit(
        Xs,
        y,
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    trial.set_user_attr(
        "effective_epochs",
        len(history.history['loss'])
    )

    pred = m.predict(Xs, verbose=0).flatten()

    return mean_absolute_error(y, pred)


# ======================================================
# RESIDUAL-LSTM = SAME STYLE AS META_LSTM
# Target residual = y_true - base_prediction
# ======================================================
# Idee:
# Bei Residual-Varianten wird zuerst ein Basis-Meta-Modell trainiert (z.B. META_ATT),
# das die Fehlergröße direkt schätzt.
#
# Danach wird ein zweites Modell (Residual-LSTM) trainiert, das NICHT den Fehler selbst,
# sondern das Residuum lernt:
#
#   residual_target = y_true - base_prediction
#
# Finale Vorhersage im Residual-Setup:
#   final_pred = base_prediction + residual_prediction
#
# Wichtig:
# Das Residual-Modell muss dieselbe Trainingslogik und denselben Stil wie META_LSTM haben,
# damit der Vergleich fair bleibt (Optuna, EarlyStopping, Dropout, L2, etc.).
# ======================================================


def build_meta_residual_lstm(trial, window, prefix="resid_lstm"):
    # ------------------------------------------------------
    # Modell-Build-Funktion für das Residual-LSTM
    # ------------------------------------------------------
    # Diese Funktion entspricht stilistisch dem META_LSTM:
    #   LSTM → Dropout → Dense(ReLU) → Dense(1)
    #
    # prefix:
    # Damit Optuna-Parameter je Residual-Variante getrennt bleiben
    # (z.B. resid_att_lstm_units vs resid_att_bilstm_lstm_units).
    # ------------------------------------------------------

    # Optuna Hyperparameter
    units = trial.suggest_int(f"{prefix}_units", 32, 128)
    dense_units = trial.suggest_int(f"{prefix}_dense", 16, 64)
    lr = trial.suggest_float(f"{prefix}_lr", 1e-4, 1e-2, log=True)
    patience = trial.suggest_int(f"{prefix}_patience", 3, 10)

    dropout = trial.suggest_float(f"{prefix}_dropout", 0.0, 0.5)
    recurrent_dropout = trial.suggest_float(f"{prefix}_recurrent_dropout", 0.0, 0.5)
    l2_reg = trial.suggest_float(f"{prefix}_l2_reg", 1e-6, 1e-2, log=True)

    # Input: Sequenzfenster (window, 1)
    inp = layers.Input(shape=(window, 1))

    # LSTM Encoder
    x = layers.LSTM(
        units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    )(inp)

    # Regularisierung
    x = layers.Dropout(dropout)(x)

    # Dense Head (wie META_LSTM)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=l2(l2_reg))(x)

    # Regressionsausgabe: Residuum-Schätzer
    out = layers.Dense(1, kernel_regularizer=l2(l2_reg))(x)

    # Modell erstellen und kompilieren
    m = Model(inp, out)
    m.compile(optimizer=Adam(lr), loss="mse")

    return m, patience


def objective_meta_residual_lstm(trial, Xs, resid_y, window, prefix="resid_lstm"):
    # ------------------------------------------------------
    # Optuna Objective für Residual-LSTM
    # ------------------------------------------------------
    # Trainiert das Residual-Modell auf resid_y und minimiert MAE.
    #
    # resid_y:
    #   Zielwert ist NICHT der Fehler selbst,
    #   sondern das Residuum (y_true - base_pred).
    # ------------------------------------------------------

    # Modell bauen
    m, patience = build_meta_residual_lstm(trial, window, prefix=prefix)

    # EarlyStopping (gleiches Schema wie bei anderen Meta-Modellen)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    # Training
    history = m.fit(
        Xs, resid_y,
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    # Logging: effektive Epochenzahl für spätere Analyse/CSV
    trial.set_user_attr("effective_epochs", len(history.history['loss']))

    # Vorhersage + Objective-Wert (MAE)
    pred = m.predict(Xs, verbose=0).flatten()
    return mean_absolute_error(resid_y, pred)



# ==========================LOAD DATA============================
# Zweck:
# CSV laden, grundlegende Validierung durchführen und Daten so vorbereiten,
# dass das restliche Skript reproduzierbar und stabil läuft.
#
# Erwartungen:
# - Pflichtspalten: country, Date, tavg
# - Weitere Feature-Spalten optional (werden automatisch genutzt, falls vorhanden)
# ===============================================================

print(">>> Loading:", DATA_PATH)

# CSV laden
df = pd.read_csv(DATA_PATH)

# Länderbereinigung:
# - Nur Zeilen mit gültigem country
# - Trim whitespace
df = df.dropna(subset=["country"])
df["country"] = df["country"].astype(str).str.strip()
df = df[df["country"] != ""].dropna()

# Pflichtspalte Date prüfen
if "Date" not in df.columns:
    raise RuntimeError("Column 'Date' is required in dataset.")

# Date in datetime konvertieren + nach Datum sortieren
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Pflichtspalte target prüfen
if "tavg" not in df.columns:
    raise RuntimeError("Column 'tavg' is required as target but not found in dataset.")


# Feature-Liste:
# Ziel ist ein robuster Default, der mit unterschiedlich vollständigen CSVs funktioniert.
FEATURES = [
    "tavg", "tmin", "tmax",
    "Temp_Max", "Temp_Mean", "Temp_Min",
    "wspd", "wgust", "Windspeed_Max", "Windgusts_Max",
    "sunshine", "Sunshine_Duration",
    "prcp", "Precipitation_Sum"
]

# Nur Features behalten, die tatsächlich in der CSV existieren
FEATURES = [c for c in FEATURES if c in df.columns]

print(">>> Using FEATURES:", FEATURES)

# Länder-Liste extrahieren
countries = df["country"].unique().tolist()

print(f">>> Countries found: {len(countries)}")
print(f">>> WINDOW fixed = {WINDOW}")



# ======================================================
# STORAGE / POOLS
# ======================================================
# Zweck:
# Während STEP A (LSTM1 pro Land) werden Daten gesammelt,
# die später für das globale Meta-Training (STEP C) benötigt werden.
#
# Gesammelt wird:
# - Sequenzen (X_pool_seq): Fenster-Sequenzen (window, 1)
# - Statistiken (X_pool_stat): feste Fenster-Features (mean/std/etc.), falls benötigt
# - Targets pro Metrik (y_pool): true errors pro Fenster (sMAPE/MAE/MASE)
# - Splits pro Land (splits): 75–90 und 90–100 windows pro Land für spätere Evaluation
# - Baseline-Pool (global_true_errors_35_90): True Errors 35–90 für globale Baseline
# ======================================================

# Pool der Sequenzfenster für Meta-Training (über alle Länder)
X_pool_seq = []

# Pool statischer Fensterfeatures (wird nur für STAT-Modelle genutzt)
X_pool_stat = []

# Targets je Metrik (Fehlerwerte), gesammelt über alle Länder
y_pool = {m: [] for m in META_METRICS}

# Dictionary pro Land: enthält vorbereitete Split-Fenster für spätere Prediction/Evaluation
splits = {}

# True Errors (35–90) über alle Länder gepoolt:
# wird später zur Berechnung der Global-Constant-Baseline genutzt
global_true_errors_35_90 = {m: [] for m in META_METRICS}

# Speichert Optuna Best-Rows der Meta-Modelle (für CSV-Export)
optuna_rows_meta = []

# Speichert Optuna Best-Configs der LSTM1 Modelle pro Land
lstm1_best_rows = []

# Optional: Daten zum Plotten einiger LSTM1 Forecasts (Debug/Visualisierung)
plot_lstm1 = []

# Optional: Platzhalter für Debug (falls du später History für erstes Land speichern willst)
first_lstm1_history = None
first_lstm1_country = None

# ===========================STEP A: TRAIN LSTM1 PER COUNTRY + COLLECT TRUE ERRORS===========================

print(f"\n=== STEP A: LSTM1 PER COUNTRY + TRUE ERRORS (Parallel - Max Workers: {MAX_WORKERS}) ===")

def process_country_lstm1(country):
    print(f" Starting LSTM1 for {country}...")
    dfc = df[df["country"] == country].reset_index(drop=True)
    if dfc.shape[0] < WINDOW + 60:
        print(f" Skipping {country}: insufficient data")
        return None, [], [], [], None, None, None

    data = dfc[FEATURES].values.astype(float)
    tavg = dfc["tavg"].values.astype(float)
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
        y1.append(norm[i+WINDOW, 0])
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

    # ======================================================
    #  PARETO (multi-objective) tuning for LSTM1
    # Objectives: MAE, sMAPE, MASE (on normalized y)
    # ======================================================
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
        "Country": country,
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
        rt = tavg[WINDOW+w35:WINDOW+w50]
        if len(rt) == len(rp) and len(rt) > 0:
            plot_data = (country, rt, rp)
    except Exception as e:
        print(f" Warning: Could not generate plot data for {country}: {e}")

    all_p = m1.predict(X1, verbose=0).flatten()
    all_raw = all_p * scaler.scale_[0] + scaler.mean_[0]
    all_true = tavg[WINDOW:WINDOW+nW]

    err_smape = np.array([smape_scalar(all_true[i], all_raw[i]) for i in range(nW)], dtype=float)
    err_mae = np.array([abs(all_true[i] - all_raw[i]) for i in range(nW)], dtype=float)
    err_mase = np.array([mase_scalar(all_true[i], all_raw[i], tavg[:WINDOW+i]) for i in range(nW)], dtype=float)

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

print(f"Processing {len(countries)} countries in parallel (max {MAX_WORKERS} at a time)...")
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
    raise RuntimeError("Meta pool too small. Check dataset size / country filtering.")

stats_dim = X_pool_stat.shape[1]
print(f"=== LSTM1 DONE ===")
print(f"Pooled meta windows: {X_pool_seq.shape[0]}")
print(f"Stats dim: {stats_dim}")
print(f"Countries kept for splits: {len(splits)}")

df_lstm1_best = pd.DataFrame(lstm1_best_rows)
df_lstm1_best_path = os.path.join(OUT_DIR, "OPTUNA_BEST_LSTM1_PER_COUNTRY.csv")
df_lstm1_best.to_csv(df_lstm1_best_path, index=False)
print("Saved:", df_lstm1_best_path)

# ========================STEP B: SUPERVISOR GLOBAL CONSTANT BASELINE(für vergleich mit meta modell ) (per metric)==============================

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

    # ============================Residual Variants (Base + LSTM Residual)==========================
    
    elif arch_name in ["ATT_RESID", "ATT_BILSTM_RESID"]:
        # 1) Tune + train base model first
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

        # 2) Residual target = y_true - base_pred (THIS is the only correct residual)
        base_pred_train = base_model.predict(X_pool_seq, verbose=0).flatten()
        resid_target = (y_pool[metric].astype(float) - base_pred_train).astype(float)

        # 3) Tune residual model with Optuna (same style as META_LSTM)
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

        # rows
        optuna_rows_meta.append(base_row)
        resid_row = pack_best_row(
            f"META_{arch_name}(residual_lstm_optuna_same_style)",
            metric,
            st_resid.best_value,
            st_resid.best_trial.params,
            {"WINDOW": WINDOW, "EffectiveEpochs": st_resid.best_trial.user_attrs.get("effective_epochs", "N/A"),
             "BaseModel": "ATT" if arch_name == "ATT_RESID" else "ATT_BILSTM"}
        )

        # predict per country: final = base + residual_hat
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

    # Train non-residual models
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0)
    model.fit(
        X_train, y_pool[metric],
        epochs=META_MAX_EPOCHS,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    # predict per country
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
                "country": c,
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
                "window": WINDOW
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
        plt.title(f"LSTM1 — {c} (35–50)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"LSTM1_{c}.png"))
        plt.close()
except Exception as e:
    print("[warn] Could not save plots:", str(e))

print(f"\n=== ALL DONE ✓ V13 (LSTM1 + 6 Meta Models, WINDOW fixed) ===")
print(f"=== Parallel processing: {MAX_WORKERS} countries/models at a time ===")
print(f"=== Total countries processed: {len(splits)} ===")



