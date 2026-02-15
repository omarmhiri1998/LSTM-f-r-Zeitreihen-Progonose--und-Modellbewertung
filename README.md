## Code-Struktur (Abschnitte im Skript)

Das Skript ist bewusst in klar markierte Blöcke gegliedert:

- `# ======================================================`
  `# IMPORTS`
  `# ======================================================`  
  Importiert alle benötigten Bibliotheken (NumPy, Pandas, TensorFlow/Keras, Optuna, Scikit-Learn, SciPy).

- `# ======================================================`
  `# CONFIG`
  `# ======================================================`  
  Zentrale Konfiguration: Datenpfad, Output-Ordner, Seed, Fensterlänge, Splits, Optuna-Budgets, Metriken, Parallelität und GPU-Setup.

- `# ======================================================`
  `# OPTUNA HELPERS`
  `# ======================================================`  
  Hilfsfunktionen zum Speichern von Optuna-Trials (`save_optuna_trials`) und zum einheitlichen Protokollieren der besten Parameter (`pack_best_row`).

- `# ======================================================`
  `# LSTM1 (per country)`
  `# ======================================================`  
  Definiert und trainiert das Prognosemodell **LSTM1 pro Land**.

- `# ======================================================`
  `# CHANGE REQUESTED: LSTM1 Optuna uses Pareto (multi-objective)`
  `# Objectives: MAE, sMAPE, MASE (on normalized target y)`
  `# ======================================================`  
  Optuna optimiert LSTM1 **multi-objektiv** (MAE, sMAPE, MASE). Aus der Pareto-Front wird per **Distance-to-Ideal** ein Kompromissmodell gewählt.

- `# ======================================================`
  `# META MODELS (base architectures)`
  `# ======================================================`  
  Definiert die globalen Meta-Architekturen, die pro Fenster die erwarteten Fehler vorhersagen.

- `# ======================================================`
  `# RESIDUAL-LSTM = SAME STYLE AS META_LSTM`
  `# Target residual = y_true - base_prediction`
  `# ======================================================`  
  Residual-Modelle werden im **gleichen Stil wie META_LSTM** trainiert. Ziel ist das Residuum:
  `residual = y_true - base_prediction`.

- `# ==========================LOAD DATA============================`  
  Laden und Vorverarbeitung der CSV-Daten, Feature-Auswahl, Sortierung nach Datum, Extraktion der Länder.

- `# ======================================================`
  `# STORAGE / POOLS`
  `# ======================================================`  
  Sammelstrukturen (Pools) für globale Meta-Trainingsdaten: Sequenzen, Statistiken, Targets pro Metrik sowie Splits pro Land.

---

## Pipeline-Logik (Steps)

### STEP A: LSTM1 pro Land + True Errors
`# ===========================STEP A: TRAIN LSTM1 PER COUNTRY + COLLECT TRUE ERRORS===========================`

Für jedes Land:

1. CSV wird gefiltert (`country`), nach `Date` sortiert und auf Features reduziert.
2. Sliding-Window-Erzeugung mit fixer Fensterlänge `WINDOW`.
3. Split auf Window-Ebene: 0–35%, 35–50%, 35–75%, 35–90%, 90–100%.
4. **Pareto-Optuna** auf 0–35% (Train) für MAE/sMAPE/MASE.
5. Bestes Kompromissmodell wird trainiert.
6. Vorhersagen werden für 35–100% erzeugt.
7. Pro Fenster werden **True Errors** berechnet:
   - sMAPE
   - MAE
   - MASE (saisonal, `m = 365`)

Parallelisierung: Länder werden via `ThreadPoolExecutor` verarbeitet (`MAX_WORKERS = 4`).

---

### STEP B: Supervisor Global-Constant Baseline pro Metrik
`# ========================STEP B: SUPERVISOR GLOBAL CONSTANT BASELINE...==============================`

Für jede Metrik wird eine konstante Baseline berechnet:

- **Global Constant Baseline** = Mittelwert der **True Errors** aus dem Bereich **35–90%**, gepoolt über alle Länder.

Diese Baseline wird später für den **Skill Score** (auf MAE-Basis) genutzt.

---

### STEP C: Training der Meta-Modelle (global)
`# ===========================STEP C: TRAIN META MODELS==========================`

Meta-Modelle werden global trainiert auf gepoolten Fenstern **35–75%** (alle Länder zusammen).

Implementierte Architekturen:

1. `META_LSTM` (nur Sequenz)
2. `META_LSTM_STAT` (Sequenz + statische Fenster-Features)
3. `META_ATT` (LSTM + Attention)
4. `META_ATT_BILSTM` (BiLSTM + Attention)
5. `META_ATT_RESID` (ATT-Basis + Residual-LSTM-Korrektur)
6. `META_ATT_BILSTM_RESID` (ATT-BiLSTM-Basis + Residual-LSTM-Korrektur)

**Residual Variants (Base + LSTM Residual)**  
`# ============================Residual Variants (Base + LSTM Residual)==========================`

Ablauf bei Residual-Modellen:
1. Basis-Modell (ATT oder ATT_BILSTM) wird per Optuna getuned und trainiert.
2. Residual-Targets werden berechnet: `y_true - base_pred`.
3. Residual-LSTM wird per Optuna getuned und trainiert (gleiche Struktur wie META_LSTM).
4. Finale Vorhersage: `base_pred + resid_pred`.

Parallelisierung: Training pro (Metrik × Architektur) via `ThreadPoolExecutor`.

---

### STEP D: Evaluation auf 90–100%
`# ============================STEP D: EVALUATION ON 90–100==========================`

Evaluation erfolgt ausschließlich auf dem letzten, unbeobachteten Segment **90–100%** (gepoolt über alle Länder):

- MAE
- Spearman-Korrelation
- Skill Score (MAE vs. Global Constant Baseline)

---

### STEP E: Export Full Timeseries (75–100)
`# ============================STEP E: EXPORT FULL TIMESERIES (75–100)==========================`

Exportiert pro Land und pro Metrik den Bereich **75–100%** mit:

- True Error
- Global Constant Baseline
- Vorhersagen aller Meta-Modelle

---

### STEP F: Speichern der Optuna-Tabellen
`# ==========================STEP F: SAVE OPTUNA TABLES (combined)==========================`

Speichert Best-Parameter/Best-Values der Meta-Modelle in einer kombinierten CSV.

---

### Optional: LSTM1-Plots
`# ===============OPTIONAL: SAVE A COUPLE LSTM1 PLOTS======================`

Speichert Diagnose-Plots für einige Länder (Segment 35–50%).

---

## Wichtige Einstellungen

- Fixe Fensterlänge: `WINDOW = 180`
- Saisonale MASE-Periode: `SEASONAL_PERIOD = 365`
- Parallelität: `MAX_WORKERS = 4` (Threads)
- GPU-Speicher: Memory-Growth aktiviert + optionales Memory-Limit (4096 MB)

---

## Output (im OUT_DIR)

Standard-Ausgabeordner:
`plots_v13_all_models_fixed46_bilstm_residlstm/`

Erzeugte Dateien (Auszug):
- `OPTUNA_BEST_LSTM1_PER_COUNTRY.csv`
- `OPTUNA_BEST_PARAMS_META_MODELS.csv`
- `ALL_META_MODEL_SUMMARY_MULTI_METRIC.csv`
- `MODEL_FULL_TIMESERIES_MULTI_METRIC.csv`
- `optuna_trials_*.csv` (falls aktiviert)
- `LSTM1_<country>.png` (optional)

---

## Kurzfazit

Die Pipeline trainiert zunächst LSTM1 pro Land (mit Pareto-Optuna) und berechnet daraus echte Fehler pro Fenster.
Darauf werden globale Meta-Modelle trainiert, die die erwarteten Fehler (sMAPE/MAE/MASE) für neue historische Fenster schätzen.
Die Bewertung erfolgt strikt im 90–100%-Segment und wird gegen eine globale konstante Baseline verglichen.
