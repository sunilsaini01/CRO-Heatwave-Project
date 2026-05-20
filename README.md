# 🌡️ CRO Heatwave Prediction System
### District-Level Heatwave Early Warning · Uttar Pradesh, India

> **Built as part of the Climate Resilience Observatory (CRO) initiative — a data-driven decision support system for the Government of Uttar Pradesh to enable timely early warning and adaptive response to extreme heat events.**

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [What is a Heatwave?](#-what-is-a-heatwave)
3. [Dataset](#-dataset)
4. [Methodology](#-methodology)
   - [Stage 1 — LSTM Baseline](#stage-1--lstm-baseline-temporal-sequence-modeling)
   - [Stage 2 — XGBoost Residual Corrector](#stage-2--xgboost-residual-corrector)
   - [Hybrid Prediction Formula](#hybrid-prediction-formula)
5. [Results](#-results)
6. [Repository Structure](#-repository-structure)
7. [Future Work](#-future-work)
8. [Author](#-author)

---

## 🔭 Project Overview

Uttar Pradesh is among India's most heatwave-prone states — a landlocked plain swept by hot *loo* winds from the Thar Desert, with over **200 million people**, many engaged in outdoor labour. The 2022 UP heatwave arrived weeks ahead of the historical season, exposing the critical gap between current forecasting capabilities and the early warning lead times that governments, hospitals, and disaster-response teams actually need.

This project addresses that gap by building a **two-stage hybrid deep learning system** capable of forecasting district-level maximum temperature (T2M_MAX) up to **10 days in advance**, at daily resolution, across **125 districts** spanning Uttar Pradesh and its climatologically-coupled neighbours (Rajasthan, Delhi, Madhya Pradesh, Bihar, Uttarakhand, Nepal).

**Key design goals:**
- ≥ 10-day forecast horizon to allow government mobilisation time
- District-level spatial resolution (not station or grid-point)
- Stable, monotonic error growth — no recursive error accumulation
- Modular two-stage architecture: temporal sequence model + tabular error corrector

---

## 🌶️ What is a Heatwave?

The **India Meteorological Department (IMD)** defines a heatwave when *both* conditions are met simultaneously:

| Condition | Threshold |
|---|---|
| Absolute temperature | T_max ≥ 40°C (plain stations) |
| Departure from normal | T_max ≥ 4.5°C above district climatological normal |

The **climatological normal** used in this project is the mean T2M_MAX for each district and calendar day-of-year, computed over the **1984–2015 baseline period** (32 years — exceeding the WMO standard of 30 years).

> **The 2022 UP Heatwave** — March–April 2022 brought one of the most severe and historically early heatwaves on record, with temperatures exceeding 44°C several weeks before the normal onset of the heat season. This event falls entirely within our **test set (2020–2025)** and serves as the primary real-world validation benchmark.

---

## 📦 Dataset

| Property | Details |
|---|---|
| **Source** | IMD Gridded Meteorological Data (NASA POWER / IMD) |
| **Geography** | 125 districts — UP + Rajasthan, Delhi, MP, Bihar, Uttarakhand, Nepal |
| **Time Period** | 1984-01-01 to 2025-12-30 (daily resolution) |
| **Total Records** | ~1,917,500 rows |
| **Target Variable** | `T2M_MAX` — Daily maximum 2m air temperature (°C) |

### Chronological Train / Validation / Test Split

| Split | Period | Rows | Purpose |
|---|---|---|---|
| **Train** | 1984 – 2010 | ~975,000 | LSTM training |
| **Validation** | 2011 – 2018 | ~360,375 | Hyperparameter tuning · XGBoost training on LSTM residuals |
| **Test** | 2019 – 2025 | ~314,625 | Final holdout — includes the 2022 UP heatwave |

> All splits are **strictly chronological** — no future information leaks into training.

### Feature Engineering

| Feature | Description | Notes |
|---|---|---|
| `T2M_MAX_lag1/3/7` | Autoregressive temperature lags | Strong temporal persistence |
| `T2M_MAX_roll7_mean` | 7-day rolling mean | Smoothed trend signal |
| `SOIL_DRYNESS_lag1` | Soil dryness (1-day lag) | Land-surface feedback |
| `ATM_DRYNESS_lag1` | Atmospheric dryness (1-day lag) | Low humidity amplifies heat |
| `RADIATION_EFFECTIVE_lag1` | Net effective radiation (1-day lag) | Energy balance signal |
| `RH2M`, `QV2M` | Relative & specific humidity | Heat stress indicators |
| `WS10M`, `WD10M_SIN/COS` | Wind speed & direction (circular encoded) | *Loo* wind signal |
| `PS` | Surface pressure | Synoptic-scale circulation |
| `PRECTOTCORR` | Precipitation | Pre-monsoon dryness proxy |
| `DOY_SIN / DOY_COS` | Circular day-of-year encoding | Seasonal cycle |
| `lat`, `lon` | District coordinates | Spatial context |

**Features dropped via VIF analysis:** `TS` (VIF 42.3), `CLRSKY_SFC_SW_DWN` (VIF 63.3), `GWETROOT` (VIF 56.6), `GWETTOP` (VIF 25.9) — all redundant with retained features.

---

## 🧠 Methodology

### Architecture Overview

```
Raw Meteorological Data (30 days × 22 features)
              │
              ▼
    ┌─────────────────┐
    │   LSTM Baseline │   → 10-day temperature forecast  [y_lstm]
    └─────────────────┘
              │
              ▼  residuals = y_true − y_lstm  (on validation set)
    ┌──────────────────────────┐
    │  XGBoost Residual Model  │   → 10 per-day correction terms  [y_xgb]
    │  (10 independent models) │
    └──────────────────────────┘
              │
              ▼
    y_final = y_lstm + y_xgb    ← Hybrid Prediction
```

---

### Stage 1 — LSTM Baseline: Temporal Sequence Modeling

**Sequence construction (direct multi-step forecasting):**

```
X_seq  →  [t-30, t-29, ..., t-1]   shape: (samples, 30, 22)
y_seq  →  [t, t+1, ..., t+9]       shape: (samples, 10)
```

- Sequences built **city-by-city** using a sliding window — no cross-district contamination
- **Direct forecasting** (not recursive): all 10 horizon days predicted in a single forward pass, eliminating error propagation

**LSTM Architecture:**

| Layer | Config |
|---|---|
| LSTM | 64 units |
| Dropout | 0.2 |
| Dense | 32 units, ReLU |
| Output | 10 units (one per forecast day), float32 |

**Training:**

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss | MSE |
| Batch size | 128 |
| Max epochs | 30 |
| Early stopping | patience = 15, restore best weights |
| LR scheduler | ReduceLROnPlateau — factor 0.5, patience 7, min LR 1e-6 |
| Precision | mixed_float16 (T4 GPU) |

---

### Stage 2 — XGBoost Residual Corrector

**Why XGBoost on validation residuals (not training residuals)?**

The LSTM partially memorises training sequences, making its training errors artificially small. Validation residuals reflect genuine LSTM weaknesses on truly unseen data — XGBoost trained here generalises properly to the test set.

**Feature construction for XGBoost:**

```
X_xgb  =  [X_last_timestep (22 features)  |  y_lstm_day_d (1 feature)]
                                                    = 23 features per sample
```

- `X_last_timestep` — the final day of the input window: *"what the atmosphere looks like right now"*
- `y_lstm_day_d` — the LSTM's own prediction for horizon day `d`: the most informative signal about where it might be wrong

**One model per horizon day (10 total)** — each day's error pattern is structurally different; a single model cannot capture this.

**XGBoost hyperparameters (regularised):**

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 100 | Fewer trees to avoid overfitting |
| `max_depth` | 3 | Shallow — learns only simple correction patterns |
| `learning_rate` | 0.05 | Conservative step size |
| `subsample` | 0.5 | Aggressive row subsampling |
| `colsample_bytree` | 0.5 | Feature subsampling |
| `min_child_weight` | 50 | Requires many samples per leaf |
| `reg_alpha` | 1.0 | L1 regularisation |
| `reg_lambda` | 5.0 | L2 regularisation |

---

### Hybrid Prediction Formula

```
y_final[d] = y_lstm[d] + y_xgb[d]     for d in {1, 2, ..., 10}
```

---

## 📊 Results

### LSTM Baseline — MAE per Forecast Day (Test Set: 2019–2025)

| Forecast Day | Validation MAE (°C) | Test MAE (°C) |
|:---:|:---:|:---:|
| Day 1 | 1.381 | 1.446 |
| Day 2 | 1.512 | 1.566 |
| Day 3 | 1.575 | 1.640 |
| Day 4 | 1.626 | 1.698 |
| Day 5 | 1.656 | 1.740 |
| Day 6 | 1.683 | 1.767 |
| Day 7 | 1.702 | 1.791 |
| Day 8 | 1.736 | 1.819 |
| Day 9 | 1.759 | 1.841 |
| Day 10 | 1.773 | 1.856 |
| **Average** | **1.640** | **1.716** |

> Error growth is smooth and monotonic — no recursive error explosion. Day 1–3 forecasts are within ≈1.4–1.6°C of observed temperatures.

---

### Hybrid Model (LSTM + XGBoost) — MAE per Forecast Day (Test Set)

| Forecast Day | LSTM MAE (°C) | Hybrid MAE (°C) | Improvement (°C) |
|:---:|:---:|:---:|:---:|
| Day 1 | 1.4135 | 1.3868 | ✅ +0.0267 |
| Day 2 | 1.5417 | 1.5314 | ✅ +0.0104 |
| Day 3 | 1.6240 | 1.6178 | ✅ +0.0062 |
| Day 4 | 1.6860 | 1.6739 | ✅ +0.0121 |
| Day 5 | 1.7357 | 1.7158 | ✅ +0.0200 |
| Day 6 | 1.7714 | 1.7456 | ✅ +0.0258 |
| Day 7 | 1.8106 | 1.7710 | ✅ +0.0395 |
| Day 8 | 1.8403 | 1.7949 | ✅ +0.0454 |
| Day 9 | 1.8576 | 1.8193 | ✅ +0.0383 |
| Day 10 | 1.8893 | 1.8362 | ✅ +0.0531 |
| **Average** | **1.7170** | **1.6893** | **✅ +0.0277** |

> XGBoost correction improves every single horizon day — with gains increasing at longer horizons (Day 7–10), exactly where the LSTM drifts most. The hybrid reduces overall MAE by **0.0277°C** relative to the standalone LSTM.

---

## 📁 Repository Structure

```
CRO-Heatwave-Project/
│
├── .gitignore                    # Ignore unnecessary files/folders
├── README.md
│
├── LSTM-baseline-model.ipynb     # Notebook 1 — EDA, feature engineering, LSTM training & evaluation
├── Hybrid-Model.ipynb            # Notebook 2 — XGBoost residual correction, hybrid evaluation
├── LICENSE                       # Open-source project license
├── CRO-Heatwave-Prediction.ipynb # Notebook 3 — Full CRO pipeline (IMD data, VIF, Mann-Kendall)

```

**Kaggle output artefacts (from Notebook 1, consumed by Notebook 2):**

| File | Shape | Description |
|---|---|---|
| `lstm_baseline_final.keras` | — | Saved LSTM model |
| `y_lstm_val.npy` | (360375, 10) | LSTM predictions — validation set |
| `y_lstm_test.npy` | (314625, 10) | LSTM predictions — test set |
| `y_val_seq.npy` | (360375, 10) | Ground truth — validation set |
| `y_test_seq.npy` | (314625, 10) | Ground truth — test set |
| `X_val_last.npy` | (360375, 22) | Last timestep features — validation |
| `X_test_last.npy` | (314625, 22) | Last timestep features — test |
| `xgb_day1.pkl … xgb_day10.pkl` | — | Trained XGBoost correctors (10 models) |
| `y_hybrid_test.npy` | (314625, 10) | Final hybrid predictions — test set |
| `mae_results.pkl` | — | MAE dictionary for both models |

---

## 🔮 Future Work

| Direction | Description |
|---|---|
| **Heatwave binary classification** | Convert T2M_MAX forecasts to IMD-compliant heatwave probability using district-level climatological normals |
| **Severity classification** | Three-class output — Moderate / Severe / Extreme — based on departure from climatological normal |
| **Graph Neural Networks** | Model spatial propagation of heat across district boundaries (Rajasthan → western UP → central UP) |
| **Attention maps** | Identify which source districts and historical timesteps drive heatwave predictions — for interpretability in government briefings |
| **Confidence intervals** | Quantile regression or conformal prediction wrappers for uncertainty-aware forecasts |
| **Extended horizon** | Push forecast window from 10 to 15 days to match operational CRO requirements |
| **Operational deployment** | Automated daily inference pipeline on fresh NASA POWER data with Streamlit/dashboard output for district collectors |

---

## 👤 Author

**Sunil Kumar**
AI/ML Engineer · Climate & Forecasting Systems

[![GitHub](https://img.shields.io/badge/GitHub-sunilsaini01-181717?style=flat&logo=github)](https://github.com/sunilsaini01)
[![Kaggle](https://img.shields.io/badge/Kaggle-sunil123kumar-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/)

---

> *This project is part of ongoing research in AI-driven climate resilience for the Indo-Gangetic plains. The system is designed to support — not replace — meteorological expertise and government decision-making.*
