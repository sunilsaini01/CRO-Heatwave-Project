# Heatwave Prediction System

### AI-Driven District-Level Heatwave Forecasting for Uttar Pradesh, India

> A hybrid deep learning system designed for early heatwave prediction using LSTM sequence modeling and XGBoost residual correction — built as part of the Climate Resilience Observatory (CRO) initiative for climate-risk forecasting and decision support.

---

## Overview

Extreme heat events are becoming more frequent, earlier, and more severe across India.  
This project focuses on building a scalable district-level forecasting system capable of predicting maximum temperature trends up to **10 days in advance** across **125 districts** in and around Uttar Pradesh.

The system combines:

- **LSTM-based temporal forecasting**
- **XGBoost residual correction**
- **Large-scale meteorological feature engineering**
- **Chronological climate forecasting pipelines**

with the goal of supporting:
- early warning systems
- climate resilience planning
- disaster response preparation
- heatwave risk analysis

---

## Key Highlights

- Forecast horizon: **10 days**
- Coverage: **125 districts**
- Dataset size: **~1.9 million records**
- Time span: **1984–2025**
- Hybrid architecture: **LSTM + XGBoost**
- Chronological split to avoid data leakage
- Stable long-horizon forecasting performance
- Built for operational climate-risk forecasting workflows

---

## Tech Stack

### Machine Learning & Deep Learning
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC6B23?style=flat)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

### Data & Research
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## System Architecture

```text
Meteorological Time-Series Data
                │
                ▼
        LSTM Forecast Model
                │
                ▼
       Residual Error Modeling
             (XGBoost)
                │
                ▼
      Final Hybrid Predictions
```

### Why Hybrid Modeling?

The LSTM captures:
- temporal dependencies
- seasonality
- long-term weather patterns

XGBoost then learns:
- systematic forecast errors
- nonlinear correction patterns
- horizon-specific residual behavior

This improves stability and long-range forecast accuracy.

---

## Dataset

| Property | Details |
|---|---|
| Source | IMD + NASA POWER |
| Geography | Uttar Pradesh + neighboring regions |
| Resolution | Daily |
| Time Period | 1984 – 2025 |
| Records | ~1.9M |
| Target | Daily Maximum Temperature (`T2M_MAX`) |

### Feature Engineering

Key engineered features include:

- temperature lag features
- rolling averages
- humidity metrics
- soil dryness indicators
- wind direction encoding
- seasonal cyclical encoding
- spatial coordinates

---

## Model Pipeline

### Stage 1 — LSTM Forecasting

- Direct multi-step forecasting
- 30-day input sequence
- 10-day output horizon
- Non-recursive forecasting approach
- Mixed precision GPU training

### Stage 2 — XGBoost Residual Correction

- One residual model per forecast day
- Learns systematic LSTM errors
- Improves long-horizon stability
- Regularized shallow-tree configuration

---

## Results

### Hybrid Model Performance

| Metric | Value |
|---|---|
| Average Test MAE | **1.689°C** |
| Forecast Horizon | **10 Days** |
| Spatial Coverage | **125 Districts** |

### Key Observation

The hybrid architecture consistently improves performance across all forecast horizons, with the largest gains appearing in longer-range predictions where standalone sequence models typically drift.

---

## Repository Structure

```bash
├── LSTM-baseline-model.ipynb
├── Hybrid-Model.ipynb
├── CRO-Heatwave-Prediction.ipynb
└── README.md
```

---

## Research & Engineering Focus

This project sits at the intersection of:

- Climate AI
- Time-Series Forecasting
- Deep Learning
- Environmental Intelligence
- Spatiotemporal Modeling
- AI for Public Infrastructure

---

## Future Directions

- Heatwave probability classification
- Graph Neural Networks for spatial heat propagation
- Forecast uncertainty estimation
- Attention-based interpretability
- Operational deployment pipelines
- Real-time dashboard integration

---

## About Me

**Sunil Kumar**  
AI/ML Engineer focused on:
- Climate Intelligence
- Forecasting Systems
- Applied Deep Learning
- Data Science & Statistical Modeling

I enjoy building research-oriented AI systems that combine real-world impact with scalable engineering.

### Connect With Me

- GitHub: https://github.com/sunilsaini01
- Kaggle: https://www.kaggle.com/

---

## Project Vision

This work is part of ongoing exploration into AI-driven climate resilience systems for large-scale public forecasting and decision support applications.
