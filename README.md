<div align="center">

# FraudShield AI

**Real-time financial fraud detection, powered by XGBoost and served over a FastAPI backend.**

<sub>Trained on 6.3M+ PaySim transactions · Sub-100ms inference · Precision-recall–calibrated decisioning</sub>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](.)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](.)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-AA4400?style=flat-square)](.)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](.)

</div>

---

### What this is

A transaction comes in — a type, an amount, a pair of before/after balances — and in under 100ms, FraudShield returns a fraud probability, a risk tier, the specific signals that drove the score, and a plain-language summary of why. It's built the way a fraud team would actually want to consume it: not just a number, but a reason.

The model is an `XGBoost` classifier trained on the PaySim simulated mobile-money dataset, wrapped in a `FastAPI` service, with a self-contained dashboard for exploring predictions interactively.

<br>

## Table of contents

- [Architecture](#architecture)
- [How a prediction is made](#how-a-prediction-is-made)
- [Model performance](#model-performance)
- [A note on the decision threshold](#a-note-on-the-decision-threshold)
- [Project layout](#project-layout)
- [Getting started](#getting-started)
- [API reference](#api-reference)
- [Roadmap](#roadmap)
- [Author](#author)

<br>

## Architecture

```
                    ┌──────────────────────┐
                    │   PaySim Dataset      │
                    │   6.3M transactions   │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Feature Engineering  │   type encoding · log-scaled amounts
                    │   (models/features.py)│   balance deltas · drain/anomaly flags
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │   Standard Scaling    │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │   XGBoost Classifier  │   (models/train.py)
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Threshold Calibration│   precision–recall curve, F1-optimal cutoff
                    └───────────┬──────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                     │
   ┌──────────▼──────────┐              ┌──────────▼──────────┐
   │   FastAPI Service    │              │   Static Dashboard   │
   │   (api/app.py)        │◄────calls───┤   (frontend/index.html)│
   │   /predict             │              │   live probability UI │
   │   /predict/batch       │              └────────────────────┘
   │   /health · /model/info│
   └────────────────────────┘
```

<br>

## How a prediction is made

1. **Feature engineering** — a raw transaction is expanded into 11 model features: type encoding, log-scaled amounts and balances, balance deltas, an "account fully drained" flag, an amount-to-balance ratio, and a destination-balance anomaly flag. See `models/features.py`.
2. **Scaling & inference** — features are standardized with the fitted `scaler.pkl` and passed through the trained XGBoost model to get a fraud probability.
3. **Decisioning** — the probability is compared against a calibrated threshold to produce a binary verdict, and bucketed into a risk tier (`LOW` / `MEDIUM` / `HIGH` / `CRITICAL`).
4. **Explanation** — a small set of interpretable rules (large amount, high-risk transfer type, drained sender, brand-new receiver) surface the *why* behind the score, and a natural-language summary is generated from them. This layer is currently a deterministic rule-based template, not an LLM call — worth knowing if you're extending it.

<br>

## Model performance

Evaluated on a held-out stratified sample (all fraud cases + 300K legitimate transactions) at the production decision threshold:

| Metric | Score |
|---|---|
| ROC-AUC | 0.9999 |
| PR-AUC | 0.9996 |
| Precision | 99.24% |
| Recall | 99.72% |
| F1 | 0.9948 |

<br>

## A note on the decision threshold

Training originally picked the F1-optimal threshold off the precision-recall curve, which landed at **0.975**. In practice this meant a transaction the model scored at 96.66% fraud probability — sender account fully drained, brand-new receiver, large transfer — still came back as *"legitimate"*, because it hadn't cleared that unusually high bar. Technically defensible on paper, actively misleading in a demo.

Re-evaluating both thresholds against the trained model on the full sample:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| **0.50 (current)** | 99.24% | 99.72% | **0.9948** |
| 0.975 (original) | 99.90% | 97.67% | 0.9877 |

0.5 wins outright — better F1 *and* meaningfully better recall (fewer missed frauds), for a negligible precision cost. The threshold now lives in `models/threshold.pkl` and is loaded at inference time, so it can be recalibrated without retraining the model.

<br>

## Project layout

```
fraud-detection-api/
├── api/
│   └── app.py                  FastAPI app — routes, schemas, CORS
├── models/
│   ├── train.py                Training pipeline: scale → SMOTE → XGBoost → threshold
│   ├── main.py                 Inference engine: predict_fraud()
│   ├── features.py             Feature engineering (FEATURE_COLS)
│   ├── xgb_fraud.json          Trained model weights
│   ├── scaler.pkl              Fitted StandardScaler
│   ├── threshold.pkl           Calibrated decision threshold
│   ├── feature_names.pkl
│   └── feature_importance.csv
├── frontend/
│   └── index.html              Self-contained interactive dashboard
├── data/
│   └── paysim.csv              Source dataset (not tracked — see below)
├── tests/
├── screenshots/
├── requirements.txt
└── README.md
```

> `data/paysim.csv` isn't included in distributed copies of this repo (it's ~470MB). Grab the [PaySim dataset from Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and drop it in `data/` before running `train.py`. It's only needed for retraining — the trained artifacts in `models/` are enough to run the API as-is.

<br>

## Getting started

**1. Clone and enter the project**
```bash
git clone https://github.com/Aarya0706/fraud-detection-api.git
cd fraud-detection-api
```

**2. Create a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell
source .venv/bin/activate       # macOS / Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. (Optional) Retrain the model**

Only needed if you want to reproduce or modify the model — trained artifacts already ship in `models/`.
```bash
python models/train.py
```

**5. Start the API**
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

**6. Open the dashboard**

Open `frontend/index.html` in your browser — it talks to the API at `localhost:8000`.

<br>

## API reference

### `POST /predict`

**Request**
```json
{
  "type": "TRANSFER",
  "amount": 800000,
  "oldbalanceOrg": 900000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 0,
  "newbalanceDest": 800000
}
```

**Response**
```json
{
  "fraud_probability": 0.9666,
  "confidence": "96.66%",
  "threshold": "50%",
  "is_fraud": true,
  "risk_level": "CRITICAL",
  "model": "XGBoost Fraud Classifier v1.0",
  "top_risk_factors": [
    "Large transaction amount",
    "High-risk transaction type (TRANSFER)",
    "Sender account fully drained",
    "Destination account has zero previous balance"
  ],
  "summary": "This TRANSFER transaction of ₹800,000.00 has been flagged as CRITICAL risk with a fraud probability of 96.66%. Key risk indicators include: Large transaction amount, High-risk transaction type (TRANSFER), Sender account fully drained, Destination account has zero previous balance."
}
```

### Other endpoints

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/predict/batch` | Score up to 500 transactions in one call |
| `GET` | `/health` | Service liveness + uptime |
| `GET` | `/model/info` | Model metadata (algorithm, training size, metrics) |

Interactive docs are auto-generated by FastAPI at `/docs` once the server is running.

<br>

## Roadmap

- [ ] Real LLM-generated summaries (the `openai` dependency is already in `requirements.txt`, not yet wired up)
- [ ] SHAP-based per-prediction explainability
- [ ] Dockerized deployment
- [ ] Transaction history / persistent storage
- [ ] Auth-gated admin dashboard

<br>

## Author

**Aarya Shirsath**
B.Tech Computer Science · VIT Bhopal

[GitHub](https://github.com/Aarya0706)

<br>

<sub>Made with ♥ by Aarya</sub>

</div>