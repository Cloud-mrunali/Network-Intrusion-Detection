# 🛡️ Network Intrusion Detection System (NIDS)
### IBM AICTE Internship — Problem Statement #40

A production-ready **Machine Learning-based Network Intrusion Detection System** that classifies network traffic into five categories: **Normal, DoS, Probe, R2L, and U2R** — powered by **IBM Cloud Lite Services**.

---

## 🎯 Problem Statement

Build a robust NIDS using ML to analyze network traffic data, identify and classify various types of cyber-attacks, and distinguish them from normal activity — providing early warning of malicious behavior.

---

## 🏗️ Architecture

```
nids_project/
├── app.py                     ← Flask REST API + Dashboard
├── train.py                   ← CLI training entry-point
├── requirements.txt
├── .env                       ← IBM Cloud + App credentials
│
├── src/
│   ├── preprocessor.py        ← Data cleaning, encoding, scaling, SMOTE
│   ├── trainer.py             ← RF, XGBoost, Ensemble training & evaluation
│   └── ibm_watson.py          ← IBM WML, COS, Cloudant integrations
│
├── notebooks/
│   └── NIDS_EDA_Training.ipynb ← EDA, visualization, full training walkthrough
│
├── templates/
│   └── index.html             ← Dashboard UI
│
├── static/
│   ├── css/style.css
│   └── js/main.js
│
├── models/                    ← Saved model artefacts (auto-generated)
├── data/                      ← Place downloaded CSV here
└── logs/                      ← Runtime logs
```

---

## 🤖 Machine Learning Models

| Model | Description |
|-------|-------------|
| **Random Forest** | 200 trees, balanced class weights |
| **XGBoost** | Gradient boosted trees, mlogloss metric |
| **Decision Tree** | Baseline, depth-20 |
| **Ensemble (RF+XGB)** | Soft-voting ensemble ⭐ Best |

### Attack Classes
| Family | Examples |
|--------|----------|
| `normal` | Legitimate connections |
| `DoS` | smurf, neptune, back, teardrop, land |
| `Probe` | ipsweep, nmap, portsweep, satan |
| `R2L` | guess_passwd, ftp_write, warezclient |
| `U2R` | buffer_overflow, rootkit, loadmodule |

---

## ☁️ IBM Cloud Lite Services

| Service | Usage | Tier |
|---------|-------|------|
| **Watson Machine Learning** | Deploy & score ML model via REST API | Lite (Free) |
| **IBM Cloud Object Storage** | Store dataset + model artefacts | Lite (25 GB free) |
| **IBM Cloudant** | NoSQL logging of detection events | Lite (1 GB free) |

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone <repo-url>
cd nids_project
pip install -r requirements.txt
```

### 2. Configure Environment
Edit `.env` with your IBM Cloud credentials:
```ini
IBM_CLOUD_API_KEY=your_api_key
IBM_WML_URL=https://us-south.ml.cloud.ibm.com
IBM_WML_SPACE_ID=your_space_id
IBM_WML_DEPLOYMENT_ID=your_deployment_id   # after deploying
IBM_CLOUDANT_URL=https://...
IBM_CLOUDANT_APIKEY=...
IBM_COS_API_KEY=...
```

### 3. Download Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection):
```
Train_data.csv  →  data/Train_data.csv
Test_data.csv   →  data/Test_data.csv
```

### 4. Train the Model
```bash
python train.py --data data/Train_data.csv
# With SMOTE disabled (faster):
python train.py --data data/Train_data.csv --no-smote
# Also upload artefacts to IBM COS:
python train.py --data data/Train_data.csv --upload-cos
```

### 5. Launch the Dashboard
```bash
python app.py
# Open: http://localhost:5000
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/health` | Health check & model status |
| POST | `/api/predict` | Single connection prediction |
| POST | `/api/predict/batch` | Batch prediction (JSON array) |
| POST | `/api/predict/wml` | Score via IBM WML endpoint |
| GET  | `/api/metrics` | Saved model performance metrics |
| GET  | `/api/alerts` | Recent detection alerts |
| GET  | `/api/attack_distribution` | Attack family counts |

### Sample Request
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "protocol_type": "tcp",
    "service": "http",
    "flag": "S0",
    "src_bytes": 0,
    "dst_bytes": 0,
    "count": 511,
    "serror_rate": 1.0,
    "same_srv_rate": 1.0
  }'
```

### Sample Response
```json
{
  "prediction": "neptune",
  "attack_family": "DoS",
  "is_attack": true,
  "confidence": 0.9987,
  "color": "#ef4444",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 📊 Exploratory Data Analysis

Run the Jupyter notebook for:
- Dataset overview & statistics
- Attack family distribution plots
- Feature correlation heatmap
- Model comparison table
- Confusion matrix
- Feature importance chart
- IBM Cloud integration walkthrough

```bash
cd notebooks
jupyter notebook NIDS_EDA_Training.ipynb
```

---

## 🔧 IBM Watson ML Deployment Workflow

```python
from src.ibm_watson import WatsonMLService

wml = WatsonMLService()
wml.connect()

# 1. Store trained model
model_uid = wml.store_model('models/nids_model.pkl', model_name='NIDS-v1')

# 2. Deploy online endpoint
dep_uid = wml.deploy(model_uid, deployment_name='NIDS-Online')

# 3. Score (add dep_uid to .env as IBM_WML_DEPLOYMENT_ID)
result = wml.predict([[0, 1, 2, 0, 0, 0, ...]])
```

---

## 📦 Dataset Features (KDD Cup 1999 — 41 Features)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | duration | numeric | Length of connection (seconds) |
| 2 | protocol_type | categorical | tcp / udp / icmp |
| 3 | service | categorical | http, ftp, smtp, … |
| 4 | flag | categorical | Connection status (SF, S0, REJ, …) |
| 5 | src_bytes | numeric | Bytes from source to dest |
| 6 | dst_bytes | numeric | Bytes from dest to source |
| … | … | … | … |
| 41 | label | target | Attack type |

---

## 📋 Requirements

- Python 3.10+
- IBM Cloud account (Free / Lite tier)
- Watson Machine Learning service instance
- IBM Cloud Object Storage instance (optional)
- IBM Cloudant instance (optional)

---

## 📄 License

MIT License — IBM AICTE Internship Project 2026
