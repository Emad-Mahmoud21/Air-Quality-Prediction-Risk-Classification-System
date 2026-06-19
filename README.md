# Air Quality Prediction & Risk Classification System

A machine learning graduation project that predicts Air Quality Index (AQI) values,
classifies pollution risk levels, and discovers pollution patterns using clustering.

---

## Project Structure

```
air_quality_project/
│
├── air_quality_notebook.ipynb   ← Full ML pipeline (EDA, training, evaluation)
├── api.py                       ← FastAPI backend (serves predictions)
├── app.py                       ← Streamlit frontend (interactive dashboard)
├── requirements.txt             ← Python dependencies
│
└── models/                      ← Auto-created after running the notebook
    ├── regressor.pkl
    ├── classifier.pkl
    ├── kmeans.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    └── feature_cols.pkl
```

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

- Go to: https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset
- Download `global air pollution dataset.csv`
- Place it in the **same folder** as the notebook

### 3. Run the notebook

Open `air_quality_notebook.ipynb` in Jupyter or VS Code and run all cells.
This will train the models and save them to the `models/` folder.

```bash
jupyter notebook air_quality_notebook.ipynb
```

### 4. Start the FastAPI backend

```bash
uvicorn api:app
```

API docs available at: http://localhost:8000/docs

### 5. Launch the Streamlit dashboard

In a **new terminal**:

```bash
streamlit run app.py
```

Dashboard available at: http://localhost:8501

---

## ML Models Used

| Task | Model | Output |
|------|-------|--------|
| Regression | Random Forest / XGBoost / Linear Regression | AQI numeric value |
| Classification | Random Forest / Logistic Regression / SVM | Safe / Moderate / Dangerous |
| Clustering | K-Means | Pollution pattern group |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/predict/aqi` | Predict AQI value (regression) |
| POST | `/predict/risk` | Predict risk level (classification) |
| POST | `/predict/cluster` | Assign pollution cluster |
| POST | `/predict/all` | Run all models at once |

### Example Request

```json
POST /predict/all
{
  "co_aqi": 2.0,
  "ozone_aqi": 45.0,
  "no2_aqi": 12.0,
  "pm25_aqi": 68.0
}
```

### Example Response

```json
{
  "aqi_value": 72.4,
  "interpretation": "Moderate — Acceptable but some pollutants may concern sensitive groups.",
  "risk_level": "Moderate",
  "color_code": "#F39C12",
  "cluster_id": 1,
  "input_values": {
    "CO AQI": 2.0,
    "Ozone AQI": 45.0,
    "NO2 AQI": 12.0,
    "PM2.5 AQI": 68.0
  }
}
```

---

## Dataset

**Global Air Pollution Dataset**
- Source: Kaggle (hasibalmuzdadid)
- Features: CO AQI, Ozone AQI, NO2 AQI, PM2.5 AQI, AQI Category
- Target: AQI Value (regression) / Risk Level (classification)
