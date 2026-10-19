The `README.md` file should be placed at the **root directory** of your project folder (e.g., inside `accident-severity/` alongside `.gitignore`, `backend/`, `frontend/`, and `data/`).

```text
accident-severity/
├── README.md               <-- Place it here in the project root
├── .gitignore
├── data/
│   ├── raw/
│   │   └── US_Accidents_March23.csv
│   └── processed/
├── backend/
└── frontend/

```

---

### Complete Production `README.md` Content

Create a file named `README.md` in your project root and paste the following content:

```markdown
# RescuGuard AI — Road Traffic Accident Severity Predictor & Emergency Response Engine

> **INSA Cyber Talent Summer Camp — Final Project**  
> *A high-performance decision-support engine for real-time emergency triage and automated emergency resource allocation.*

---

## 📌 Project Overview

**RescuGuard AI** is a production-grade MVP that leverages Machine Learning and modern web APIs to predict road accident severity (Levels 1–4) in real time. Rather than operating purely as an offline analytical model, it acts as an intelligent decision-support layer for emergency command centers by automatically transforming incident parameters (location, weather, visibility, road geometry) into actionable emergency dispatch protocols.

### Key Capabilities
- **Real-Time Inference API:** Built on **FastAPI** for ultra-low latency prediction handling.
- **Large Dataset Pipeline:** Optimized scikit-learn pipeline tailored for high-volume datasets (including the 3 GB Kaggle US Accidents dataset with ~7.7M records).
- **Automated Dispatch Triage:** Maps predicted severity levels directly to recommended emergency assets (ambulances, trauma units, traffic police, heavy rescue).
- **Modern Control Dashboard:** Clean, responsive, glassmorphism-inspired UI for dispatchers and command centers.

---

## 🇪🇹 Strategic Vision & Relevance to Ethiopia

Traffic incidents along major freight arteries (such as the **Addis Ababa–Djibouti Logistics Corridor**) and dense urban routes in **Addis Ababa** cause severe emergency delays and loss of life during the critical "Golden Hour."

Positioned within **INSA’s (Information Network Security Administration)** mandate for national technological sovereignty and smart city infrastructure:
1. **Command & Control Integration (C4ISR):** Serves as a predictive module for centralized emergency command centers.
2. **Resource Optimization:** Maximizes the impact of finite public safety resources (ambulances and rescue teams) through automated priority triaging.
3. **Sovereign & Localizable Architecture:** Built with modular open-source tooling, allowing immediate drop-in replacement with localized Ethiopian traffic incident datasets.

---

##  Project Architecture & Directory Structure

```text
accident-severity/
├── README.md                  # Project documentation & presentation guide
├── .gitignore                 # Version control exclusions (venv, models, raw data)
├── data/
│   ├── raw/                   # Place 'US_Accidents_March23.csv' here
│   └── processed/             # Holds exported model artifact ('model_pipeline.joblib')
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI entry point & static UI mount
│   │   ├── config.py          # Environment & application paths
│   │   ├── schemas/           # Pydantic data validation schemas
│   │   │   └── accident.py
│   │   ├── services/          # Model inference & priority mapping engine
│   │   │   └── predictor.py
│   │   └── api/               # API endpoints (/predict, /health)
│   │       └── endpoints.py
│   ├── train.py               # Memory-optimized ML pipeline trainer script
│   └── requirements.txt       # Python dependencies
└── frontend/
    ├── index.html             # Emergency Command Center dashboard
    ├── css/
    │   └── styles.css         # Modern dark-mode UI styling
    └── js/
        └── app.js             # Async REST API fetch & DOM renderer

```

---

## Setup & Installation Guide

### 1. Prerequisites

* **Python 3.9+**
* **Git**

### 2. Environment Setup

Clone the repository and set up a virtual environment:

```bash
# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

# Upgrade core build tools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r backend/requirements.txt

```

---

## Data Setup & Model Training

1. Download `US_Accidents_March23.csv` (~3 GB) and place it in the `data/raw/` directory:
```text
data/raw/US_Accidents_March23.csv

```


2. Run the memory-optimized trainer:
```bash
cd backend
python train.py

```


*Note: By default, `train.py` samples 300,000 rows for rapid model training on standard developer laptops. Set `sample_size=None` in `train.py` to train on all 7.7M rows if running on a high-RAM server.*

---

## Running the Application

Launch the FastAPI backend server:

```bash
# Executed from the backend/ directory
uvicorn app.main:app --reload --port 8000

```

* **Dashboard UI:** Open `http://127.0.0.1:8000` in your web browser.
* **Interactive API Docs (Swagger UI):** Access `http://127.0.0.1:8000/docs`.

---

## API Endpoint Specification

### `POST /api/v1/predict`

**Sample Request Body:**

```json
{
  "Start_Lat": 39.8651,
  "Start_Lng": -84.0587,
  "Temperature(F)": 36.9,
  "Humidity(%)": 91.0,
  "Visibility(mi)": 10.0,
  "Bump": 0,
  "Crossing": 1,
  "Traffic_Signal": 1,
  "Weather_Condition": "Clear",
  "Sunrise_Sunset": "Day"
}

```

**Sample Response Body:**

```json
{
  "predicted_severity": 4,
  "severity_label": "Critical Highway Emergency (Level 4)",
  "priority_level": "CRITICAL",
  "response_protocol": "Mass Casualty & Major Arterial Emergency",
  "probabilities": {
    "Severity 1": 0.021,
    "Severity 2": 0.115,
    "Severity 3": 0.214,
    "Severity 4": 0.650
  },
  "dispatch_recommendation": [
    "Immediate Trauma Dispatch",
    "Fire & Extrication Response",
    "Full Sector Traffic Reroute"
  ]
}

```

---

##  Built With

* **Backend:** Python 3, FastAPI, Pydantic, Scikit-Learn, Pandas, Joblib
* **Frontend:** Vanilla HTML5, CSS3 (CSS Variables & Grid), Modern JavaScript (ES6 Async/Fetch)
* **Deployment & Server:** Uvicorn, StaticFiles
* **Developed by:** Fisehatsion Adisu from AASTU

```

```