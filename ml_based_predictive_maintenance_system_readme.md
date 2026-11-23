# PREDX: ML-Based Predictive Maintenance System

An intelligent, data-driven predictive maintenance and equipment failure prediction platform built using Machine Learning and web technologies. **PREDX** analyzes industrial equipment operational parameters to estimate failure probabilities, categorize risk levels, and generate actionable maintenance alerts before unexpected breakdowns occur.

## 📋 Table of Contents

* [Overview](#-overview)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Tech Stack](#-tech-stack)
* [Project Structure](#-project-structure)
* [Dataset Format](#-dataset-format)
* [API Endpoints](#-api-endpoints)
* [Monitored Parameters & Alert Thresholds](#-monitored-parameters--alert-thresholds)
* [Getting Started](#-getting-started)
* [Future Scope](#-future-scope)
* [License](#-license)

## 🔍 Overview

Traditional reactive and fixed-schedule preventive maintenance strategies often result in either costly unplanned downtime or unnecessary servicing. **PREDX** transitions industrial maintenance toward a proactive, predictive approach.

By processing historical and operational equipment telemetry through a **Random Forest Classifier**, the system detects hidden patterns of degradation, classifies machinery into distinct risk tiers (**Low, Medium, High, Critical**), and presents real-time diagnostics via an interactive web dashboard.

## ✨ Key Features

* **Early Failure Prediction:** Uses a Random Forest Classification model to compute exact equipment failure probabilities (`0%` to `100%`).
* **Automated Risk Categorization:** Dynamically classifies equipment health into **Low**, **Medium**, **High**, and **Critical** risk levels to help maintenance teams prioritize interventions.
* **Intelligent Alert Generation:** Automatically flags parameter anomalies (e.g., high temperature, excessive vibration, low oil levels) with specific warning and critical thresholds.
* **Batch & Single-Unit Processing:** Supports drag-and-drop batch uploads via `.csv` or `.xlsx` files (capable of handling 50,000+ records) as well as real-time single-unit parameter evaluation.
* **Interactive Dashboard & Reporting:** Visualizes unit health distribution, top alert triggers, diagnostics by equipment type (Boilers, Compressors, Motors, Heat Exchangers, Fans, Pumps), and allows one-click report exports.

## 🏗 System Architecture

The application follows a modular client-server workflow divided into five core components:

1. **Data Processing Module:** Handles missing values, data cleaning, feature standardization, and feature engineering (generating stress scores and efficiency metrics).
2. **ML Prediction Module:** Loads the serialized Scikit-learn Random Forest model to infer failure probabilities.
3. **Alert Generation Module:** Evaluates operational telemetry against safety thresholds to produce granular alert logs.
4. **Flask API Module:** Serves as the RESTful communication layer between the ML engine and the user interface.
5. **Dashboard Visualization Module:** A responsive frontend built with HTML, CSS, and JavaScript for file uploads, real-time monitoring, and report exports.

## 🛠 Tech Stack

* **Backend:** Python, Flask
* **Machine Learning & Data Processing:** Scikit-learn, Pandas, NumPy, Joblib
* **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
* **Supported Data Formats:** CSV (`.csv`), Excel (`.xlsx`)

## 📁 Project Structure

```text
ml-predictive-maintenance/
├── app.py                              # Flask application & REST API endpoints
├── train_model.py                      # Data preprocessing, feature engineering & model training
├── models/
│   └── rf_predictive_model.joblib      # Serialized Random Forest model
├── data/
│   └── equipment_maintenance_data.xlsx # Sample equipment telemetry dataset
├── static/
│   ├── css/
│   │   └── style.css                   # Dashboard styling
│   └── js/
│       └── main.js                     # Frontend logic, file upload & dynamic rendering
├── templates/
│   └── index.html                      # Web dashboard interface
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

## 📄 Dataset Format

To run batch predictions via the dashboard or `/predict-file` endpoint, upload a `.csv` or `.xlsx` file containing the following columns:

| Equipment_ID | Equipment_Type | Temperature | Pressure | Vibration | Running_Hours | Current | Power_Consumption | Oil_Level |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `H190` | `Heat Exchanger` | `97.3` | `7.8` | `4.5` | `3800` | `35.1` | `145.0` | `29.0` |
| `M200` | `Motor` | `108.1` | `8.2` | `5.5` | `4200` | `32.0` | `152.4` | `30.0` |
| `P556` | `Pump` | `84.0` | `9.4` | `3.2` | `4871` | `28.5` | `130.0` | `27.0` |

## 🔌 API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/predict` | `POST` | Accepts JSON/form input values for a single equipment unit and returns failure probability, risk category, and alerts. |
| `/predict-file` | `POST` | Accepts `.csv` or `.xlsx` datasets for bulk equipment processing and returns aggregated diagnostics and unit-level risk registers. |
| `/health` | `GET` | Monitors backend server and prediction engine operational status. |

## 📊 Monitored Parameters & Alert Thresholds

The model evaluates the following operational features across industrial equipment types (*Boiler, Compressor, Motor, Heat Exchanger, Fan, Pump*):

* **Temperature:** Warning at $\ge 90$, Critical at $\ge 105$
* **Vibration:** Warning at $\ge 4.5$, Critical at $\ge 5.5$
* **Running Hours:** Warning at $\ge 4000$, Critical at $\ge 4800$
* **Oil Level:** Warning at $\le 30$, Critical at $\le 20$
* **Pressure:** Warning at $\ge 8.5$
* **Current Consumption:** Warning at $\ge 35$, Critical at $\ge 39.8$
* **Power Consumption:** Warning at $\ge 160$

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/ml-predictive-maintenance.git
   cd ml-predictive-maintenance
   ```

2. **Install the required dependencies:**

   ```bash
   pip install flask pandas numpy scikit-learn joblib openpyxl
   ```

3. **Run the application:**

   ```bash
   python app.py
   ```

4. **Access the Dashboard:**
   Open your browser and navigate to `http://127.0.0.1:5000` to upload your equipment dataset (e.g., `equipment_maintenance_data.xlsx`) and view predictions.

## 🔮 Future Scope

* **IoT Sensor Integration:** Direct streaming of real-time telemetry from industrial IoT sensors to eliminate manual file uploads.
* **Cloud Deployment:** Hosting on cloud infrastructure for centralized, multi-site equipment monitoring and scalability.
* **Deep Learning Models:** Incorporating neural networks and time-series forecasting (e.g., LSTM) to capture complex degradation trends.
* **Real-Time Push Notifications:** Instant email/SMS alert dispatches when critical equipment thresholds are breached.
* **Mobile Companion App:** Cross-platform mobile interface for on-floor maintenance engineers.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.