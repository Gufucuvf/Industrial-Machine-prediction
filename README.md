Overview
Traditional reactive and fixed-schedule preventive maintenance strategies often result in either costly unplanned downtime or unnecessary servicing. PREDX transitions industrial maintenance toward a proactive, predictive approach.By processing historical and operational equipment telemetry through a Random Forest Classifier, the system detects hidden patterns of degradation, classifies machinery into distinct risk tiers (Low, Medium, High, Critical), and presents real-time diagnostics via an interactive web dashboard
Key Features
Early Failure Prediction: Uses a Random Forest Classification model to compute exact equipment failure probabilities (0% to 100%).
Automated Risk Categorization: Dynamically classifies equipment health into Low, Medium, High, and Critical risk levels to help maintenance teams prioritize interventions.
Intelligent Alert Generation: Automatically flags parameter anomalies (e.g., high temperature, excessive vibration, low oil levels) with specific warning and critical thresholds.
Batch & Single-Unit Processing: Supports drag-and-drop batch uploads via .csv or .xlsx files (capable of handling 50,000+ records) as well as real-time single-unit parameter evaluation.
Interactive Dashboard & Reporting: Visualizes unit health distribution, top alert triggers, diagnostics by equipment type (Boilers, Compressors, Motors, Heat Exchangers, Fans, Pumps), and allows one-click report exports.
System Architecture
The application follows a modular client-server workflow divided into five core components:
Data Processing Module: Handles missing values, data cleaning, feature standardization, and feature engineering (generating stress scores and efficiency metrics).
ML Prediction Module: Loads the serialized Scikit-learn Random Forest model to infer failure probabilities.
Alert Generation Module: Evaluates operational telemetry against safety thresholds to produce granular alert logs.
Flask API Module: Serves as the RESTful communication layer between the ML engine and the user interface.
Dashboard Visualization Module: A responsive frontend built with HTML, CSS, and JavaScript for file uploads, real-time monitoring, and report exports.
Tech Stack
Backend: Python, Flask
Machine Learning & Data Processing: Scikit-learn, Pandas, NumPy, Joblib
Frontend: HTML5, CSS3, JavaScript (Vanilla)
Supported Data Formats: CSV (.csv), Excel (.xlsx)
Project Structure
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
Getting Started
PrerequisitesEnsure you have Python 3.8+ installed on your system.
Installation
Clone the repository:git clone https://github.com/your-username/ml-predictive-maintenance.git
cd ml-predictive-maintenance
Install the required dependencies:
pip install flask pandas numpy scikit-learn joblib openpyxl
Run the application:
python app.py
Access the Dashboard:
Open your browser and navigate to http://127.0.0.1:5000 to upload your equipment dataset (e.g., equipment_maintenance_data.xlsx) and view predictions.
