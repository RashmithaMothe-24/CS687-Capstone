How to Run the Project

Create Virtual Environment

Run: python3 -m venv .venv

Activate:

Mac/Linux: source .venv/bin/activate

Windows: .venv\Scripts\activate

Install Dependencies

Run: pip install -r requirements.txt

Run Backend (Data Preparation)

Navigate to the backend folder: cd backend

Open the Jupyter notebook: jupyter notebook fraud_detection_backend.ipynb

Run Frontend (Web Application)

From the project root folder, run: streamlit run frontend/App.py

Usage Flow

Upload the dataset (creditcard.csv)

Preprocess data and generate features

Train models (Isolation Forest, XGBoost, etc.)

Evaluate performance (ROC-AUC, Recall, F1-score)

Monitor data drift and export reports