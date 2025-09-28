# EEG Epilepsy Dashboard

This project includes an interactive Streamlit dashboard developed to predict epileptic seizures using EEG signals. XGBoost was used as the model.

---

## Project Files

- `brain.py` : Streamlit dashboard script
- `best_xgb_model.pkl` : Trained XGBoost model
- `Epileptic Seizure Recognition.csv` : Sample EEG dataset
- `README.md` : Project descriptions
- `images/` : Folder for dashboard and images

---

## Dashboard Preview
<img width="583" height="500" alt="newplot" src="https://github.com/user-attachments/assets/f382d54e-2de5-4a14-a35f-abe7d76cba13" />



---

## Features

1. EEG Signal Graph
   - Interactive Plotly graph
   - Channel selection and Moving Average filter
   - Color highlighting of seizure regions

2. Prediction Result
   - Class prediction from EEG signal using XGBoost model
   - Colorful and visualized output

3. Feature Importance
   - Shows which features (EEG channels) the model uses more
   - Easy to understand with a horizontal bar graph

4. SHAP Values
   - Shows how effective each feature is in the model's prediction

---

## Installation and Execution

1. Clone the repo:

```bash
git clone https://github.com/kullanici_adi/eeg-epilepsy-dashboard.git
cd eeg-epilepsy-dashboard

Install the required libraries:
pip install streamlit pandas numpy matplotlib plotly shap xgboost
Launch the Streamlit dashboard:
streamlit run brain.py
