# 💳 Credit Card Fraud Detection App

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green?style=for-the-badge)

## 🌐 Live Application
Experience the app in real-time here:  
**[Credit Card Fraud Detection Live App](https://creditcardfrauddetectionapp-f5capphozzzzs6sxhvue5dk.streamlit.app/)**

---

## 📝 Project Overview
This project is an end-to-end **Machine Learning solution** designed to identify fraudulent credit card transactions. By leveraging advanced data analysis and predictive modeling, the system can distinguish between legitimate customer behavior and criminal activity with high accuracy.

The application is built using **Streamlit**, providing an interactive dashboard where users can manually input transaction details to receive an instant risk assessment.

### 🧠 Core Logic & Features
The detection engine is powered by an **XGBoost Classifier**. It doesn't just look at the transaction amount; it evaluates a complex set of features:
* **Temporal Analysis:** Automatically extracts the hour, day, and month from transaction timestamps to detect unusual spending times.
* **Geospatial Intelligence:** Calculates the distance between the cardholder's home coordinates and the merchant's location to identify suspicious geographical gaps.
* **Demographic & Behavioral Patterns:** Processes categorical data like merchant types, job categories, and spending categories using pre-trained Label Encoders.

---

## 📊 Dataset Reference
The model was trained and validated using the **Credit Card Fraud Detection** dataset from Kaggle. This dataset contains simulated transactions that provide a realistic representation of fraud patterns.

* **Original Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

---

## 🛠️ Technical Stack
* **Modeling:** XGBoost (Extreme Gradient Boosting)
* **Data Processing:** Pandas, Scikit-learn (StandardScaler, LabelEncoder)
* **Deployment:** Streamlit Cloud
* **Serialization:** Joblib (for saving/loading the model and preprocessors)

---

## 📂 Project Structure
* `app.py`: The main application script for the Streamlit interface.
* `Credit_card_fraud_detection.ipynb`: The research notebook containing data cleaning, EDA, and model training.
* `model.pkl`: The optimized XGBoost model.
* `scaler.pkl` & `encoders.pkl`: Pre-trained objects used to normalize and encode input data to match the training format.

---
> **Note:** This project is developed for educational purposes to demonstrate the application of Machine Learning in financial security.
