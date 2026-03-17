# 💳 Credit Card Fraud Detection App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)

## 📌 Project Concept
The **Credit Card Fraud Detection App** is a robust machine learning system designed to identify fraudulent transactions in real-time. In an era of digital payments, protecting customers from unauthorized activity is paramount. 

This project handles the entire data science lifecycle: from processing complex features like merchant location and user profession to solving the challenge of **imbalanced data** using **SMOTE**. The final result is a high-performance **XGBoost** model served through an intuitive **Streamlit** dashboard.

## 📊 Dataset Information
The model was trained using the **Credit Card Fraud Detection** dataset from Kaggle, which includes simulated credit card transactions covering a wide range of scenarios.

* **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
* **Key Features:** Merchant details, category of spend, transaction amount, geographic coordinates (Lat/Long), and user job profiles.

## ✨ Key Features
* **Instant Prediction:** Input transaction details manually to get an immediate "Fraud" or "Legitimate" classification.
* **Probability Scores:** View the confidence level of the model for each prediction.
* **Smart Preprocessing:** Automatically extracts time-based features (Hour, Day, Month) and calculates distances between the user and the merchant.
* **Handling Unseen Data:** The application includes logic to handle categorical values not seen during the training phase.

## 🛠️ Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/Credit_card_fraud_detection_App.git](https://github.com/YOUR_USERNAME/Credit_card_fraud_detection_App.git)
cd Credit_card_fraud_detection_App
