# Credit_card_fraud_detection_App 💳

A machine learning-powered web application designed to detect fraudulent credit card transactions in real-time. This project utilizes an **XGBoost Classifier** model to analyze transaction patterns and identify potential risks based on historical data. 

## 🌐 Live Demo
You can access the live application here:  
**[Credit Card Fraud Detection App](https://creditcardfrauddetectionapp-f5capphozzzzs6sxhvue5dk.streamlit.app/)**

## 📝 Project Overview
The goal of this project is to build a robust system capable of distinguishing between legitimate and fraudulent credit card activities. The model was trained on a large dataset containing various transaction attributes like amount, category, merchant details, and geographic location.

### Key Features:
* **Real-time Prediction:** Input transaction details manually to get an instant fraud probability score.
* **Automated Preprocessing:** The app automatically handles feature engineering (extracting hour, day, and month from timestamps) and geographic distance calculations.
* **Robust Encoding:** Uses saved `LabelEncoders` and `StandardScaler` to ensure consistency between training and production data.
* **High Performance:** Powered by an optimized XGBoost model for high accuracy and recall.

## 📊 Dataset
The dataset used for training and testing is the **Credit Card Fraud Detection** dataset available on Kaggle. It contains simulated credit card transactions including both legitimate and fraudulent samples.
* **Original Data Link:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** Streamlit (Web App Interface)
* **Libraries:** Scikit-learn, XGBoost, Pandas, Numpy, Joblib
* **Model:** XGBoost Classifier

## 🚀 How to Run Locally
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Credit_card_fraud_detection_App.git](https://github.com/YourUsername/Credit_card_fraud_detection_App.git)
    cd Credit_card_fraud_detection_App
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch the app:**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure
* `app.py`: The main Streamlit application script.
* `Credit_card_fraud_detection.ipynb`: Jupyter notebook containing the full data analysis, preprocessing, and model training workflow.
* `model.pkl`: The trained XGBoost model.
* `scaler.pkl`: Saved StandardScaler object for feature normalization.
* `encoders.pkl`: Dictionary containing LabelEncoders for categorical features.
* `requirements.txt`: List of required libraries for deployment.

---
*Created for Financial Security and Fraud Awareness.*
