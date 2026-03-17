import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# ==========================================
# 1. App Configuration
# ==========================================
st.set_page_config(page_title="Fraud Detection System 💳", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.markdown("---")

# ==========================================
# 2. Loading & Processing Functions
# ==========================================
@st.cache_resource
def load_resources():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')  # Load label encoders
    return model, scaler, encoders

try:
    model, scaler, encoders = load_resources()
except FileNotFoundError as e:
    st.error(f"⚠️ Please make sure the model files exist in the same directory: {e.filename}")
    st.stop()

# Safe encoding function to handle unseen labels (returns -1 for unknown values)
def safe_encode(le, series):
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    return series.apply(lambda x: le_dict.get(x, -1))

# Preprocessing function to match the model's training format
def preprocess_data(df):
    process_df = df.copy()

    # Extract time/date features if the column exists
    if "trans_date_trans_time" in process_df.columns:
        process_df["trans_date_trans_time"] = pd.to_datetime(process_df["trans_date_trans_time"])
        process_df['trans_hour'] = process_df['trans_date_trans_time'].dt.hour
        process_df['trans_day'] = process_df['trans_date_trans_time'].dt.dayofweek
        process_df['trans_month'] = process_df['trans_date_trans_time'].dt.month

    # Calculate distances if coordinate columns exist
    if all(col in process_df.columns for col in ['lat', 'merch_lat', 'long', 'merch_long']):
        process_df['dist_lat'] = process_df['lat'] - process_df['merch_lat']
        process_df['dist_long'] = process_df['long'] - process_df['merch_long']
        process_df['dist_total'] = np.sqrt(process_df['dist_lat']**2 + process_df['dist_long']**2)

    # Drop unnecessary columns
    cols_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state',
                    'zip', 'dob', 'trans_num', 'trans_date_trans_time', 'is_fraud']
    process_df.drop(columns=[col for col in cols_to_drop if col in process_df.columns],
                    inplace=True, errors='ignore')

    # Encode categorical columns using label encoders
    for col in ['merchant', 'category', 'gender', 'job']:
        if col in process_df.columns:
            process_df[col] = safe_encode(encoders[col], process_df[col])

    # Align columns to match scaler's expected feature order
    if hasattr(scaler, 'feature_names_in_'):
        # Fill any missing columns with 0 (important for manual input)
        for col in scaler.feature_names_in_:
            if col not in process_df.columns:
                process_df[col] = 0
        process_df = process_df[scaler.feature_names_in_]

    # Scale the features
    X_scaled = scaler.transform(process_df)
    return X_scaled

# ==========================================
# 3. User Interface (Tabs: File Upload | Manual Entry)
# ==========================================
tab1, tab2 = st.tabs(["📂 Upload Data File (CSV)", "✍️ Manual Transaction Entry"])

# ----------------- Tab 1: File Upload -----------------
with tab1:
    st.write("Upload a `fraudTest.csv` file or any file with the same columns to test it in bulk.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="file_upload")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"### 👁️ Quick Data Preview (Total rows: {len(df):,})")
        st.dataframe(df.head())

        if st.button("🔍 Scan Transactions in File", type="primary"):
            with st.spinner('Analyzing data... Please wait (may take a few seconds for large files)'):
                try:
                    # Preprocess and predict
                    X_scaled = preprocess_data(df)
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)[:, 1]

                    # Save results
                    df['Prediction'] = ["Fraud 🔴" if p == 1 else "Legitimate 🟢" for p in predictions]
                    df['Fraud_Probability (%)'] = (probabilities * 100).round(2)

                    st.success("✅ Scan completed successfully!")

                    # --- Quick Statistics ---
                    total_fraud = sum(predictions)
                    st.warning(f"🚨 Detected {total_fraud:,} fraudulent transactions out of {len(df):,} total transactions.")

                    # --- Color-coded results preview (first 1000 rows to avoid browser freeze) ---
                    def color_fraud(val):
                        if val == "Fraud 🔴":
                            return 'background-color: rgba(255, 75, 75, 0.2); color: #ff3333; font-weight: bold;'
                        else:
                            return 'background-color: rgba(75, 255, 75, 0.2); color: #00cc00; font-weight: bold;'

                    display_cols = ['amt', 'merchant', 'category', 'Prediction', 'Fraud_Probability (%)']
                    display_cols = [c for c in display_cols if c in df.columns]

                    st.write("### 📊 Results Preview (First 1,000 transactions):")
                    preview_df = df[display_cols].head(1000)
                    st.dataframe(preview_df.style.map(color_fraud, subset=['Prediction']))

                    # --- Download full results ---
                    st.write("### 📥 Download Full Report:")
                    csv_data = df[display_cols].to_csv(index=False).encode('utf-8-sig')

                    st.download_button(
                        label="Download Full Results File (CSV)",
                        data=csv_data,
                        file_name='fraud_detection_full_results.csv',
                        mime='text/csv',
                    )

                except Exception as e:
                    st.error(f"❌ An error occurred during processing: {e}")

# ----------------- Tab 2: Manual Entry -----------------
with tab2:
    st.write("Enter transaction details manually to test them instantly.")

    with st.form("manual_entry_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            amt = st.number_input("Transaction Amount (amt)", min_value=0.0, value=150.0)
            gender = st.selectbox("Gender", options=encoders['gender'].classes_)
            category = st.selectbox("Purchase Category (category)", options=encoders['category'].classes_)
            merchant = st.text_input("Merchant Name (merchant)", value="fraud_Rippin, Kub and Mann")

        with col2:
            lat = st.number_input("Customer Latitude (lat)", value=33.9659)
            long = st.number_input("Customer Longitude (long)", value=-80.9355)
            merch_lat = st.number_input("Merchant Latitude (merch_lat)", value=33.9863)
            merch_long = st.number_input("Merchant Longitude (merch_long)", value=-81.2007)

        with col3:
            city_pop = st.number_input("City Population (city_pop)", value=333497)
            job = st.selectbox("Occupation (job)", options=encoders['job'].classes_)
            unix_time = st.number_input("Unix Timestamp (unix_time)", value=1371816865)
            trans_date = st.date_input("Transaction Date", datetime.date.today())
            trans_time = st.time_input("Transaction Time", datetime.datetime.now().time())

        submit_btn = st.form_submit_button("🛡️ Check This Transaction")

        if submit_btn:
            # Combine date and time
            dt_combine = datetime.datetime.combine(trans_date, trans_time)

            # Build DataFrame from manual input
            manual_data = pd.DataFrame([{
                'trans_date_trans_time': dt_combine,
                'merchant': merchant,
                'category': category,
                'amt': amt,
                'gender': gender,
                'lat': lat,
                'long': long,
                'city_pop': city_pop,
                'job': job,
                'unix_time': unix_time,
                'merch_lat': merch_lat,
                'merch_long': merch_long
            }])

            try:
                X_scaled = preprocess_data(manual_data)
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0][1]

                st.markdown("---")
                if pred == 1:
                    st.error(f"🚨 Warning: This transaction is **FRAUDULENT**! (Confidence: {prob*100:.2f}%)")
                else:
                    st.success(f"✅ Transaction is **LEGITIMATE**. (Fraud probability: {prob*100:.2f}%)")
            except Exception as e:
                st.error(f"❌ An error occurred while checking the manual transaction: {e}")