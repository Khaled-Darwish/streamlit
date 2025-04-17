import streamlit as st
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from io import BytesIO

st.set_page_config(page_title="KNN SMILES Predictor on PTP1B Target", layout="centered")

st.title("🧪 KNN SMILES Activity Predictor")
st.write("Upload a trained **KNN model (.pkl)** and a **CSV file** with SMILES and 881-bit PubChem fingerprints to predict activity (0 or 1).")

# Upload model
model_file = st.file_uploader("🔍 Upload your trained KNN model (.pkl)", type=["pkl"])

# Upload CSV
csv_file = st.file_uploader("📄 Upload your fingerprint CSV file (.csv)", type=["csv"])

if model_file and csv_file:
    try:
        # Load model
        knn_model = pickle.load(model_file)

        # Load CSV data
        df = pd.read_csv(csv_file)
        st.subheader("📊 Preview of Uploaded Data")
        st.dataframe(df.head())

        # Check that there's at least 882 columns (SMILES + 881 bits)
        if df.shape[1] < 882:
            st.error("The CSV should contain a 'SMILES' column followed by 881 fingerprint bit columns.")
        else:
            # Extract SMILES and fingerprint data
            smiles = df.iloc[:, 0]
            X = df.iloc[:, 1:882]

            # Make predictions
            preds = knn_model.predict(X)
            probs = knn_model.predict_proba(X)

            # Add predictions to DataFrame
            df['Predicted_Activity'] = preds
            df['Probability_Class_0'] = probs[:, 0]
            df['Probability_Class_1'] = probs[:, 1]

            st.subheader("✅ Prediction Results")
            st.dataframe(df[['SMILES', 'Predicted_Activity', 'Probability_Class_0', 'Probability_Class_1']].head())

            # Download button
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(df)
            st.download_button(
                label="📥 Download Prediction Results",
                data=csv_data,
                file_name='knn_predictions.csv',
                mime='text/csv'
            )
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
else:
    st.info("Please upload both a KNN `.pkl` model and a fingerprint `.csv` file to continue.")
