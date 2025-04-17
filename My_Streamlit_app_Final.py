import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO

# 🎯 Page configuration
st.set_page_config(page_title="Activity Predictor", layout="centered")

st.title("🔬 Activity Predictor using KNN Model NOW")
st.markdown("Upload your trained KNN model and a CSV file of compounds with PubChem fingerprints.")

# 📤 Upload model
model_file = st.file_uploader("📦 Upload your trained KNN model (.pkl)", type=['pkl'])

# 📤 Upload fingerprint CSV
csv_file = st.file_uploader("📄 Upload your CSV file with PubChem fingerprints", type=['csv'])

# 🧠 Predict button
if model_file and csv_file and st.button("🚀 Predict Activity; 1 for Active, 0 for Inactive"):
    try:
        # Load the model
        knn_model = pickle.load(model_file)

        # Load the CSV file
        df = pd.read_csv(csv_file)

        st.success("Files loaded successfully!")

        # Automatically remove non-numeric or non-fingerprint columns
        non_feature_cols = ['id', 'compound_id', 'compound', 'name', 'smiles', 'SMILES']
        feature_cols = [col for col in df.columns if col.lower() not in [c.lower() for c in non_feature_cols]]
        
        X = df[feature_cols]

        # Get compound IDs if available
        id_col = None
        for col in df.columns:
            if col.lower() in ['compound_id', 'id', 'compound', 'name']:
                id_col = col
                break
        
        compound_ids = df[id_col] if id_col else pd.Series([f"Compound_{i}" for i in range(len(df))])


        # Predict IC50
        predicted_ic50 = knn_model.predict(X)

        # Output DataFrame
        output_df = pd.DataFrame({
            'Compound_ID': compound_ids,
            'Predicted_Activity': predicted_ic50
        })

        st.dataframe(output_df)

        # Convert to CSV for download
        csv_output = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", data=csv_output, file_name="predicted_Activity.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
