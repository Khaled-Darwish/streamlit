import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
import tempfile

# Streamlit app title
st.title("Molecular Activity Prediction App")

st.write("""
Upload your trained **KNN model (.pkl)** and a **fingerprint CSV file** to predict activities.
""")

# Upload model file
model_file = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])

# Upload CSV file
csv_file = st.file_uploader("Upload your fingerprint CSV file", type=["csv"])

if model_file and csv_file:
    try:
        # Load model
        with tempfile.NamedTemporaryFile(delete=False) as tmp_model:
            tmp_model.write(model_file.read())
            knn_model = joblib.load(tmp_model.name)

        # Confirm model type
        if not hasattr(knn_model, 'predict'):
            st.error("The uploaded model is not a valid scikit-learn model with a predict() method.")
            st.stop()

        # Load CSV
        df = pd.read_csv(csv_file)

        st.success("Files uploaded and loaded successfully!")

        # Show first few rows
        st.subheader("First few rows of the input CSV")
        st.dataframe(df.head())

        # Extract features
        X = df.iloc[:, 1:882]  # Adjust if needed

        # Predict
        st.subheader("Making Predictions...")
        predictions = knn_model.predict(X)
        probabilities = knn_model.predict_proba(X)

        # Add predictions
        df['Predicted_Activity'] = predictions
        df['Probability_Class_0'] = probabilities[:, 0]
        df['Probability_Class_1'] = probabilities[:, 1]

        # Display updated DataFrame
        st.subheader("Prediction Results")
        st.dataframe(df)

        # Download option
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_download = convert_df(df)

        st.download_button(
            label="Download Predictions as CSV",
            data=csv_download,
            file_name='predictions_output.csv',
            mime='text/csv',
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload both a model file and a CSV file.")
