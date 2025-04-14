import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem.rdmolops import MorganGenerator
from io import StringIO

# ---------------- Setup Morgan Generator ----------------
generator = MorganGenerator(radius=3, fpSize=2048)

def get_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# ---------------- Load Model ----------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# ---------------- Predict Function ----------------
def predict_ic50(smiles_list, model):
    results = []
    for smi in smiles_list:
        fp = get_morgan_fingerprint(smi)
        if fp is None:
            results.append((smi, "Invalid SMILES"))
        else:
            df = pd.DataFrame([fp], columns=[f"bit_{i}" for i in range(len(fp))])
            pred = model.predict(df)[0]
            results.append((smi, pred))
    return pd.DataFrame(results, columns=["SMILES", "Predicted IC50"])

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="IC50 Predictor", layout="centered")
st.title("💊 IC50 Predictor using Random Forest & Morgan Fingerprints")
st.markdown("Upload a trained `.pkl` model and provide SMILES to predict IC₅₀ values.")

# Upload model
model_file = st.file_uploader("📁 Upload Random Forest Model (.pkl)", type=["pkl"])
model = None

if model_file:
    model = load_model(model_file)
    st.success("✅ Model loaded successfully.")

    # Input method
    st.subheader("🧪 Enter SMILES")
    input_mode = st.radio("Choose input method:", ("Paste Text", "Upload File"))

    smiles_list = []

    if input_mode == "Paste Text":
        text_input = st.text_area("Enter one SMILES per line", height=200)
        if text_input:
            smiles_list = [s.strip() for s in text_input.strip().splitlines() if s.strip()]
    else:
        file_input = st.file_uploader("Upload a .csv or .txt file with SMILES", type=["csv", "txt"])
        if file_input:
            content = file_input.read().decode("utf-8")
            if file_input.name.endswith(".csv"):
                df = pd.read_csv(StringIO(content))
                smiles_list = df.iloc[:, 0].dropna().astype(str).tolist()
            else:
                smiles_list = [line.strip() for line in content.splitlines() if line.strip()]

    # Predict
    if smiles_list and st.button("🔍 Predict IC50"):
        results_df = predict_ic50(smiles_list, model)
        st.subheader("📊 Prediction Results")
        st.dataframe(results_df)

        # Download
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "predicted_ic50.csv", "text/csv")

elif model_file is None:
    st.info("Please upload a model file to begin.")
