import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from io import StringIO

# ---------------- Load Model ----------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# ---------------- Fingerprint Function (Morgan) ----------------
def get_morgan_fingerprint(smiles, radius=3, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# ---------------- Prediction Logic ----------------
def predict_ic50(smiles_list, model):
    results = []
    for smi in smiles_list:
        fp = get_morgan_fingerprint(smi)
        if fp is None:
            results.append((smi, "Invalid SMILES"))
        else:
            pred = model.predict([fp])[0]
            results.append((smi, pred))
    return pd.DataFrame(results, columns=["SMILES", "Predicted IC50"])

# ---------------- Streamlit UI ----------------
st.title("IC50 Predictor with Morgan Fingerprints & Random Forest")

# Upload model file
model_file = st.file_uploader("Upload Trained Random Forest Model (.pkl)", type=["pkl"])
model = None

if model_file:
    model = load_model(model_file)
    st.success("Model loaded successfully.")

    # SMILES input options
    st.subheader("Enter SMILES")
    input_mode = st.radio("Choose input method:", ("Text Input", "File Upload"))

    smiles_list = []

    if input_mode == "Text Input":
        text_input = st.text_area("Paste SMILES here (one per line)", height=200)
        if text_input:
            smiles_list = [s.strip() for s in text_input.strip().splitlines() if s.strip()]
    else:
        file_input = st.file_uploader("Upload a CSV or TXT file with SMILES", type=["csv", "txt"])
        if file_input:
            content = file_input.read().decode("utf-8")
            if file_input.name.endswith(".csv"):
                df = pd.read_csv(StringIO(content))
                smiles_list = df.iloc[:, 0].dropna().astype(str).tolist()
            else:
                smiles_list = [line.strip() for line in content.splitlines() if line.strip()]

    if smiles_list and st.button("Predict IC50"):
        results_df = predict_ic50(smiles_list, model)
        st.dataframe(results_df)

        # Download predictions
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "predicted_ic50.csv", "text/csv")

elif model_file is None:
    st.info("Please upload a Random Forest model to continue.")
