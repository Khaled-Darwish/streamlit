import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

# --------- Load Model -------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# --------- Generate PubChem Fingerprint -------------
def get_pubchem_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        fp = rdMolDescriptors.GetHashedPubChemFingerprint(mol)
        arr = np.zeros((1,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        return None

# --------- Predict Function -------------
def predict_ic50(smiles_list, model):
    fingerprints = []
    valid_smiles = []
    for smi in smiles_list:
        fp = get_pubchem_fingerprint(smi)
        if fp is not None:
            fingerprints.append(fp)
            valid_smiles.append(smi)
        else:
            fingerprints.append(None)
            valid_smiles.append(smi)

    predictions = []
    for smi, fp in zip(valid_smiles, fingerprints):
        if fp is None:
            predictions.append("Invalid SMILES")
        else:
            pred = model.predict([fp])[0]
            predictions.append(pred)

    return pd.DataFrame({"SMILES": valid_smiles, "Predicted IC50": predictions})

# --------- Streamlit UI -------------
st.title("IC50 Prediction using Random Forest Model and PubChem Fingerprints")
st.markdown("""
Upload a `.pkl` file of your trained Random Forest model, and input SMILES to predict IC₅₀ values.
""")

# Upload model
model_file = st.file_uploader("Upload your Random Forest Model (.pkl)", type=["pkl"])
if model_file:
    model = load_model(model_file)

    # SMILES input
    smiles_input = st.text_area("Enter SMILES (one per line)", height=200)

    if st.button("Predict IC50"):
        if smiles_input.strip() == "":
            st.warning("Please enter at least one SMILES string.")
        else:
            smiles_list = [s.strip() for s in smiles_input.strip().splitlines() if s.strip()]
            result_df = predict_ic50(smiles_list, model)
            st.dataframe(result_df)

            # Downloadable CSV
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions as CSV", csv, "predicted_ic50.csv", "text/csv")

