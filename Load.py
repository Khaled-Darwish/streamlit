import pickle
import streamlit as st
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor

# Function to calculate PubChem fingerprints
def calculate_maccs_fingerprint(smiles):
    """Calculates the MACCS fingerprint for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)
    return None

# Load the Random Forest model from the pkl file
@st.cache_resource
def load_model(model_path):
    """Loads the Random Forest model from the specified pickle file."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_ic50(smiles, model):
    """Predicts the IC50 value for a given SMILES string using the loaded model."""
    fingerprint = calculate_maccs_fingerprint(smiles)
    if fingerprint is not None:
        # Reshape the fingerprint to be 2D for prediction
        fingerprint = fingerprint.reshape(1, -1)
        predicted_ic50 = model.predict(fingerprint)[0]
        return predicted_ic50
    else:
        return None

def main():
    st.title("IC50 Prediction using Random Forest Model")
    st.write("Enter the SMILES string of the molecule to predict its IC50 value.")

    # File uploader for the Random Forest model
    model_file = st.file_uploader("Upload your trained Random Forest model (.pkl file)", type=["pkl"])

    smiles_input = st.text_input("Enter SMILES string:")

    if model_file is not None:
        # Save the uploaded model temporarily
        with open("rf_model.pkl", "wb") as f:
            f.write(model_file.read())
        loaded_model = load_model("rf_model.pkl")

        if st.button("Predict IC50"):
            if smiles_input:
                predicted_value = predict_ic50(smiles_input, loaded_model)
                if predicted_value is not None:
                    st.success(f"Predicted IC50: {predicted_value:.4f}")
                else:
                    st.error("Invalid SMILES string. Please check the input.")
            else:
                st.warning("Please enter a SMILES string.")
    else:
        st.info("Please upload your trained Random Forest model file (.pkl) to make predictions.")

if __name__ == "__main__":
    main()