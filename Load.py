import streamlit as st
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# --- Function to generate PubChem fingerprints from SMILES ---
def generate_pubchem_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetPubChemFingerprint(mol)
            # Convert the fingerprint to a NumPy array
            return np.array(list(fp))
        else:
            return None
    except Exception as e:
        st.error(f"Error processing SMILES: {e}")
        return None

# --- Load the trained model ---
try:
    with open('RFC_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --- Streamlit App Layout ---
st.title('Compound Activity (pIC50) Prediction')
st.write('Enter the SMILES annotation of a compound to predict its pIC50 activity.')

smiles_input = st.text_input('SMILES String:')

if smiles_input:
    # Generate PubChem fingerprint
    fingerprint = generate_pubchem_fingerprint(smiles_input)

    if fingerprint is not None:
        # Reshape the fingerprint to match the model's expected input shape (usually 2D)
        fingerprint = fingerprint.reshape(1, -1)

        # Make the prediction
        try:
            prediction = model.predict(fingerprint)[0]
            st.subheader('Prediction:')
            st.write(f'The predicted pIC50 value is: **{prediction:.2f}**')
        except Exception as e:
            st.error(f"Error during prediction: {e}")
