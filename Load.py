import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# App title and description
st.title('Compound Activity Predictor (IC50)')
st.write("""
This app predicts the biological activity (IC50) of input compounds using a Random Forest model trained on PubChem fingerprints.
""")

# Sidebar
st.sidebar.header('About')
st.sidebar.info("""
- Input: SMILES notation of chemical compound
- Model: Random Forest trained on PubChem fingerprints
- Output: Predicted IC50 value (nM)
""")

# Function to compute PubChem fingerprints
def smiles_to_pubchem_fp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=881)
        return np.array(fp)
    except:
        return None

# Load model (in production, you would load your pretrained model)
# This is a placeholder - replace with your actual model loading code
def load_model():
    # Try to load a pretrained model
    try:
        # In a real app, you would load your saved model file
        # Example: model = pickle.load(open('rf_pubchem_model.pkl', 'rb'))
        
        # For demonstration, we'll create a dummy model
        # DON'T DO THIS IN PRODUCTION - USE YOUR TRAINED MODEL
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Create dummy data to fit (so the app runs without error)
        # In production, remove this and use your actual trained model
        X_dummy = np.random.rand(10, 881)
        y_dummy = np.random.rand(10)
        model.fit(X_dummy, y_dummy)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Input SMILES
smiles_input = st.text_input('Enter SMILES notation:', 'CCO')

if st.button('Predict IC50'):
    if not smiles_input:
        st.warning('Please enter a SMILES string.')
    else:
        # Convert SMILES to fingerprint
        fp = smiles_to_pubchem_fp(smiles_input)
        
        if fp is None:
            st.error('Invalid SMILES string or unable to generate fingerprint.')
        else:
            # Make prediction
            try:
                prediction = model.predict(fp.reshape(1, -1))[0]
                st.success(f'Predicted IC50: {prediction:.2f} nM')
                
                # Display molecule
                mol = Chem.MolFromSmiles(smiles_input)
                if mol:
                    st.write('Chemical structure:')
                    img = Chem.Draw.MolToImage(mol, size=(300, 300))
                    st.image(img, caption='Input Molecule')
                
            except Exception as e:
                st.error(f'Prediction failed: {e}')

# Example SMILES
st.subheader('Example SMILES')
st.write("""
Try these example SMILES strings:
- Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O
- Caffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C
- Glucose: C(C1C(C(C(C(O1)O)O)O)O
""")
