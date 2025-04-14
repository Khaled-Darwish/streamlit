import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem  # <-- This is the critical fix
from sklearn.ensemble import RandomForestRegressor
import pickle

# App title and description
st.title('Compound Activity Predictor (IC50)')
st.write("""
Predict IC50 values from SMILES strings using a Random Forest model trained on PubChem fingerprints.
""")

# Sidebar
st.sidebar.header('About')
st.sidebar.info("""
- **Input**: Valid SMILES notation (e.g., `CCO` for ethanol)
- **Model**: Random Forest (PubChem fingerprints)
- **Output**: Predicted IC50 (nM)
""")

# Function to validate SMILES and compute fingerprints
def smiles_to_pubchem_fp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("❌ Invalid SMILES: Could not parse molecule.")
            return None
        
        # Add Hydrogens (optional but recommended)
        mol = Chem.AddHs(mol)  
        
        # Generate hashed Morgan fingerprint (radius=2, nBits=881)
        fp = AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=881)  # <-- Now works
        return np.array(fp)
    except Exception as e:
        st.error(f"❌ Error generating fingerprint: {str(e)}")
        return None

# Load model (replace with your trained model)
def load_model():
    try:
        # Example: Load a pre-trained model
        # model = pickle.load(open("rf_pubchem_model.pkl", "rb"))
        
        # For demo: Train a dummy model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X_dummy = np.random.rand(100, 881)  # 100 random samples
        y_dummy = np.random.rand(100) * 10   # Random IC50 (0-10 nM)
        model.fit(X_dummy, y_dummy)
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

# Cache the model to avoid reloading
if "model" not in st.session_state:
    st.session_state.model = load_model()

# Input SMILES
smiles_input = st.text_input(
    "Enter SMILES notation:",
    placeholder="e.g., CCO (ethanol), CC(=O)O (acetic acid)",
    value="CCO"  # Default example
)

if st.button("Predict IC50"):
    if not smiles_input.strip():
        st.warning("⚠️ Please enter a SMILES string.")
    else:
        # Check SMILES validity first
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error("❌ Invalid SMILES format. Example valid SMILES: `CCO` (ethanol)")
        else:
            # Generate fingerprint
            fp = smiles_to_pubchem_fp(smiles_input)
            if fp is not None:
                try:
                    # Predict IC50
                    model = st.session_state.model
                    ic50_pred = model.predict(fp.reshape(1, -1))[0]
                    st.success(f"✅ Predicted IC50: **{ic50_pred:.2f} nM**")
                    
                    # Display molecule
                    st.subheader("Molecule Structure")
                    img = MolToImage(mol, size=(400, 400))
                    st.image(img, caption=f"SMILES: {smiles_input}")
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

# Example SMILES
st.subheader("Example SMILES")
examples = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "Glucose": "C(C1C(C(C(C(O1)O)O)O)O",
}

for name, smiles in examples.items():
    st.write(f"- **{name}**: `{smiles}`")
