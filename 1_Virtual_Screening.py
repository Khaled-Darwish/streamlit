import streamlit as st
import pandas as pd
import subprocess
import os
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from io import BytesIO
import base64

st.header("Virtual Screening")

# Model
def the_model(input_data):
    load_model = joblib.load('rf_model.pkl')
    # Make prediction
    prediction = load_model.predict(input_data)
    prediction_probability=load_model.predict_proba(input_data)
    
    x=pd.DataFrame(prediction_probability,columns=["Pi","Pa"])
    st.header('Prediction Result')
    
    prediction_output = pd.Series(prediction, name='Result')

    #proba_output=pd.Series(prediction_probability,name="prediction_proba")
    
    molecule_name = pd.Series(reading_data["Molecule Name"], name='Molecule Name')
    
    Result= pd.concat([molecule_name, x, prediction_output], axis=1)
    
    result = []
    for x in Result["Result"]:
        if x==1:
            result.append("Active")
        if x==0:
            result.append("Inactive")
    Result["Result"]=result
    st.write(Result)
    prediction_csv = Result.to_csv(index=False,sep=",")
    st.download_button(label="Download prediction results",data=prediction_csv,file_name="vs_results.csv")



uplouded_file=st.file_uploader("Please upload your input file", type=['txt'])


if st.button('Predict'):
    reading_data = pd.read_table(uplouded_file, sep=' ', names=["Smiles","Molecule Name"])
    reading_data.to_csv('molecule.smi', sep = '\t', index = False, header=None)
    st.subheader('Input data')
    st.markdown(reading_data.to_html(escape=False, index=False), unsafe_allow_html=True)
   ## st.write(reading_data)

  
    first_column = reading_data.iloc[:, 0] 
    Mol_descriptors,desc_names =descriptors(first_column)
    df_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)
    df=df_with_200_descriptors[["MolWt","MolLogP","NumHAcceptors","NumHDonors"]]
    st.subheader("Lipinski Rule of 5 Descriptors")
    st.write(df)
    
    
    the_model(pubfp_calc)
else:
    st.warning('Limit 250 compounds per file')
    
    
