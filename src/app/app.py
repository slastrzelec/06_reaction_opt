import streamlit as st
from src.predictor import ReactionOptimizer
from rdkit import Chem
from rdkit.Chem import Draw

st.set_page_config(page_title="Buchwald-Hartwig Optimizer", layout="wide")

optimizer = ReactionOptimizer(model_path='data/trained_models/model.pkl')

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input Substrates")
    aryl_smiles = st.text_input("Aryl Halide SMILES")
    amine_smiles = st.text_input("Nucleophile (Amine) SMILES")
    
    if st.button("Predict Optimal Conditions"):
        # Walidacja
        if validate_smiles(aryl_smiles) and validate_smiles(amine_smiles):
            # Predykcja
            results = optimizer.predict_all_combinations(aryl_smiles, amine_smiles)
            
            # Wizualizacja
            st.subheader("Top 5 Recommendations")
            st.dataframe(results[:5])
            
            st.subheader("Yield Comparison")
            st.bar_chart(data=results)
        else:
            st.error("Invalid SMILES")