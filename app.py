import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from rdkit import Chem
from rdkit.Chem import Draw
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Buchwald-Hartwig Optimizer",
    page_icon="🧪",
    layout="wide"
)

@st.cache_resource
def load_model():
    data_dir = r"C:\Users\slast\PYTHON\0_projekty do portfolio\06_reaction_opt\data"
    models_dir = os.path.join(data_dir, 'trained_models')
    
    with open(os.path.join(models_dir, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    return model

def validate_smiles(smiles):
    """Validate SMILES string"""
    if not smiles or smiles.strip() == '':
        return False
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol is not None and mol.GetNumAtoms() > 0

st.title("🧪 Buchwald-Hartwig C-N Coupling Optimizer")
st.markdown("Optimize reaction conditions for C-N cross-coupling.")

try:
    model = load_model()
    st.sidebar.success("✓ Model loaded")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.header("📝 Input Substrates")

example_aryl = "ClC1=CC=C(C=C1)Br"
example_amine = "Nc1ccccc1"

aryl_smiles = st.sidebar.text_input("Aryl Halide SMILES", value=example_aryl)
amine_smiles = st.sidebar.text_input("Nucleophile (Amine) SMILES", value=example_amine)
predict_button = st.sidebar.button("🔮 Predict", use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    if aryl_smiles.strip() and validate_smiles(aryl_smiles):
        try:
            mol = Chem.MolFromSmiles(aryl_smiles)
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, caption="Aryl Halide")
        except:
            st.warning("Cannot draw")

with col2:
    if amine_smiles.strip() and validate_smiles(amine_smiles):
        try:
            mol = Chem.MolFromSmiles(amine_smiles)
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, caption="Nucleophile")
        except:
            st.warning("Cannot draw")

if predict_button:
    if not validate_smiles(aryl_smiles) or not validate_smiles(amine_smiles):
        st.error("❌ Invalid SMILES")
    else:
        with st.spinner("Predicting..."):
            bases = ['P2Et', 'BTMG', 'MTBD']
            ligands = ['XPhos', 't-BuXPhos', 't-BuBrettPhos', 'AdBrettPhos']
            additives = ['No_Additive', '3,5-dimethylisoxazole', 'Cs2CO3', 'K2CO3', 
                        'K3PO4', 'LiCl', 'Na2CO3']
            
            results = []
            
            for base in bases:
                for ligand in ligands:
                    for additive in additives:
                        # Build feature vector - ONLY categorical features
                        features = {}
                        
                        for b in bases:
                            features[f'base_{b}'] = 1 if b == base else 0
                        
                        for l in ligands:
                            features[f'ligand_{l}'] = 1 if l == ligand else 0
                        
                        # All possible additives from model
                        all_additives = [
                            '3,5-dimethylisoxazole', '3-methyl-5-phenylisoxazole',
                            '3-methylisoxazole', '3-phenylisoxazole', '4-phenylisoxazole',
                            '5-(2,6-difluorophenyl)isoxazole', '5-Phenyl-1,2,4-oxadiazole',
                            '5-methyl-3-(1H-pyrrol-1-yl)isoxazole', '5-methylisoxazole',
                            '5-phenylisoxazole', 'N,N-dibenzylisoxazol-3-amine',
                            'N,N-dibenzylisoxazol-5-amine', 'No_Additive',
                            'benzo[c]isoxazole', 'benzo[d]isoxazole',
                            'ethyl-3-methoxyisoxazole-5-carboxylate',
                            'ethyl-3-methylisoxazole-5-carboxylate',
                            'ethyl-5-methylisoxazole-3-carboxylate',
                            'ethyl-5-methylisoxazole-4-carboxylate',
                            'ethyl-isoxazole-3-carboxylate',
                            'ethyl-isoxazole-4-carboxylate',
                            'methyl-5-(furan-2-yl)isoxazole-3-carboxylate',
                            'methyl-5-(thiophen-2-yl)isoxazole-3-carboxylate',
                            'methyl-isoxazole-5-carboxylate'
                        ]
                        
                        for add in all_additives:
                            features[f'additive_{add}'] = 1 if add == additive else 0
                        
                        # Create DataFrame with proper order
                        X_pred = pd.DataFrame([features])
                        
                        # Get feature names from model
                        feature_names = model.feature_names_in_
                        
                        # Reorder to match model
                        X_pred = X_pred[feature_names]
                        
                        # Predict
                        try:
                            yield_pred = model.predict(X_pred)[0]
                            yield_pred = np.clip(yield_pred, 0, 100)
                        except Exception as e:
                            st.error(f"Error: {e}")
                            st.stop()
                        
                        results.append({
                            'Base': base,
                            'Ligand': ligand,
                            'Additive': additive,
                            'Predicted Yield': yield_pred
                        })
            
            results_df = pd.DataFrame(results).sort_values('Predicted Yield', ascending=False)
            
            st.markdown("---")
            st.subheader("🎯 Top 10 Recommendations")
            
            top_10 = results_df.head(10).reset_index(drop=True)
            top_10['Rank'] = range(1, 11)
            top_10['Yield %'] = top_10['Predicted Yield'].apply(lambda x: f"{x:.1f}")
            
            st.dataframe(
                top_10[['Rank', 'Base', 'Ligand', 'Additive', 'Yield %']],
                use_container_width=True,
                hide_index=True
            )

st.markdown("---")
st.markdown("<p style='text-align: center'>🧬 Buchwald-Hartwig Optimizer v1.0</p>", unsafe_allow_html=True)