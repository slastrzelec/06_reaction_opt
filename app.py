import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from rdkit import Chem
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Buchwald-Hartwig Optimizer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    [data-testid="stMetricLabel"] {
        color: #333333;
    }
    [data-testid="stMetricValue"] {
        color: #1f77b4;
        font-size: 28px;
    }
    .main {
        padding-top: 2rem;
    }
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join("data", "trained_models", "best_model.pkl")
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.stop()
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def validate_smiles(smiles):
    """Validate SMILES string"""
    if not smiles or smiles.strip() == '':
        return False
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol is not None and mol.GetNumAtoms() > 0

def generate_pdf_report(results_df, aryl_smiles, amine_smiles, top_n=10):
    """Generate PDF report with molecular structures"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        import io
        import requests
        from PIL import Image as PILImage
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30
        )
        story.append(Paragraph("Buchwald-Hartwig C-N Coupling Optimizer", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Input info
        story.append(Paragraph("<b>Input Substrates:</b>", styles['Heading2']))
        story.append(Paragraph(f"Aryl Halide SMILES: <font color='blue'>{aryl_smiles}</font>", styles['Normal']))
        story.append(Paragraph(f"Nucleophile SMILES: <font color='blue'>{amine_smiles}</font>", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Add molecular structures
        story.append(Paragraph("<b>Molecular Structures:</b>", styles['Heading2']))
        
        struct_table_data = []
        
        # Get images from PubChem
        try:
            aryl_img = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{aryl_smiles}/PNG", timeout=5)
            if aryl_img.status_code == 200:
                aryl_pil = PILImage.open(io.BytesIO(aryl_img.content))
                aryl_buf = io.BytesIO()
                aryl_pil.save(aryl_buf, format="PNG")
                aryl_buf.seek(0)
                aryl_img_obj = Image(aryl_buf, width=1.5*inch, height=1.5*inch)
            else:
                aryl_img_obj = Paragraph("Aryl Halide Structure", styles['Normal'])
        except:
            aryl_img_obj = Paragraph("Aryl Halide Structure", styles['Normal'])
        
        try:
            amine_img = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{amine_smiles}/PNG", timeout=5)
            if amine_img.status_code == 200:
                amine_pil = PILImage.open(io.BytesIO(amine_img.content))
                amine_buf = io.BytesIO()
                amine_pil.save(amine_buf, format="PNG")
                amine_buf.seek(0)
                amine_img_obj = Image(amine_buf, width=1.5*inch, height=1.5*inch)
            else:
                amine_img_obj = Paragraph("Nucleophile Structure", styles['Normal'])
        except:
            amine_img_obj = Paragraph("Nucleophile Structure", styles['Normal'])
        
        struct_table_data = [
            [Paragraph("<b>Aryl Halide</b>", styles['Normal']), Paragraph("<b>Nucleophile</b>", styles['Normal'])],
            [aryl_img_obj, amine_img_obj]
        ]
        
        struct_table = Table(struct_table_data, colWidths=[2.5*inch, 2.5*inch])
        struct_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6f2ff')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(struct_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Results table
        story.append(Paragraph("<b>Top 10 Recommended Conditions:</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        top_results = results_df.head(top_n).reset_index(drop=True)
        top_results['Rank'] = range(1, len(top_results) + 1)
        top_results['Yield %'] = top_results['Predicted Yield'].apply(lambda x: f"{x:.1f}%")
        
        table_data = [['Rank', 'Base', 'Ligand', 'Additive', 'Yield %']]
        for _, row in top_results.iterrows():
            table_data.append([
                str(int(row['Rank'])),
                row['Base'],
                row['Ligand'],
                row['Additive'][:30],
                row['Yield %']
            ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        story.append(Paragraph("<i>⚠️ This is a preliminary model for research purposes only. Always validate results experimentally.</i>", styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    except ImportError:
        st.warning("ReportLab not available. Install with: pip install reportlab")
        return None
    except Exception as e:
        st.warning(f"PDF generation error: {str(e)[:100]}")
        return None

st.title("🧪 Buchwald-Hartwig C-N Coupling Optimizer")
st.markdown("**Optimize reaction conditions for C-N cross-coupling.**")
st.markdown("---")

try:
    model = load_model()
    st.sidebar.success("✓ Model loaded")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

st.sidebar.header("📝 Input Substrates")

example_aryl = "ClC1=CC=C(C=C1)Br"
example_amine = "Nc1ccccc1"

aryl_smiles = st.sidebar.text_input("Aryl Halide SMILES", value=example_aryl)
amine_smiles = st.sidebar.text_input("Nucleophile (Amine) SMILES", value=example_amine)
predict_button = st.sidebar.button("🔮 Predict", use_container_width=True)

# Display structures using PubChem
col1, col2 = st.columns(2)

with col1:
    if aryl_smiles.strip() and validate_smiles(aryl_smiles):
        st.subheader("Aryl Halide")
        st.image(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{aryl_smiles}/PNG", width=300)

with col2:
    if amine_smiles.strip() and validate_smiles(amine_smiles):
        st.subheader("Nucleophile")
        st.image(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{amine_smiles}/PNG", width=300)

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
                        features = {}
                        
                        for b in bases:
                            features[f'base_{b}'] = 1 if b == base else 0
                        
                        for l in ligands:
                            features[f'ligand_{l}'] = 1 if l == ligand else 0
                        
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
                        
                        X_pred = pd.DataFrame([features])
                        feature_names = model.feature_names_in_
                        X_pred = X_pred[feature_names]
                        
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
            
            # Add to history
            st.session_state.history.append({
                'Aryl SMILES': aryl_smiles,
                'Amine SMILES': amine_smiles,
                'Top Result': results_df.iloc[0]['Base'],
                'Yield': f"{results_df.iloc[0]['Predicted Yield']:.1f}%",
                'Timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            })
            
            st.markdown("---")
            st.subheader("🎯 Top 10 Recommendations")
            st.markdown("**Predicted reaction yields sorted by performance:**")
            
            top_10 = results_df.head(10).reset_index(drop=True)
            top_10['Rank'] = range(1, 11)
            top_10['Yield %'] = top_10['Predicted Yield'].apply(lambda x: f"{x:.1f}")
            
            st.dataframe(
                top_10[['Rank', 'Base', 'Ligand', 'Additive', 'Yield %']],
                use_container_width=True,
                hide_index=True
            )
            
            # Detailed information
            with st.expander("📊 Detailed Statistics"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Yield", f"{results_df['Predicted Yield'].max():.1f}%")
                with col2:
                    st.metric("Average Yield", f"{results_df['Predicted Yield'].mean():.1f}%")
                with col3:
                    st.metric("Total Combinations", len(results_df))
                with col4:
                    st.metric("Std Dev", f"{results_df['Predicted Yield'].std():.1f}%")
                
                # Base performance
                st.markdown("**Base Performance:**")
                base_stats = results_df.groupby('Base')['Predicted Yield'].agg(['mean', 'max', 'min'])
                st.dataframe(base_stats.round(1), use_container_width=True)
                
                # Ligand performance
                st.markdown("**Ligand Performance:**")
                ligand_stats = results_df.groupby('Ligand')['Predicted Yield'].agg(['mean', 'max', 'min'])
                st.dataframe(ligand_stats.round(1), use_container_width=True)
            
            # Export options
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = top_10.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"buchwald_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                pdf_buffer = generate_pdf_report(results_df, aryl_smiles, amine_smiles)
                if pdf_buffer:
                    st.download_button(
                        label="📄 Download PDF",
                        data=pdf_buffer,
                        file_name=f"buchwald_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )

# History sidebar
st.sidebar.markdown("---")
st.sidebar.header("📜 History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.sidebar.dataframe(history_df, use_container_width=True, hide_index=True)
    
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.sidebar.info("No predictions yet")

st.markdown("---")
st.markdown("<p style='text-align: center'><small>🧬 Buchwald-Hartwig Optimizer v2.0 | Research Tool</small></p>", unsafe_allow_html=True)