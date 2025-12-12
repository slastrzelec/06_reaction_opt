class FeatureExtractor:
    def __init__(self, scaler_path):
        self.scaler = load(scaler_path)
    
    def extract_from_smiles(self, smiles):
        """Oblicza ~50 deskryptorów RDKit"""
        return descriptors_dict
    
    def prepare_for_prediction(self, aryl_smiles, amine_smiles):
        """Łączy cechy substratu + generuje feature vector"""
        return scaled_features_array