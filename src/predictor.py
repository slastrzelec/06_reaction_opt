class ReactionOptimizer:
    def __init__(self, model_path):
        self.model = load(model_path)
        self.extractor = FeatureExtractor()
        self.bases = ['P2Et', 'BTMG', 'MTBD']
        self.ligands = ['XPhos', 't-BuXPhos', 't-BuBrettPhos', 'AdBrettPhos']
        self.additives = [list_of_additives]
    
    def predict_all_combinations(self, aryl_smiles, amine_smiles):
        """
        Predicts yield for ALL combinations and returns ranked list
        """
        results = []
        for base in self.bases:
            for ligand in self.ligands:
                for additive in self.additives:
                    features = self.extract_features(aryl_smiles, amine_smiles, 
                                                     base, ligand, additive)
                    yield_pred = self.model.predict(features)
                    results.append({
                        'base': base,
                        'ligand': ligand,
                        'additive': additive,
                        'predicted_yield': yield_pred
                    })
        
        return sorted(results, key=lambda x: x['predicted_yield'], reverse=True)