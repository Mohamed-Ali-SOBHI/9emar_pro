#!/usr/bin/env python3
"""
Ensemble predictions pour réduire les erreurs par prédictions multiples
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictorModel:
    """Prédicteur ensemble avec réduction d'erreurs par répétition"""
    
    def __init__(self, model_path="models/optimal_model"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.features = None
        self.load_model()
    
    def load_model(self):
        """Charger le modèle optimisé"""
        try:
            self.model = joblib.load(self.model_path / "trained_model_realistic.pkl")
            self.scaler = joblib.load(self.model_path / "scaler_realistic.pkl")
            
            with open(self.model_path / "features_realistic.json", 'r') as f:
                self.features = json.load(f)
            
            print(f"[OK] Model loaded with {len(self.features)} features")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    
    def bootstrap_prediction(self, X, n_bootstrap=50):
        """Prédiction avec bootstrap aggregating"""
        predictions = []
        
        for i in range(n_bootstrap):
            # Échantillonnage bootstrap des données
            sample_indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[sample_indices]
            
            # Prédiction sur l'échantillon
            pred_proba = self.model.predict_proba(X_bootstrap)
            predictions.append(pred_proba)
        
        # Moyenne des prédictions
        mean_proba = np.mean(predictions, axis=0)
        std_proba = np.std(predictions, axis=0)
        
        return mean_proba, std_proba
    
    def feature_perturbation_prediction(self, X, n_perturbations=30, perturbation_strength=0.05):
        """Prédiction avec perturbation légère des features (pour LogisticRegression)"""
        predictions = []
        
        for i in range(n_perturbations):
            # Perturbation aléatoire des features (au lieu de subsampling)
            perturbation = np.random.normal(0, perturbation_strength, X.shape)
            X_perturbed = X + perturbation
            
            # Prédiction sur données perturbées
            pred_proba = self.model.predict_proba(X_perturbed)
            predictions.append(pred_proba)
        
        # Moyenne des prédictions
        mean_proba = np.mean(predictions, axis=0)
        std_proba = np.std(predictions, axis=0)
        
        return mean_proba, std_proba
    
    def noise_injection_prediction(self, X, n_noise=25, noise_std=0.01):
        """Prédiction avec injection de bruit gaussien"""
        predictions = []
        
        for i in range(n_noise):
            # Ajouter du bruit gaussien
            noise = np.random.normal(0, noise_std, X.shape)
            X_noisy = X + noise
            
            # Prédiction sur données bruitées
            pred_proba = self.model.predict_proba(X_noisy)
            predictions.append(pred_proba)
        
        # Moyenne des prédictions
        mean_proba = np.mean(predictions, axis=0)
        std_proba = np.std(predictions, axis=0)
        
        return mean_proba, std_proba
    
    def ensemble_predict(self, X, methods=['bootstrap', 'perturbation', 'noise'], 
                        confidence_threshold=0.6):
        """
        Prédiction ensemble combinant plusieurs méthodes
        
        Args:
            X: Features d'entrée
            methods: Liste des méthodes à utiliser
            confidence_threshold: Seuil de confiance pour prédiction
        
        Returns:
            dict avec prédictions, confiances et incertitudes
        """
        all_predictions = []
        method_weights = {'bootstrap': 0.4, 'perturbation': 0.4, 'noise': 0.2}
        
        results = {}
        
        # Bootstrap sampling
        if 'bootstrap' in methods:
            bootstrap_pred, bootstrap_std = self.bootstrap_prediction(X, n_bootstrap=40)
            all_predictions.append(bootstrap_pred * method_weights['bootstrap'])
            results['bootstrap_std'] = bootstrap_std.mean(axis=1)
        
        # Feature perturbation (remplace subsampling)
        if 'perturbation' in methods:
            perturb_pred, perturb_std = self.feature_perturbation_prediction(X, n_perturbations=25)
            all_predictions.append(perturb_pred * method_weights['perturbation'])
            results['perturbation_std'] = perturb_std.mean(axis=1)
        
        # Noise injection
        if 'noise' in methods:
            noise_pred, noise_std = self.noise_injection_prediction(X, n_noise=20)
            all_predictions.append(noise_pred * method_weights['noise'])
            results['noise_std'] = noise_std.mean(axis=1)
        
        # Combinaison pondérée
        if all_predictions:
            final_proba = np.sum(all_predictions, axis=0)
            
            # Prédictions finales
            final_predictions = np.argmax(final_proba, axis=1)
            max_probabilities = np.max(final_proba, axis=1)
            
            # Convertir en labels
            label_map = {0: 'd', 1: 'l', 2: 'w'}
            final_labels = [label_map[pred] for pred in final_predictions]
            
            # Calculer incertitude globale
            overall_uncertainty = np.mean([
                results.get('bootstrap_std', np.zeros(len(X))),
                results.get('perturbation_std', np.zeros(len(X))),
                results.get('noise_std', np.zeros(len(X)))
            ], axis=0)
            
            # Classification par confiance
            high_confidence = max_probabilities > confidence_threshold
            medium_confidence = (max_probabilities > 0.45) & (max_probabilities <= confidence_threshold)
            low_confidence = max_probabilities <= 0.45
            
            results.update({
                'predictions': final_labels,
                'probabilities': final_proba,
                'max_probability': max_probabilities,
                'uncertainty': overall_uncertainty,
                'high_confidence': high_confidence,
                'medium_confidence': medium_confidence,
                'low_confidence': low_confidence,
                'confidence_distribution': {
                    'high': np.sum(high_confidence),
                    'medium': np.sum(medium_confidence), 
                    'low': np.sum(low_confidence)
                }
            })
            
            return results
        else:
            raise ValueError("No valid prediction methods specified")
    
    def evaluate_ensemble_improvement(self, test_data_path="preprocessed_data_enhanced.csv"):
        """Évaluer l'amélioration apportée par l'ensemble"""
        print("="*60)
        print("ENSEMBLE PREDICTION EVALUATION")
        print("="*60)
        
        # Charger données de test
        df = pd.read_csv(test_data_path)
        y_true = df['result']
        
        exclude_cols = [
            'result', 'team_id', 'opponent_id', 'date', 'team_name', 'opponent_name',
            'league', 'favorite_wins', 'favorite'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].select_dtypes(include=[np.number])
        
        # Sélectionner les features du modèle
        X_selected = X[self.features]
        X_scaled = self.scaler.transform(X_selected)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_true, test_size=0.2, random_state=42, stratify=y_true
        )
        
        print(f"[OK] Test set: {len(X_test)} samples")
        
        # 1. Prédiction standard (baseline)
        standard_pred = self.model.predict(X_test)
        standard_accuracy = accuracy_score(y_test, standard_pred)
        
        print(f"\n[BASELINE] Standard prediction:")
        print(f"Accuracy: {standard_accuracy:.4f}")
        
        # 2. Prédiction ensemble
        ensemble_results = self.ensemble_predict(
            X_test, 
            methods=['bootstrap', 'perturbation', 'noise'],
            confidence_threshold=0.6
        )
        
        ensemble_pred = ensemble_results['predictions']
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        print(f"\n[ENSEMBLE] Multiple predictions:")
        print(f"Accuracy: {ensemble_accuracy:.4f}")
        print(f"Improvement: {ensemble_accuracy - standard_accuracy:+.4f}")
        print(f"Confidence distribution: {ensemble_results['confidence_distribution']}")
        
        # 3. Analyse par niveau de confiance
        high_conf_mask = ensemble_results['high_confidence']
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = accuracy_score(
                y_test[high_conf_mask], 
                np.array(ensemble_pred)[high_conf_mask]
            )
            print(f"\n[HIGH CONFIDENCE] Predictions with >60% confidence:")
            print(f"Count: {np.sum(high_conf_mask)} samples")
            print(f"Accuracy: {high_conf_accuracy:.4f}")
        
        # 4. Classification report détaillé
        print(f"\n[DETAILED] Ensemble Classification Report:")
        print(classification_report(y_test, ensemble_pred))
        
        return {
            'standard_accuracy': standard_accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'improvement': ensemble_accuracy - standard_accuracy,
            'ensemble_results': ensemble_results
        }

def main():
    """Test du système ensemble"""
    try:
        # Créer le prédicteur ensemble
        ensemble_predictor = EnsemblePredictorModel()
        
        # Évaluer l'amélioration
        results = ensemble_predictor.evaluate_ensemble_improvement()
        
        print("="*60)
        print("ENSEMBLE EVALUATION COMPLETED")
        print("="*60)
        print(f"Standard Model: {results['standard_accuracy']:.4f}")
        print(f"Ensemble Model: {results['ensemble_accuracy']:.4f}")
        print(f"Improvement: {results['improvement']:+.4f}")
        
        if results['improvement'] > 0.005:  # Amélioration > 0.5%
            print("✅ SIGNIFICANT IMPROVEMENT - Ensemble reduces errors!")
        else:
            print("ℹ️  Minimal improvement - Standard model already optimal")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Ensemble evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()