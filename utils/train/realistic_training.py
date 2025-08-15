#!/usr/bin/env python3
"""
Training réaliste sans artifices pour obtenir des performances honnêtes.
Objectif: modèle équilibré et utilisable en production.
"""

import sys
import os
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

def load_original_enhanced_data():
    """Charge les données avec features enhanced mais sans artifices"""
    print("[1/6] Loading realistic enhanced data...")
    
    df = pd.read_csv("preprocessed_data_enhanced_no_uncertainty.csv")
    y = df['result']
    
    # Exclusions minimales - seulement les vraiment nécessaires
    exclude_cols = [
        'result', 'team_id', 'opponent_id', 'date', 'team_name', 'opponent_name',
        'league', 'favorite_wins', 'favorite'
        # match_uncertainty déjà supprimée du dataset
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])
    
    print(f"[OK] Data loaded: {len(df)} samples, {X.shape[1]} features")
    print(f"[OK] Exclusions minimales appliquées - features H2H et draw incluses")
    print(f"[OK] Target distribution:")
    for val, count in y.value_counts().items():
        print(f"   {val}: {count} ({count/len(y)*100:.1f}%)")
    
    return X, y

def realistic_model_training(X_train, X_test, y_train, y_test):
    """Entraînement réaliste sans class weights extrêmes"""
    print("[3/6] Realistic model training...")
    
    # Class weights modérés (pas de boost artificiel)
    models = {
        "LogisticRegression_Realistic": {
            "model": LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='lbfgs',
                class_weight='balanced',  # Standard balanced
                random_state=42,
                max_iter=1000,
                multi_class='ovr'
            ),
            "use_scaling": True
        },
        "RandomForest_Balanced": {
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                class_weight='balanced',  # Standard balanced
                random_state=42,
                n_jobs=-1
            ),
            "use_scaling": False
        },
        "LogisticRegression_NoWeights": {
            "model": LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='lbfgs',
                class_weight=None,  # Pas de weights du tout
                random_state=42,
                max_iter=1000,
                multi_class='ovr'
            ),
            "use_scaling": True
        }
    }
    
    # Scale once
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    best_model = None
    best_overall_score = 0
    best_name = None
    best_metrics = None
    
    for name, config in models.items():
        print(f"\n--- Testing {name} ---")
        
        model = config["model"]
        
        if config["use_scaling"]:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test
        
        try:
            # Train
            model.fit(X_tr, y_train)
            
            # Evaluate
            y_test_pred = model.predict(X_te)
            
            # Metrics
            test_acc = accuracy_score(y_test, y_test_pred)
            test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
            test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
            
            # Per-class metrics
            report = classification_report(y_test, y_test_pred, output_dict=True)
            draw_precision = report.get('d', {}).get('precision', 0)
            draw_recall = report.get('d', {}).get('recall', 0)
            draw_f1 = report.get('d', {}).get('f1-score', 0)
            
            # Predictions distribution
            pred_counts = pd.Series(y_test_pred).value_counts()
            actual_counts = y_test.value_counts()
            
            # Over-prediction penalty
            draw_overpred = pred_counts.get('d', 0) - actual_counts.get('d', 0)
            overpred_penalty = max(0, draw_overpred / actual_counts.get('d', 1)) * 0.1
            
            print(f"Accuracy: {test_acc:.4f}")
            print(f"Weighted F1: {test_f1_weighted:.4f}")
            print(f"Macro F1: {test_f1_macro:.4f}")
            print(f"Draw - P: {draw_precision:.4f}, R: {draw_recall:.4f}, F1: {draw_f1:.4f}")
            print(f"Draw over-prediction: {draw_overpred:+d} ({draw_overpred/actual_counts.get('d', 1)*100:+.1f}%)")
            
            # Overall score (balance entre accuracy et équilibre)
            overall_score = (test_acc * 0.6 + test_f1_macro * 0.4) - overpred_penalty
            
            results[name] = {
                "model": model,
                "scaler": scaler if config["use_scaling"] else None,
                "test_accuracy": test_acc,
                "test_f1_weighted": test_f1_weighted,
                "test_f1_macro": test_f1_macro,
                "draw_precision": draw_precision,
                "draw_recall": draw_recall,
                "draw_f1": draw_f1,
                "draw_overprediction": draw_overpred,
                "overall_score": overall_score,
                "use_scaling": config["use_scaling"]
            }
            
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_model = model
                best_name = name
                best_metrics = results[name]
                
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            continue
    
    print(f"\n[BEST REALISTIC MODEL] {best_name}")
    print(f"Overall Score: {best_metrics['overall_score']:.4f}")
    print(f"Accuracy: {best_metrics['test_accuracy']:.4f}")
    print(f"Draw Recall: {best_metrics['draw_recall']:.4f}")
    print(f"Draw Over-prediction: {best_metrics['draw_overprediction']:+d}")
    
    return best_model, best_name, best_metrics, results

def final_realistic_evaluation(model, X_test, y_test, model_name, scaler=None):
    """Évaluation finale réaliste"""
    print("[4/6] Final realistic evaluation...")
    
    if scaler is not None:
        X_test_eval = scaler.transform(X_test)
    else:
        X_test_eval = X_test
    
    y_pred = model.predict(X_test_eval)
    
    print(f"\nFinal Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    
    # Analyse de réalisme
    actual_dist = y_test.value_counts(normalize=True).sort_index()
    pred_dist = pd.Series(y_pred).value_counts(normalize=True).sort_index()
    
    print(f"\nRealism Check - Distribution Comparison:")
    print(f"Class    Actual   Predicted   Difference")
    for cls in ['d', 'l', 'w']:
        actual_pct = actual_dist.get(cls, 0) * 100
        pred_pct = pred_dist.get(cls, 0) * 100
        diff = pred_pct - actual_pct
        print(f"  {cls}     {actual_pct:5.1f}%      {pred_pct:5.1f}%      {diff:+5.1f}%")
    
    # Verdict de réalisme
    max_diff = max(abs(pred_dist.get(cls, 0) - actual_dist.get(cls, 0)) for cls in ['d', 'l', 'w'])
    
    if max_diff < 0.1:
        realism = "EXCELLENT - Distributions très proches"
    elif max_diff < 0.15:
        realism = "GOOD - Distributions acceptables"
    elif max_diff < 0.2:
        realism = "ACCEPTABLE - Léger biais"
    else:
        realism = "POOR - Distributions déséquilibrées"
    
    print(f"\nRealism Verdict: {realism}")
    print(f"Max distribution difference: {max_diff:.3f}")
    
    return y_pred

def main():
    """Pipeline réaliste principal"""
    print("="*80)
    print("REALISTIC TRAINING PIPELINE (PRODUCTION-READY)")
    print("="*80)
    
    # Load realistic data
    X, y = load_original_enhanced_data()
    
    # Clean data
    imputer = SimpleImputer(strategy='median')
    X_cleaned = imputer.fit_transform(X)
    X = pd.DataFrame(X_cleaned, columns=X.columns)
    
    # Conservative feature selection (top 60 features)
    print("[2/6] Conservative feature selection...")
    selector = SelectKBest(score_func=f_classif, k=60)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    X_selected = pd.DataFrame(X_selected, columns=selected_features)
    print(f"[OK] Selected {len(selected_features)} most important features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[OK] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train realistic models
    best_model, best_name, best_metrics, all_results = realistic_model_training(
        X_train, X_test, y_train, y_test
    )
    
    if best_model is None:
        return False
    
    # Final evaluation
    y_pred = final_realistic_evaluation(
        best_model, X_test, y_test, best_name, 
        best_metrics.get('scaler')
    )
    
    # Cross-validation
    print("[5/6] Cross-validation...")
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Update final metrics
    best_metrics.update({
        'cv_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_features': len(selected_features),
        'n_samples': len(X),
        'model_type': best_name,
        'realistic_training': True
    })
    
    # Save realistic model
    print("[6/6] Saving realistic model...")
    models_dir = Path("models/optimal_model")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(best_model, models_dir / "trained_model_realistic.pkl")
    
    # Save scaler if used
    if best_metrics.get('scaler') is not None:
        joblib.dump(best_metrics['scaler'], models_dir / "scaler_realistic.pkl")
    
    # Save metrics
    clean_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for k, v in best_metrics.items() if k not in ['model', 'scaler']}
    
    with open(models_dir / "metrics_realistic.json", 'w') as f:
        json.dump(clean_metrics, f, indent=2)
    
    with open(models_dir / "features_realistic.json", 'w') as f:
        json.dump(selected_features, f, indent=2)
    
    # Final summary
    print("\n" + "="*80)
    print("REALISTIC TRAINING COMPLETED")
    print("="*80)
    print(f"Model: {best_name}")
    print(f"Accuracy: {best_metrics['test_accuracy']:.4f}")
    print(f"CV Accuracy: {best_metrics['cv_accuracy']:.4f}")
    print(f"Macro F1: {best_metrics['test_f1_macro']:.4f}")
    print(f"Draw Recall: {best_metrics['draw_recall']:.4f} ({best_metrics['draw_recall']*100:.1f}%)")
    print(f"Draw Precision: {best_metrics['draw_precision']:.4f}")
    print(f"Draw Over-prediction: {best_metrics['draw_overprediction']:+d}")
    print(f"Features: {best_metrics['n_features']}")
    
    # Production readiness assessment
    acc = best_metrics['test_accuracy']
    draw_recall = best_metrics['draw_recall']
    overpred = abs(best_metrics['draw_overprediction'])
    
    if acc > 0.50 and draw_recall > 0.25 and overpred < 100:
        verdict = "PRODUCTION READY"
    elif acc > 0.45 and draw_recall > 0.20:
        verdict = "ACCEPTABLE FOR TESTING"
    else:
        verdict = "NEEDS IMPROVEMENT"
    
    print(f"\nProduction Assessment: {verdict}")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)