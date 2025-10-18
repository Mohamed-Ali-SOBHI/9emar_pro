# 📁 Project Structure - Football Match Prediction ML Pipeline

## 🗂️ Final Clean Structure

```
C:\Users\moham\Downloads\foot\
├── 📄 CLAUDE.md                              # Project documentation & instructions
├── 📄 PROJECT_STRUCTURE.md                   # This structure guide
├── 🐍 train_complete_pipeline.py             # Main training pipeline
├── 🐍 prediction_complete_pipeline.py        # Main prediction pipeline
├── 
├── 📊 DATA FILES
├── 📄 preprocessed_data.csv                  # Basic preprocessed data (68 features)
├── 📄 preprocessed_data_with_odds.csv        # Data + betting odds (68 features)
├── 📄 preprocessed_data_enhanced.csv         # Advanced features (267 features)
├── 
├── 🗂️ Data/                                  # Raw match data by league/year
├── ├── 📁 Bundesliga/                        # Raw Bundesliga data files
├── ├── 📁 EPL/                               # Raw Premier League data files
├── ├── 📁 La_liga/                           # Raw La Liga data files
├── ├── 📁 Ligue_1/                           # Raw Ligue 1 data files
├── ├── 📁 Serie_A/                           # Raw Serie A data files
├── └── 📁 odds/                              # Betting odds by league
├── 
├── 🗂️ models/
├── └── 📁 optimal_model/                     # Final production model
├──     ├── 🔧 trained_model_realistic.pkl    # Trained LogisticRegression model
├──     ├── 🔧 scaler_realistic.pkl           # Feature scaler
├──     ├── 📄 features_realistic.json        # 60 selected features
├──     └── 📄 metrics_realistic.json         # Model performance metrics
├── 
├── 🗂️ utils/
├── ├── 📁 common/                            # Shared utilities
├── │   ├── 🐍 enhanced_feature_engineering.py # 267 advanced features
├── │   └── 🐍 preprocessing.py               # Core preprocessing
├── ├── 
├── ├── 📁 train/                             # Training pipeline
├── │   ├── 🐍 StatisticsScrapper.py          # Data collection from understat.com
├── │   ├── 🐍 advanced_merge.py              # Merge with betting odds
├── │   └── 🐍 realistic_training.py          # Final model training
├── ├── 
├── └── 📁 predict/                           # Prediction pipeline
├──     ├── 🐍 BWINScrapper.py                # BWIN odds scraping
├──     ├── 🐍 LivescoreScrapper.py           # Live match data
├──     ├── 🐍 predict_matches.py             # Match prediction logic
├──     ├── 🐍 run_predictions.py             # Automated predictions
├──     └── 🐍 odds api.py                    # Odds API integration
├── 
├── 🗂️ Match of ze day/                       # Current season data
├── 🗂️ predictions_results/                   # Generated predictions
└── 🗂️ venv/                                  # Virtual environment
```

## 🚀 Key Commands

### Training Pipeline
```bash
# Complete training from scratch
python train_complete_pipeline.py

# Just train model (if data exists)
python utils/train/realistic_training.py
```

### Prediction Pipeline
```bash
# Generate predictions for today
python prediction_complete_pipeline.py

# Manual prediction run
python utils/predict/run_predictions.py
```

### Feature Engineering
```bash
# Generate enhanced features
python utils/common/enhanced_feature_engineering.py
```

## 🏆 Final Model Performance

- **Algorithm**: LogisticRegression (production-optimized)
- **Accuracy**: 51.8% (vs 33.3% random baseline)
- **Draw Recall**: 34.5% (excellent balance)
- **Features**: 60 selected from 267 engineered features
- **Cross-Validation**: 50.2% (stable)
- **Status**: Production-ready ✅

## 🧹 Cleaned Up (Removed)

### ❌ Test Scripts Removed
- `utils/train/advanced_training.py` - Advanced models test
- `utils/train/optimized_training.py` - Optimization experiments  
- `utils/train/xgboost_training.py` - XGBoost experiments

### ❌ Duplicate Files Removed
- `utils/train/enhanced_feature_engineering.py` - Moved to common/
- `utils/train/preprocessing.py` - Moved to common/
- `utils/predict/enhanced_feature_engineering.py` - Removed duplicate
- `preprocessed_data_enhanced_no_uncertainty.csv` - Temporary file

### ❌ Underperforming Models Removed
- `models/optimized_model/` - 53% accuracy but 2% draw recall

## 📊 Data Flow Summary

```
Raw Data (1,184 files) 
    ↓ [StatisticsScrapper.py]
preprocessed_data.csv (68 features)
    ↓ [advanced_merge.py]  
preprocessed_data_with_odds.csv (68 features)
    ↓ [enhanced_feature_engineering.py]
preprocessed_data_enhanced.csv (267 features)
    ↓ [realistic_training.py]
Final Model (60 best features → 51.8% accuracy)
```

## 🎯 Next Steps for Improvement

To exceed 52% accuracy, integrate external data:
1. **Weather data** (OpenWeatherMap API)
2. **Injury reports** (ESPN/Sky Sports scraping)
3. **Motivation context** (league standings, enjeux)
4. **Referee data** (historical tendencies)

Current model represents the ceiling for internal data analysis - external data needed for further improvement.