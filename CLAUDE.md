# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 🏈 Football Match Prediction - ML Pipeline

## Project Overview

Machine learning project that predicts football match results (win/draw/loss) using team statistics and betting odds. The pipeline processes data from 5 major European leagues (2014-2024) and achieves 53.08% accuracy using a realistic LogisticRegression model.

## Core Architecture

### Data Pipeline Flow
1. **Data Collection** (`utils/StatisticsScrapper.py`) - Scrapes match statistics from understat.com
2. **Preprocessing** (`utils/preprocessing.py`) - Feature engineering and data cleaning
3. **Odds Integration** (`utils/advanced_merge.py`) - Merges betting odds with match data
4. **Advanced Features** (`utils/enhanced_feature_engineering.py`) - Creates complex derived features
5. **Model Training** (`utils/realistic_training.py`) - Final model training (53.08% accuracy)

### Key Components

**Main Pipeline Scripts:**
- `main_complete_pipeline.py` - Complete end-to-end pipeline (scraping → training)
- `data_pipeline.py` - Smart pipeline (predictions if model exists, else data preparation)
- `predict_match.py` - Individual match prediction with examples

**Data Processing Utils:**
- `utils/preprocessing.py` - Core data processing and feature engineering
- `utils/advanced_merge.py` - Betting odds integration and team name normalization
- `utils/enhanced_feature_engineering.py` - Advanced statistical features
- `utils/realistic_training.py` - Production model training without artificial boosting

**Model Storage:**
- `models/optimal_model/` - Contains trained model, scaler, features list, and metrics

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (Windows)
venv\Scripts\activate

# Virtual environment is included in repo with dependencies pre-installed
```

### Pipeline Execution
```bash
# Complete pipeline (data collection → model training)
python main_complete_pipeline.py

# Smart pipeline (predictions if model exists, else data preparation)
python data_pipeline.py

# Individual match predictions with examples
python predict_match.py

# Train only the final model (requires preprocessed data)
python utils/realistic_training.py
```

### Individual Pipeline Steps
```bash
# Data scraping only
python utils/StatisticsScrapper.py

# Preprocessing only (requires raw data in Data/ directory)
python utils/preprocessing.py

# Merge with betting odds (requires preprocessed_data.csv)
python utils/advanced_merge.py

# Enhanced feature engineering (requires preprocessed_data_with_odds.csv)
python utils/enhanced_feature_engineering.py
```

## Data Structure

### Generated Datasets
1. **`preprocessed_data.csv`** - Basic preprocessed match data
2. **`preprocessed_data_with_odds.csv`** - Data merged with betting odds
3. **`preprocessed_data_enhanced.csv`** - Advanced features (130+ features)

### Data Coverage
- **Leagues:** Bundesliga, EPL, La Liga, Ligue 1, Serie A
- **Period:** 2014-2024 (10 seasons)
- **Matches:** 11,520 matches analyzed
- **Features:** 60 optimized features (xG stats, form, H2H, odds)

### Directory Structure
```
Data/
├── Bundesliga/          # Raw match data by team/season
├── EPL/
├── La_liga/
├── Ligue_1/
├── Serie_A/
└── odds/                # Betting odds by league
    ├── Bundesliga/
    ├── EPL/
    ├── La_liga/
    ├── Ligue_1/
    └── Serie_A/

Match of ze day/         # Current season data for predictions
├── Bundesliga/
├── EPL/
├── La_liga/
├── Ligue_1/
└── Serie_A/
```

## Model Architecture & Performance

### Final Model Configuration
- **Algorithm:** LogisticRegression (no artificial class weights)
- **Accuracy:** 53.08% (vs 33.3% random baseline)
- **F1 Macro:** 42.2%
- **Draw Recall:** 4.4% (realistic - draws are inherently unpredictable)

```python
# Production model configuration
LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    class_weight=None,  # No artificial weights
    random_state=42,
    max_iter=1000,
    multi_class='ovr'
)
```

### Class Distribution
- **Loss (defeats):** 43.4% of matches
- **Win (victories):** 31.2% of matches  
- **Draw:** 25.5% of matches

### Tested Approaches (All Rejected)
- ❌ **Ensemble Learning:** 52.39% (-0.69%)
- ❌ **LSTM Temporal Features:** No improvement
- ❌ **Smart Features + Market Intelligence:** 52.08% (-1.00%)

**Conclusion:** 53.08% appears to be the natural ceiling for this prediction approach.

## Development Guidelines

### Code Architecture Patterns

**Pipeline Design**: The project follows a sequential pipeline pattern where each step depends on the previous one's output:
1. Raw data (CSV files in `Data/` directory)
2. Preprocessed data (`preprocessed_data.csv`)
3. Merged with odds (`preprocessed_data_with_odds.csv`)
4. Enhanced features (`preprocessed_data_enhanced.csv`)
5. Trained model (`models/optimal_model/`)

**Error Handling**: All pipeline scripts include robust error handling and file existence checks. They can skip steps if output already exists.

**Subprocess Pattern**: `main_complete_pipeline.py` uses subprocess calls to run individual utilities, avoiding import conflicts and ensuring clean execution environments.

### Feature Engineering Principles

**Anti-Patterns to Avoid** (based on project learnings):
- SMOTE on temporal data (causes data leakage)
- Extreme class weights (creates artificial over-prediction)
- Draw-specific artificial features
- Over-optimization without real improvement

**Validated Approach:**
- Simple, robust models
- Clean features without leakage
- Honest performance validation
- Acceptance of natural performance ceiling

### Prediction System Usage

```python
# Example prediction workflow
from predict_match import FootballPredictor

predictor = FootballPredictor()
predictor.load_model()

# Match data with key features
match_data = {
    'team_xG_last_5': 2.1,      # Team offensive performance
    'opponent_xG_last_5': 0.9,   # Opponent performance  
    'form_score_5': 0.8,         # Recent form
    'B365H': 1.8,               # Home odds
    'B365D': 3.6,               # Draw odds
    'B365A': 4.5,               # Away odds
    # ... other required features
}

result = predictor.predict_match(match_data)
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Testing & Validation

**No Automated Tests**: This project does not include unit tests or automated testing frameworks. Testing is done through pipeline execution and metric validation.

**Model Validation**: The realistic_training.py script includes cross-validation and proper train/test splits to ensure honest performance metrics.

**Data Validation**: Each pipeline step validates input data existence and format before processing.

### Dependencies & Environment

**Virtual Environment**: The project includes a `venv/` directory with pre-installed dependencies. No separate requirements.txt file exists.

**Key Libraries**: pandas, numpy, scikit-learn, joblib, tqdm, beautifulsoup4, requests (for scraping)

### Production Considerations

**Model Ready For:**
- ✅ Football match analysis
- ✅ Sports decision support
- ✅ Academic research
- ⚠️ Sports betting (with caution - 53.08% is modest)

**Performance Ceiling**: 53.08% accuracy represents the realistic ceiling for this approach. The project philosophy emphasizes honest, production-ready results over artificially inflated metrics.

---

**Final Result**: An honest model achieving 53.08% accuracy that respects the complex reality of football prediction, without misleading technical artifices.