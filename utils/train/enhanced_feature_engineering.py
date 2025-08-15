import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import from common modules to avoid duplication
from utils.common.enhanced_feature_engineering import (
    add_advanced_team_features,
    add_opponent_comparative_features,
    add_betting_intelligence_features,
    add_temporal_features,
    add_form_momentum_features,
    add_home_away_features,
    add_interaction_features,
    add_all_enhanced_features,
    enhance_dataset
)

# All functions are now imported from common module - no need to redefine them
if __name__ == "__main__":
    enhanced_df = enhance_dataset()
    print("Enhanced feature engineering completed!")