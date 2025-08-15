import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import from common modules to avoid duplication
from utils.common.preprocessing import (
    load_all_data,
    remove_duplicate_matches,
    calculate_rolling_stats_optimized,
    calculate_head_to_head_stats_optimized,
    calculate_form_indicators,
    calculate_season_metrics,
    remove_post_match_features,
    add_feature_engineering,
    preprocess_data_memory_only,
    preprocess_data,
    analyze_data_quality
)

# All functions are now imported from common module - no need to redefine them
if __name__ == "__main__":
    # Run preprocessing
    processed_df = preprocess_data(add_features=True, remove_duplicates_flag=True) 
    
    # Analyze data quality if df is not empty
    if processed_df is not None and not processed_df.empty:
        analyze_data_quality(processed_df)
    else:
        print("Preprocessing returned an empty or None DataFrame. Skipping analysis.")