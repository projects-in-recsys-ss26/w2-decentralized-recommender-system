import os
import pandas as pd

def read_data(filepath: str) -> pd.DataFrame:
    """Loads a Parquet file efficiently into RAM."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found!")
        
    print(f"Loading data from: {filepath}...")
    df = pd.read_parquet(filepath)
    print(f"-> Successfully loaded. Shape: {df.shape} (rows, columns)")
    return df


def merge_categories(checkins_df: pd.DataFrame, categories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Links check-ins with hierarchical categories.
    Uses a left-join to avoid losing any check-ins.
    """
    print("Linking check-ins with category hierarchy...")
    
    # Column names differ slightly between the two dataframes:
    # checkins_df: 'venue_category_id'
    # categories_df: 'category_id'
    merged_df = pd.merge(
        checkins_df,
        categories_df,
        left_on='venue_category_id',
        right_on='category_id',
        how='left'  # Left join ensures all check-ins are retained
    )
    
    # Cleanup: Remove the duplicate ID column 'category_id'
    merged_df = merged_df.drop(columns=['category_id'])
    
    # Quality check: Were there IDs that were not in the mapping?
    missing_mask = merged_df['level_1'].isna()
    missing_count = missing_mask.sum()
    
    if missing_count > 0:
        print(f"⚠️ Warning: No category hierarchy found for {missing_count} check-ins.")
        print("Here are the affected entries:")
        
        # Filter out faulty rows and display only relevant columns
        missing_rows = merged_df[missing_mask][
            ['user_id', 'venue_id', 'venue_category_id', 'venue_category_name']
        ]
        
        # to_string() ensures Pandas prints everything nicely without abbreviation
        print(missing_rows.to_string())
    else:
        print("✅ All categories successfully mapped!")
        
    return merged_df

def fix_missing_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces deprecated or missing Foursquare Category IDs with valid alternatives.
    """
    print("Fixing known missing category IDs...")
    
    # Mapping of old ID -> new ID
    replacements = {
        "4e51a0c0bd41d3446defbb2e": "4bf58dd8d48988d12d951735" # Ferry -> Boat and Ferry
    }
    
    # replace() searches for old keys in the column and replaces them with new values
    df['venue_category_id'] = df['venue_category_id'].replace(replacements)
    
    return df


def print_statistics(df: pd.DataFrame) -> None:
    """
    Prints detailed statistics about the DataFrame:
    - Number of users
    - Check-in statistics per user (mean, median, min, max)
    - Time range of check-ins
    - Venue and Category counts
    """
    print("\n" + "="*50)
    print("📊 DATA STATISTICS")
    print("="*50)
    
    # Number of users
    num_users = df['user_id'].nunique()
    print(f"\n👥 Number of unique users: {num_users}")
    
    # Total number of check-ins
    num_checkins = len(df)
    print(f"📍 Total number of check-ins: {num_checkins}")
    
    # Venues and Categories
    num_venues = df['venue_id'].nunique()
    num_specific_cats = df['venue_category_name'].nunique()
    num_level1_cats = df['level_1'].nunique()
    
    print(f"🏪 Unique Venues (Locations): {num_venues}")
    print(f"🏷️ Unique Specific Categories: {num_specific_cats}")
    print(f"🌍 Unique Level-1 Categories: {num_level1_cats}")
    
    # Check-ins per user
    checkins_per_user = df.groupby('user_id').size()
    print(f"\n📈 Check-ins per user:")
    print(f"   Average: {checkins_per_user.mean():.2f}")
    print(f"   Median:  {checkins_per_user.median():.2f}")
    print(f"   Minimum: {checkins_per_user.min()}")
    print(f"   Maximum: {checkins_per_user.max()}")
    
    # Time range
    if 'timestamp' in df.columns:
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        print(f"\n⏰ Time range of check-ins:")
        print(f"   From: {min_time}")
        print(f"   To:   {max_time}")
    
    # Column info
    print(f"\n📋 DataFrame shape: {df.shape} (rows × columns)")
    print(f"   Columns: {', '.join(df.columns.tolist())}")
    
    # Trainable days statistics (days with >= 5 check-ins per user)
    # Extract date from utc_time
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['utc_time']).dt.date
    
    # Group by user and date, count check-ins per day
    daily_checkins = df_copy.groupby(['user_id', 'date']).size()
    
    # Filter days with >= 5 check-ins per user
    days_with_5plus = (daily_checkins >= 5).groupby('user_id').sum()
    
    # Total number of trainable days
    total_trainable_days = days_with_5plus.sum()
    
    print(f"\n🎯 Trainable days (≥5 check-ins per day):")
    print(f"   Total trainable day-records: {total_trainable_days}")
    print(f"   Average per user: {days_with_5plus.mean():.2f}")
    print(f"   Median per user:  {days_with_5plus.median():.2f}")
    print(f"   Min per user:     {days_with_5plus.min()}")
    print(f"   Max per user:     {days_with_5plus.max()}")
    
    print("="*50 + "\n")

def pipeline(checkins_path: str, categories_path: str):
    """Main pipeline, controlling everything."""
    print("=== Starting Data Preprocessing ===")
    
    # 1. Load data
    df_checkins = read_data(checkins_path)
    df_categories = read_data(categories_path)

    # 2. Fix missing categories (e.g., deprecated Ferry ID)
    df_checkins = fix_missing_categories(df_checkins)
    
    # 3. Merge
    df_final = merge_categories(df_checkins, df_categories)

    # 4. Print statistics
    print_statistics(df_final)
    
    # 5. Return 
    return df_final