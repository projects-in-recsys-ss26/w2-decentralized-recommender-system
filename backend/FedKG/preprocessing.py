import pandas as pd

# Venue categories that represent a user's fixed "anchor" locations: home,
# office and the place they live. A user who mostly just bounces between these
# few routine places carries little useful next-POI signal, so we drop them.
ROUTINE_CATEGORIES = {
    "Home (private)",
    "Office",
    "Residential Building (Apartment / Condo)",
    "Housing Development",
}


def filter_routine_users(df, routine_categories=ROUTINE_CATEGORIES,
                         max_routine_ratio=0.5, min_checkins=5):
    """Remove users dominated by routine home/office check-ins.

    For every user we compute the share of check-ins that fall into
    `routine_categories` (home, office, residential...). If that share is at or
    above `max_routine_ratio`, the user is treated as a low-signal "commuter"
    (too many repeating home/office visits) and removed entirely. Users with
    fewer than `min_checkins` check-ins are always kept (too few to judge).
    """
    is_routine = df['category_name'].isin(routine_categories)
    total = df.groupby('user_id').size()                      # check-ins per user
    routine = is_routine.groupby(df['user_id']).sum()         # routine check-ins per user
    ratio = (routine / total).fillna(0.0)

    drop_users = set(ratio.index[(ratio >= max_routine_ratio) & (total >= min_checkins)])
    print(f"Filtering routine (home/office) users: removing {len(drop_users)} "
          f"of {total.shape[0]} users (routine check-in share >= {max_routine_ratio:.0%}).")

    return df[~df['user_id'].isin(drop_users)].copy()


def preprocess_foursquare_data(input_file, output_file,
                               filter_routine=True, max_routine_ratio=0.5,
                               min_checkins=5):
    print(f"Reading original data from '{input_file}'...")

    # Define column names based on the TSMC2014 Foursquare dataset format
    columns = [
        'user_id', 'venue_id', 'category_id', 'category_name',
        'latitude', 'longitude', 'timezone_offset', 'utc_time'
    ]

    # Load the tab-separated text file
    df = pd.read_csv(input_file, sep='\t', header=None, names=columns, encoding='latin-1')
    # ---------------------------------------------------------
    # 1. Convert Timestamp
    # ---------------------------------------------------------
    print("Converting timestamps to ISO 8601 format...")
    # Convert string format like "Tue Apr 03 18:00:09 +0000 2012" to datetime objects
    df['utc_time'] = pd.to_datetime(df['utc_time'], format='%a %b %d %H:%M:%S %z %Y')
    # Format the datetime objects into ISO strings (e.g., 2012-04-07T17:42:24Z)
    df['iso_time'] = df['utc_time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    # ---------------------------------------------------------
    # 1b. Filter out routine (home/office heavy) users
    # ---------------------------------------------------------
    # Done BEFORE re-indexing so the mapped user/venue IDs stay contiguous and
    # the item vocabulary (itemnum) does not contain gaps from removed users.
    if filter_routine:
        print("Filtering users with too many repeating home/office check-ins...")
        df = filter_routine_users(df, max_routine_ratio=max_routine_ratio,
                                  min_checkins=min_checkins)

    # ---------------------------------------------------------
    # 2. Re-Indexing (Mapping to continuous integers)
    # ---------------------------------------------------------
    print("Mapping cryptic IDs to continuous integers...")
    # pd.factorize() converts unique strings into numeric IDs starting at 0. 
    # We add 1 to start from 1.
    df['mapped_user_id'] = pd.factorize(df['user_id'])[0] + 1
    df['mapped_venue_id'] = pd.factorize(df['venue_id'])[0] + 1
    df['mapped_category_id'] = pd.factorize(df['category_id'])[0] + 1
    
    # ---------------------------------------------------------
    # 3. Trajectory Sorting
    # ---------------------------------------------------------
    print("Sorting data by user and chronologically by time...")
    # Sort primarily by the new User ID, then chronologically by time
    df = df.sort_values(by=['mapped_user_id', 'utc_time'])
    
    # ---------------------------------------------------------
    # 4. Save the Output File
    # ---------------------------------------------------------
    # Select only the relevant columns in the desired order
    output_columns = [
        'mapped_user_id', 'mapped_venue_id', 'mapped_category_id',
        'category_name', 'latitude', 'longitude', 
        'timezone_offset', 'iso_time'
    ]
    
    df_final = df[output_columns]
    
    print(f"Saving processed data to '{output_file}'...")
    # Save as tab-separated file without header and index
    df_final.to_csv(output_file, sep='\t', index=False, header=False)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    # Define file paths
    INPUT_PATH = "data/dataset_TSMC2014_NYC.txt"
    OUTPUT_PATH = "data/processed_nyc.txt"

    # filter_routine=True drops users whose check-ins are dominated by routine
    # home/office places; raise max_routine_ratio to keep more users, or set
    # filter_routine=False to disable the filter entirely.
    preprocess_foursquare_data(
        INPUT_PATH, OUTPUT_PATH,
        filter_routine=True,
        max_routine_ratio=0.3,
        min_checkins=20,
    )