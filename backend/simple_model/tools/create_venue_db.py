import pandas as pd
import os

def create_venue_db(df=None, user_features_df=None, output_path=None):
    if df is None:
        input_file = "../../../data/foursquare_checkins_nyc.parquet"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(base_dir, input_file)
        print(f"Reading check-ins from {input_path}...")
        try:
            df = pd.read_parquet(input_path)
        except Exception as e:
            print(f"Error reading {input_path}: {e}")
            return None

    if output_path is None:
        output_file = "../../../data/venues.parquet"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, output_file)
    
    print(f"Found check-ins: {len(df)}")
    
    # Spalten, die wir für die Venues brauchen
    required_cols = ['venue_id', 'latitude', 'longitude', 'venue_category_name']
    
    # Prüfe ob alle Spalten existieren
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in file: {missing_cols}")
        print(f"Existing columns: {df.columns.tolist()}")
        return None
        
    print("Extracting unique venues and calculating popularity...")
    # Zähle, wie oft jeder Ort besucht wurde (Global Popularität)
    venue_counts = df.groupby('venue_id').size().reset_index(name='checkin_count')
    
    # Gruppiere nach venue_id und nimm den ersten Eintrag (da sich Koordinaten/Kategorie nicht ändern sollten)
    venues_df = df[required_cols].drop_duplicates(subset=['venue_id'])
    
    # Verbinde die Popularitäts-Metrik mit den Orten
    venues_df = pd.merge(venues_df, venue_counts, on='venue_id')
    
    # Cluster-spezifische Counts berechnen, falls user_features_df übergeben wurde
    if user_features_df is not None and 'cluster' in user_features_df.columns:
        print("Calculating cluster-specific visit counts...")
        # Merge Check-ins mit User-Clustern
        merged_df = df.merge(user_features_df[['user_id', 'cluster']], on='user_id', how='inner')
        
        # Zähle Besuche pro Venue und Cluster
        cluster_counts = merged_df.groupby(['venue_id', 'cluster']).size().reset_index(name='count')
        
        # Pivotieren: Jede Cluster-ID wird eine Spalte (z.B. 0 -> cluster_0_count)
        cluster_pivot = cluster_counts.pivot(index='venue_id', columns='cluster', values='count').fillna(0)
        
        # Spalten umbenennen
        cluster_pivot.columns = [f'cluster_{int(c)}_count' for c in cluster_pivot.columns]
        cluster_pivot = cluster_pivot.reset_index()
        
        # An venues_df anfügen
        venues_df = pd.merge(venues_df, cluster_pivot, on='venue_id', how='left')
        
        # Für Venues ohne Besuche von bestimmten Clustern N/A mit 0 füllen
        cluster_cols = [col for col in venues_df.columns if col.startswith('cluster_') and col.endswith('_count')]
        venues_df[cluster_cols] = venues_df[cluster_cols].fillna(0).astype(int)
    
    # Optional: umbenennen, falls gewünscht
    venues_df = venues_df.rename(columns={
        'venue_category_name': 'category'
    })
    
    # Sortiere nach Popularität
    venues_df = venues_df.sort_values('checkin_count', ascending=False)
    
    print(f"Found unique venues: {len(venues_df)}")
    
    print(f"Saving venues to {output_path}...")
    
    # Verzeichnis sicherstellen
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    venues_df.to_parquet(output_path, index=False)
    print("Done! 🎉")
    
    return venues_df

if __name__ == "__main__":
    create_venue_db()
