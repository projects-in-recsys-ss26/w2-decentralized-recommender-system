import pandas as pd
import numpy as np


def split_data_chronologically(df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2):
    """
    Splittet die Daten chronologisch in Train, Val und Test, 
    um Data Leakage (Blick in die Zukunft) zu vermeiden.
    """
    print("🔄 Splitting data chronologically...")
    
    # Sicherstellen, dass nach Zeit sortiert ist
    if not pd.api.types.is_datetime64_any_dtype(df['utc_time']):
        df['utc_time'] = pd.to_datetime(df['utc_time'])
    df = df.sort_values(by='utc_time').reset_index(drop=True)
    
    n_total = len(df)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"📊 Split result: Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})")
    return train_df, val_df, test_df

def evaluate_recommender(model, test_df: pd.DataFrame):
    """
    Bewertet das Modell basierend auf der Hit Rate für spezifische und Level-1-Kategorien.
    """
    print("🚀 Starte Dual-Level Evaluierung...")
    
    if not pd.api.types.is_datetime64_any_dtype(test_df['utc_time']):
        test_df['utc_time'] = pd.to_datetime(test_df['utc_time'])
    test_df['local_time'] = test_df['utc_time'] + pd.to_timedelta(test_df['timezone_offset'], unit='m')
    test_df['hour'] = test_df['local_time'].dt.hour
    
    # Metriken für Spezifisch
    hits_at_1_spec = 0
    hits_at_k_spec = 0
    
    # Metriken für Level 1
    hits_at_1_lvl1 = 0
    hits_at_k_lvl1 = 0
    
    total = len(test_df)
    
    for _, row in test_df.iterrows():
        actual_specific = row['venue_category_name']
        actual_level1 = row['level_1']
        hour = row['hour']
        
        # Modell nach Empfehlungen fragen
        recommendations = model.recommend(hour)
        rec_specific = recommendations['specific']
        rec_level1 = recommendations['level_1']
        
        # --- Spezifische Kategorie prüfen ---
        if rec_specific:
            if actual_specific == rec_specific[0]:
                hits_at_1_spec += 1
            if actual_specific in rec_specific:
                hits_at_k_spec += 1
                
        # --- Level 1 Kategorie prüfen ---
        if rec_level1:
            if actual_level1 == rec_level1[0]:
                hits_at_1_lvl1 += 1
            if actual_level1 in rec_level1:
                hits_at_k_lvl1 += 1
                
    # Ergebnisse berechnen
    acc_1_spec = (hits_at_1_spec / total) * 100
    hit_k_spec = (hits_at_k_spec / total) * 100
    
    acc_1_lvl1 = (hits_at_1_lvl1 / total) * 100
    hit_k_lvl1 = (hits_at_k_lvl1 / total) * 100
    
    # Output formatieren
    print("\n" + "="*50)
    print("📈 EVALUIERUNGS-ERGEBNISSE (Time-Based Baseline)")
    print("="*50)
    print(f"Test-Samples gesamt: {total}\n")
    
    print("🎯 SPEZIFISCHE KATEGORIE (z.B. Sushi Restaurant)")
    print(f"   Accuracy @ 1: {acc_1_spec:.2f}%")
    print(f"   Hit Rate @ K: {hit_k_spec:.2f}%\n")
    
    print("🌍 LEVEL 1 KATEGORIE (z.B. Food)")
    print(f"   Accuracy @ 1: {acc_1_lvl1:.2f}%")
    print(f"   Hit Rate @ K: {hit_k_lvl1:.2f}%")
    print("="*50)
    
    return acc_1_spec, hit_k_spec, acc_1_lvl1, hit_k_lvl1