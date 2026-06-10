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

def evaluate_recommender(model, test_df: pd.DataFrame, user_features_df: pd.DataFrame = None):
    """
    Bewertet das Modell basierend auf der Hit Rate für spezifische und Level-1-Kategorien.
    
    Args:
        model: TimeBasedBaselineRecommender Modell
        test_df: Test-Daten
        user_features_df: Optional - DataFrame mit user_id und cluster Spalten für Cluster-basierte Eval
    """
    print("🚀 Starte Dual-Level Evaluierung...")
    
    if not pd.api.types.is_datetime64_any_dtype(test_df['utc_time']):
        test_df['utc_time'] = pd.to_datetime(test_df['utc_time'])
    test_df['local_time'] = test_df['utc_time'] + pd.to_timedelta(test_df['timezone_offset'], unit='m')
    test_df['hour'] = test_df['local_time'].dt.hour
    
    # Merge User-Cluster Falls vorhanden
    if user_features_df is not None:
        test_df = test_df.merge(user_features_df[['user_id', 'cluster']], on='user_id', how='left')
        use_clusters = True
    else:
        use_clusters = False
    
    # ===== METRIKEN FÜR GLOBAL RECOMMENDATIONS =====
    hits_at_1_spec_global = 0
    hits_at_k_spec_global = 0
    hits_at_1_lvl1_global = 0
    hits_at_k_lvl1_global = 0
    
    # ===== METRIKEN FÜR CLUSTER-BASIERTE RECOMMENDATIONS =====
    hits_at_1_spec_cluster = 0
    hits_at_k_spec_cluster = 0
    hits_at_1_lvl1_cluster = 0
    hits_at_k_lvl1_cluster = 0
    
    total = len(test_df)
    cluster_total = 0  # Zählt nur Samples mit validem Cluster
    
    for _, row in test_df.iterrows():
        actual_specific = row['venue_category_name']
        actual_level1 = row['level_1']
        hour = row['hour']
        
        # ===== GLOBAL RECOMMENDATIONS =====
        recommendations_global = model.recommend(hour)
        rec_specific_global = recommendations_global['specific']
        rec_level1_global = recommendations_global['level_1']
        
        # Spezifische Kategorie (Global)
        if rec_specific_global:
            if actual_specific == rec_specific_global[0]:
                hits_at_1_spec_global += 1
            if actual_specific in rec_specific_global:
                hits_at_k_spec_global += 1
        
        # Level 1 Kategorie (Global)
        if rec_level1_global:
            if actual_level1 == rec_level1_global[0]:
                hits_at_1_lvl1_global += 1
            if actual_level1 in rec_level1_global:
                hits_at_k_lvl1_global += 1
        
        # ===== CLUSTER-BASIERTE RECOMMENDATIONS =====
        if use_clusters and pd.notna(row.get('cluster')):
            user_cluster = int(row['cluster'])
            recommendations_cluster = model.recommend(hour, user_cluster=user_cluster)
            rec_specific_cluster = recommendations_cluster['specific']
            rec_level1_cluster = recommendations_cluster['level_1']
            
            # Spezifische Kategorie (Cluster)
            if rec_specific_cluster:
                if actual_specific == rec_specific_cluster[0]:
                    hits_at_1_spec_cluster += 1
                if actual_specific in rec_specific_cluster:
                    hits_at_k_spec_cluster += 1
            
            # Level 1 Kategorie (Cluster)
            if rec_level1_cluster:
                if actual_level1 == rec_level1_cluster[0]:
                    hits_at_1_lvl1_cluster += 1
                if actual_level1 in rec_level1_cluster:
                    hits_at_k_lvl1_cluster += 1
            
            cluster_total += 1
    
    # ===== ERGEBNISSE BERECHNEN =====
    acc_1_spec_global = (hits_at_1_spec_global / total) * 100
    hit_k_spec_global = (hits_at_k_spec_global / total) * 100
    acc_1_lvl1_global = (hits_at_1_lvl1_global / total) * 100
    hit_k_lvl1_global = (hits_at_k_lvl1_global / total) * 100
    
    if use_clusters and cluster_total > 0:
        acc_1_spec_cluster = (hits_at_1_spec_cluster / cluster_total) * 100
        hit_k_spec_cluster = (hits_at_k_spec_cluster / cluster_total) * 100
        acc_1_lvl1_cluster = (hits_at_1_lvl1_cluster / cluster_total) * 100
        hit_k_lvl1_cluster = (hits_at_k_lvl1_cluster / cluster_total) * 100
    
    # ===== OUTPUT FORMATIEREN =====
    print("\n" + "="*70)
    print("📈 EVALUIERUNGS-ERGEBNISSE (Time-Based Baseline)")
    print("="*70)
    print(f"Test-Samples gesamt: {total}\n")
    
    print("🌍 GLOBAL RECOMMENDATIONS")
    print("-" * 70)
    print("🎯 SPEZIFISCHE KATEGORIE (z.B. Sushi Restaurant)")
    print(f"   Accuracy @ 1: {acc_1_spec_global:.2f}%")
    print(f"   Hit Rate @ K: {hit_k_spec_global:.2f}%\n")
    print("🌍 LEVEL 1 KATEGORIE (z.B. Dining and Drinking)")
    print(f"   Accuracy @ 1: {acc_1_lvl1_global:.2f}%")
    print(f"   Hit Rate @ K: {hit_k_lvl1_global:.2f}%")
    
    if use_clusters and cluster_total > 0:
        print("\n" + "-" * 70)
        print("🎯 CLUSTER-BASIERTE RECOMMENDATIONS")
        print(f"   (Samples mit Cluster: {cluster_total} / {total})")
        print("-" * 70)
        print("🎯 SPEZIFISCHE KATEGORIE (z.B. Sushi Restaurant)")
        print(f"   Accuracy @ 1: {acc_1_spec_cluster:.2f}%")
        print(f"   Hit Rate @ K: {hit_k_spec_cluster:.2f}%\n")
        print("🌍 LEVEL 1 KATEGORIE (z.B. Dining and Drinking)")
        print(f"   Accuracy @ 1: {acc_1_lvl1_cluster:.2f}%")
        print(f"   Hit Rate @ K: {hit_k_lvl1_cluster:.2f}%")
        
        # Improvement berechnen
        print("\n" + "-" * 70)
        print("📊 IMPROVEMENT (Cluster vs Global)")
        print("-" * 70)
        spec_1_improvement = acc_1_spec_cluster - acc_1_spec_global
        spec_k_improvement = hit_k_spec_cluster - hit_k_spec_global
        lvl1_1_improvement = acc_1_lvl1_cluster - acc_1_lvl1_global
        lvl1_k_improvement = hit_k_lvl1_cluster - hit_k_lvl1_global
        
        print(f"Spezifische @ 1: {spec_1_improvement:+.2f}% {'📈' if spec_1_improvement > 0 else '📉'}")
        print(f"Spezifische @ K: {spec_k_improvement:+.2f}% {'📈' if spec_k_improvement > 0 else '📉'}")
        print(f"Level 1 @ 1:     {lvl1_1_improvement:+.2f}% {'📈' if lvl1_1_improvement > 0 else '📉'}")
        print(f"Level 1 @ K:     {lvl1_k_improvement:+.2f}% {'📈' if lvl1_k_improvement > 0 else '📉'}")
    
    print("="*70)
    
    if use_clusters and cluster_total > 0:
        return (acc_1_spec_global, hit_k_spec_global, acc_1_lvl1_global, hit_k_lvl1_global,
                acc_1_spec_cluster, hit_k_spec_cluster, acc_1_lvl1_cluster, hit_k_lvl1_cluster)
    else:
        return acc_1_spec_global, hit_k_spec_global, acc_1_lvl1_global, hit_k_lvl1_global