import pandas as pd
import numpy as np
import math
import json
import os
import requests


def split_data_chronologically(df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2):
    """
    Splittet die Daten chronologisch PRO USER in Train, Val und Test, 
    um Data Leakage (Blick in die Zukunft) zu vermeiden.
    """
    print("🔄 Splitting data chronologically per user...")
    
    # Sicherstellen, dass nach Zeit sortiert ist
    if not pd.api.types.is_datetime64_any_dtype(df['utc_time']):
        df['utc_time'] = pd.to_datetime(df['utc_time'])
    
    df = df.sort_values(by=['user_id', 'utc_time']).reset_index(drop=True)
    
    # Berechne die relative Position jedes Check-ins für jeden User
    user_counts = df.groupby('user_id').size()
    ranks = df.groupby('user_id').cumcount()
    totals = df['user_id'].map(user_counts)
    
    train_mask = ranks < (totals * train_ratio)
    val_mask = (ranks >= (totals * train_ratio)) & (ranks < (totals * (train_ratio + val_ratio)))
    test_mask = ranks >= (totals * (train_ratio + val_ratio))
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"📊 Split result: Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})")
    return train_df, val_df, test_df

def evaluate_recommender(model, test_df: pd.DataFrame, user_features_df: pd.DataFrame = None, silent=False):
    """
    Bewertet das Modell basierend auf der Hit Rate für spezifische und Level-1-Kategorien.
    
    Args:
        model: TimeBasedBaselineRecommender Modell
        test_df: Test-Daten
        user_features_df: Optional - DataFrame mit user_id und cluster Spalten für Cluster-basierte Eval
        silent: Boolean - Unterdrückt Konsolenausgaben
    """
    if not silent:
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
        if use_clusters:
            if pd.notna(row.get('cluster')):
                user_cluster = int(row['cluster'])
                recommendations_cluster = model.recommend(hour, user_cluster=user_cluster)
            else:
                # Fallback auf Global Recommendation für unbekannte User (Cold Start)
                recommendations_cluster = model.recommend(hour)
                
            rec_specific_cluster = recommendations_cluster['specific']
            rec_level1_cluster = recommendations_cluster['level_1']
            
            # Spezifische Kategorie (Cluster oder Global-Fallback)
            if rec_specific_cluster:
                if actual_specific == rec_specific_cluster[0]:
                    hits_at_1_spec_cluster += 1
                if actual_specific in rec_specific_cluster:
                    hits_at_k_spec_cluster += 1
            
            # Level 1 Kategorie (Cluster oder Global-Fallback)
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
    if not silent:
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
            print("🎯 CLUSTER-BASIERTE RECOMMENDATIONS (Blended System Performance)")
            print(f"   (Evaluierte Samples gesamt: {cluster_total} / {total})")
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
    
    # Metriken in ein sauberes Dictionary packen
    metrics = {
        "global_acc_1_spec": acc_1_spec_global,
        "global_hit_k_spec": hit_k_spec_global,
        "global_acc_1_lvl1": acc_1_lvl1_global,
        "global_hit_k_lvl1": hit_k_lvl1_global,
    }
    
    if use_clusters and cluster_total > 0:
        metrics.update({ "cluster_acc_1_spec": acc_1_spec_cluster, "cluster_hit_k_spec": hit_k_spec_cluster, "cluster_acc_1_lvl1": acc_1_lvl1_cluster, "cluster_hit_k_lvl1": hit_k_lvl1_cluster })
        
    return metrics

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Berechnet die Distanz zwischen zwei Koordinaten in Metern, vectorized für Arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000
    return c * r

def evaluate_poi_retrieval(trained_dict, test_df, user_features_df, venues_df, max_users=200, distance_threshold_meters=20, top_places_list=[1, 2, 3, 5, 10], alpha_cluster_weight=0.5, silent=False):
    """
    Führt eine "Leave-One-Out" Evaluation direkt gegen die lokale venue-Datenbank durch.
    """
    if not silent:
        print(f"\n--- Starte POI Retrieval Evaluation (Max Users: {max_users or 'ALL'}) ---")
    
    if venues_df is None or venues_df.empty:
        print("⚠️ Keine Venues Datenbank gefunden. Überspringe POI Evaluation.")
        return

    # 1. Test-Set: Nur den absolut letzten Check-in pro User nehmen ("Leave-One-Out")
    if 'utc_time' in test_df.columns:
        test_samples = test_df.sort_values('utc_time').groupby('user_id').tail(1)
    else:
        test_samples = test_df.groupby('user_id').tail(1)
        
    if max_users and max_users < len(test_samples):
        test_samples = test_samples.sample(n=max_users, random_state=42)

    # 3. Metriken vorbereiten
    cat_hit = 0
    poi_hits = {n: 0 for n in top_places_list}
    
    total = len(test_samples)
    total_evaluated = 0
    user_clusters = dict(zip(user_features_df['user_id'], user_features_df['cluster']))

    if not silent:
        print("Frage lokale Venues Datenbank ab...")
    
    for idx, (_, row) in enumerate(test_samples.iterrows()):
        if not silent and idx % 10 == 0 and idx > 0:
            print(f"  > Evaluiere User {idx}/{total}...")
            
        user_id = row['user_id']
        actual_cat = row['venue_category_name']  # Spezifische Kategorie für das Modell
        actual_lat = float(row['latitude'])
        actual_lng = float(row['longitude'])
        actual_venue_id = row.get('venue_id')
        
        # Lokale Stunde berechnen
        if 'utc_time' in row and pd.notnull(row['utc_time']):
            try:
                utc_time = pd.to_datetime(row['utc_time'])
                if utc_time.tzinfo is not None:
                    utc_time = utc_time.tz_convert(None)
                tz_offset = int(row.get('timezone_offset', 0))
                local_time = utc_time + pd.Timedelta(minutes=tz_offset)
                hour = local_time.hour
            except:
                hour = 12
        else:
            hour = 12
            
        cluster = user_clusters.get(user_id, None)
        
        # Top 3 Kategorien aus dem trainierten Dictionary holen
        cluster_data = trained_dict.get(cluster, {})
        pred_cats = cluster_data.get(hour, []) if isinstance(cluster_data, dict) else []
        
        # Metrik 1: Kategorie Hit
        if actual_cat in pred_cats:
            cat_hit += 1
            
        # Lokale Venues abfragen
        places_per_cat = {n: [] for n in top_places_list}
        max_n = max(top_places_list)
        
        for cat in pred_cats:
            # 1. Filtern nach Kategorie (query)
            filtered_df = venues_df[venues_df['category'].str.lower() == cat.lower()].copy()
            if filtered_df.empty:
                continue
                
            # 2. Distanz berechnen
            distances = haversine_vectorized(actual_lat, actual_lng, filtered_df['latitude'].values, filtered_df['longitude'].values)
            filtered_df['distance'] = distances
            
            # 3. Filtern nach Radius (1500m) und sortieren
            nearby_df = filtered_df[filtered_df['distance'] <= 1500].copy()
            if actual_venue_id is not None and 'venue_id' in nearby_df.columns:
                mask = nearby_df['venue_id'] == actual_venue_id
                
                if mask.any():
                    # Popularität global um 1 reduzieren, Minimum 1
                    nearby_df.loc[mask, 'checkin_count'] = (nearby_df.loc[mask, 'checkin_count'] - 1).clip(lower=1)
                    
                    # Popularität für diesen spezifischen Cluster um 1 reduzieren, falls vorhanden (Minimum 0)
                    cluster_col = f'cluster_{int(cluster)}_count' if cluster is not None else None
                    if cluster_col and cluster_col in nearby_df.columns:
                        nearby_df.loc[mask, cluster_col] = (nearby_df.loc[mask, cluster_col] - 1).clip(lower=0)

            # Sortieren: Blended Score (Min-Max Normalisierung)
            cluster_col = f'cluster_{int(cluster)}_count' if cluster is not None else None
            
            if cluster_col and cluster_col in nearby_df.columns:
                # Normalisierung
                max_global = nearby_df['checkin_count'].max()
                max_cluster = nearby_df[cluster_col].max()
                
                norm_global = nearby_df['checkin_count'] / max_global if max_global > 0 else 0
                norm_cluster = nearby_df[cluster_col] / max_cluster if max_cluster > 0 else 0
                
                # Blended Score berechnen
                nearby_df['blended_score'] = (alpha_cluster_weight * norm_cluster) + ((1.0 - alpha_cluster_weight) * norm_global)
                
                # Sortieren
                nearby_df = nearby_df.sort_values(['blended_score', 'checkin_count'], ascending=[False, False])
            else:
                nearby_df = nearby_df.sort_values('checkin_count', ascending=False)
            
            top_n_venues = nearby_df.head(max_n)
            
            if len(top_n_venues) > 0:
                for n in top_places_list:
                    for _, place in top_n_venues.head(n).iterrows():
                        places_per_cat[n].append(place)

        # Metrik 2 & 3: Place Hit über Distanz (innerhalb Threshold, hier 20m)
        def check_match(places_list):
            for p in places_list:
                if p['distance'] <= distance_threshold_meters:
                    return True
            return False
            
        for n in top_places_list:
            if check_match(places_per_cat[n]):
                poi_hits[n] += 1
            
        total_evaluated += 1

    print("\n" + "="*50)
    print(f"📊 LOCAL POI RETRIEVAL RESULTS (N={total_evaluated} evaluiert von {total})")
    print("="*50)
    
    # Metriken in ein sauberes Dictionary packen
    cat_hit_rate = (cat_hit/total_evaluated*100) if total_evaluated > 0 else 0
    results = {"local_cat_hit_rate": cat_hit_rate}
    
    if not silent:
        print("\n" + "="*50)
        print(f"📊 LOCAL POI RETRIEVAL RESULTS (N={total_evaluated} evaluiert von {total})")
        print("="*50)
        print(f"🎯 Specific Category Hit Rate:            {cat_hit_rate:.1f} %")
    
    for n in top_places_list:
        hit_rate = (poi_hits[n]/total_evaluated*100) if total_evaluated > 0 else 0
        results[f"local_poi_hit_rate_{n}_per_cat"] = hit_rate
        if not silent:
            print(f"📍 POI Hit Rate ({n} Places pro Cat):        {hit_rate:.1f} %")
            
    if not silent:
        print("="*50 + "\n")
    return results