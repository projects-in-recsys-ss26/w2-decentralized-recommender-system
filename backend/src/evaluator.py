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

def haversine_distance(lat1, lon1, lat2, lon2):
    """Berechnet die Distanz zwischen zwei Koordinaten in Metern."""
    R = 6371000  # Erdradius in Metern
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def evaluate_poi_retrieval(trained_dict, test_df, user_features_df, api_key, max_users=200, distance_threshold_meters=100, offline_mode=False):
    """
    Führt eine "Leave-One-Out" Foursquare Evaluation durch.
    """
    print(f"\n--- Starte POI Retrieval Evaluation (Max Users: {max_users or 'ALL'}) ---")
    
    if not api_key:
        print("⚠️ Kein Foursquare API Key gefunden. Überspringe POI Evaluation.")
        return

    # 1. Test-Set: Nur den absolut letzten Check-in pro User nehmen ("Leave-One-Out")
    if 'utc_time' in test_df.columns:
        test_samples = test_df.sort_values('utc_time').groupby('user_id').tail(1)
    else:
        test_samples = test_df.groupby('user_id').tail(1)
        
    # Falls wir das API Limit schonen wollen, begrenzen wir die Nutzeranzahl
    if max_users and max_users < len(test_samples):
        test_samples = test_samples.sample(n=max_users, random_state=42)

    # 2. Lokalen Cache laden (verhindert doppelte API Aufrufe und spart Foursquare Kontingent)
    cache_file = "foursquare_api_cache.json"
    api_cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            try:
                api_cache = json.load(f)
            except Exception:
                api_cache = {}  # Falls die Datei durch den letzten Absturz beschädigt wurde

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-Places-Api-Version": "2025-06-17",
    }

    # 3. Metriken vorbereiten
    cat_hit_3 = 0
    poi_hit_3 = 0  # Hatten wir einen Treffer im Top 1 Place der 3 Kategorien? (Insgesamt 3 Places)
    poi_hit_9 = 0  # Hatten wir einen Treffer in den Top 3 Places der 3 Kategorien? (Insgesamt 9 Places)
    
    total = len(test_samples)
    total_evaluated = 0  # Zählt, wie viele User wir VOR einem eventuellen API-Limit geschafft haben
    user_clusters = dict(zip(user_features_df['user_id'], user_features_df['cluster']))

    print("Frage Foursquare API ab (Nutze Caching, wo möglich)...")
    
    api_limit_reached = False
    api_cache_updated = False
    
    for idx, (_, row) in enumerate(test_samples.iterrows()):
        if idx % 10 == 0 and idx > 0:
            print(f"  > Evaluiere User {idx}/{total}...")
            if api_cache_updated:
                # Zwischenspeichern des Caches nur, wenn es Änderungen gab
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(api_cache, f)
                api_cache_updated = False
            
        user_id = row['user_id']
        actual_cat = row['venue_category_name']  # Spezifische Kategorie für das Modell
        actual_lat = float(row['latitude'])
        actual_lng = float(row['longitude'])
        
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
            cat_hit_3 += 1
            
        # Foursquare Places abfragen
        places_top1 = []
        places_top3 = []
        
        for cat in pred_cats:
            if api_limit_reached:
                break
                
            # Limit = 3, damit wir sowohl die 3er- als auch die 9er-Metrik testen können!
            cache_key = f"{actual_lat}_{actual_lng}_{cat}_3_POPULARITY"
            
            if cache_key in api_cache:
                results = api_cache[cache_key]
            else:
                if offline_mode:
                    # Im Offline-Modus stoppen wir die Evaluierung für neue POIs sofort
                    api_limit_reached = True
                    results = []
                else:
                    params = {
                        "query": cat,
                        "ll": f"{actual_lat},{actual_lng}",
                        "radius": 1500,
                        "limit": 3,
                        "sort": "POPULARITY",
                        "fields": "fsq_place_id,name,latitude,longitude"
                    }
                    try:
                        resp = requests.get("https://places-api.foursquare.com/places/search", headers=headers, params=params)
                        if resp.status_code == 200:
                            results = resp.json().get('results', [])
                            api_cache[cache_key] = results
                            api_cache_updated = True
                        elif resp.status_code in [403, 429]:
                            print(f"\n❌ FOURSQUARE API LIMIT EXCEEDED (Status {resp.status_code})! Stoppe Live-Anfragen.")
                            api_limit_reached = True
                            results = []
                        else:
                            results = []
                    except Exception as e:
                        results = []
                    
            if len(results) > 0:
                places_top1.append(results[0])  # Nur das beste Restaurant dieser Kategorie
                places_top3.extend(results)     # Alle 3 Top-Restaurants dieser Kategorie

        if api_limit_reached and len(places_top1) == 0:
            break  # Bricht die User-Schleife ab, da wir ohne API keine POI-Evaluation mehr machen können

        # Metrik 2 & 3: Place Hit über Haversine Distanz (innerhalb von 100m)
        def check_match(places_list):
            for p in places_list:
                if p.get('latitude') and p.get('longitude'):
                    dist = haversine_distance(actual_lat, actual_lng, p['latitude'], p['longitude'])
                    if dist <= distance_threshold_meters:
                        return True
            return False
            
        if check_match(places_top1): poi_hit_3 += 1
        if check_match(places_top3): poi_hit_9 += 1
            
        total_evaluated += 1

    # Abschließendes Speichern des Caches (verhindert Datenverlust)
    if api_cache_updated:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(api_cache, f)

    print("\n" + "="*50)
    print(f"📊 FOURSQUARE POI RETRIEVAL RESULTS (N={total_evaluated} evaluiert von {total})")
    print("="*50)
    
    # Metriken in ein sauberes Dictionary packen
    cat_hit_rate = (cat_hit_3/total_evaluated*100) if total_evaluated > 0 else 0
    poi_hit_3_rate = (poi_hit_3/total_evaluated*100) if total_evaluated > 0 else 0
    poi_hit_9_rate = (poi_hit_9/total_evaluated*100) if total_evaluated > 0 else 0
    
    print(f"🎯 Specific Category Hit Rate @ 3:      {cat_hit_rate:.1f} %")
    print(f"📍 POI Hit Rate @ 3 (1 Place pro Cat):  {poi_hit_3_rate:.1f} %")
    print(f"📍 POI Hit Rate @ 9 (3 Places pro Cat): {poi_hit_9_rate:.1f} %")
    print("="*50 + "\n")
    
    return {
        "fsq_cat_hit_rate_at_3": cat_hit_rate,
        "fsq_poi_hit_rate_at_3": poi_hit_3_rate,
        "fsq_poi_hit_rate_at_9": poi_hit_9_rate
    }