import os
import json
from dotenv import load_dotenv
import pickle  # Neu importieren für das Speichern
import pandas as pd
import numpy as np
from src.evaluator import evaluate_recommender, split_data_chronologically, evaluate_poi_retrieval
from src.timebased_recommender import TimeBasedBaselineRecommender
from src.user_clustering import UserPartitioningRecommender
from src.preprocess_data import pipeline
from src.visualize_trajectory import plot_user_trajectory
from tools.create_venue_db import create_venue_db

CHECKINS_FILE = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\data\\foursquare_checkins_nyc.parquet"
CATEGORIES_FILE = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\data\\foursquare_categories.parquet"
MODEL_OUTPUT_PATH = "trained_model.pkl"  # Pfad für die Modelldatei
USER_FEATURES_PATH = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\data\\user_partitioning.parquet"  # User-Features Parquet
PREPROCESSED_CHECKINS_FILE = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\data\\preprocessed_checkins_nyc.parquet"
VENUES_FILE = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\data\\venues.parquet"
KMEANS_MODEL_PATH = "user_clustering_model.pkl"  # K-Means Modell

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
N_PLACES_PER_CAT = 2   # n: Anzahl der empfohlenen Places pro vorhergesagter Kategorie
K_PREDICTED_CATS = 5   # k: Anzahl der vorhergesagten Kategorien
U_USER_CLUSTERS = 20   # u: Anzahl der User-Cluster
ALPHA_CLUSTER_WEIGHT = 0.5 # alpha: Gewichtung für Cluster-Popularität vs. Globale Popularität (0.0 bis 1.0)
# =============================================================================

def save_model_dictionary(model_dict, filepath):
    """Hilfsmethode zum Speichern des trainierten Dictionaries"""
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"=== Modell erfolgreich unter '{filepath}' gespeichert! 💾 ===")

def run_k_optimization_experiment(checkin_df, venues_df, num_runs=1):
    """
    Evaluiert das Modell mit verschiedenen Werten für k (Anzahl der User-Cluster).
    """
    print("\n" + "="*70)
    print("🔍 STARTING K-OPTIMIZATION EXPERIMENT")
    print("="*70)
    
    os.makedirs("statistics", exist_ok=True)
    global_train, _, global_test = split_data_chronologically(checkin_df, train_ratio=0.6, val_ratio=0.2)
    
    seeds = [42 + i for i in range(num_runs)]
    results = []
    k_values = [2, 3, 5, 8, 10, 15, 20, 30, 50]
    
    for k in k_values:
        print(f"\n\n--- K-Optimization Experiment mit k={k} über {num_runs} Runs ---")
        
        run_metrics_list = []
        for run_idx, seed in enumerate(seeds):
            print(f"\n  ▶ Run {run_idx + 1}/{num_runs} (Seed {seed})")
            np.random.seed(seed)
            
            user_clustering_model = UserPartitioningRecommender(k=k, top_categories=9)
            user_features_sub = user_clustering_model.fit(global_train)
            
            # top_k=3 ist fixiert für dieses Experiment
            model_sub = TimeBasedBaselineRecommender(top_k=3, use_user_clusters=True)
            model_sub.fit(global_train, user_cluster_df=user_features_sub)
            
            rec_metrics = evaluate_recommender(model_sub, global_test, user_features_df=user_features_sub)
            poi_metrics = evaluate_poi_retrieval(model_sub.popular_specific_by_hour_and_cluster, global_test, user_features_sub, venues_df, max_users=None, distance_threshold_meters=20, top_places_list=[1, 2, 3, 5, 10])
            
            combined = {**rec_metrics, **(poi_metrics if poi_metrics else {})}
            run_metrics_list.append(combined)
            
        avg_metrics = {key: np.mean([run.get(key, 0) for run in run_metrics_list]) for key in run_metrics_list[0].keys()}
        results.append({"k": k, **avg_metrics})
        
        pd.DataFrame(results).to_csv("statistics/k_optimization_results.csv", index=False)
        with open("statistics/k_optimization_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
    print("\n✅ K-Optimization Experiment abgeschlossen! Ergebnisse in 'statistics/' gespeichert.")

def run_top_k_experiment(checkin_df, venues_df, num_runs=1):
    """
    Evaluiert das Modell mit verschiedenen top_k Werten für die Vorhersage.
    """
    print("\n" + "="*70)
    print("📈 STARTING TOP-K EXPERIMENT")
    print("="*70)
    
    os.makedirs("statistics", exist_ok=True)
    global_train, _, global_test = split_data_chronologically(checkin_df, train_ratio=0.6, val_ratio=0.2)
    
    seeds = [42 + i for i in range(num_runs)]
    results = []
    top_k_values = [1, 2, 3, 4, 5, 6, 8, 10]
    
    for top_k in top_k_values:
        print(f"\n\n--- Top-K Experiment mit top_k={top_k} über {num_runs} Runs ---")
        
        run_metrics_list = []
        for run_idx, seed in enumerate(seeds):
            print(f"\n  ▶ Run {run_idx + 1}/{num_runs} (Seed {seed})")
            np.random.seed(seed)
            
            user_clustering_model = UserPartitioningRecommender(k=20, top_categories=9)
            user_features_sub = user_clustering_model.fit(global_train)
            
            model_sub = TimeBasedBaselineRecommender(top_k=top_k, use_user_clusters=True)
            model_sub.fit(global_train, user_cluster_df=user_features_sub)
            
            rec_metrics = evaluate_recommender(model_sub, global_test, user_features_df=user_features_sub)
            poi_metrics = evaluate_poi_retrieval(model_sub.popular_specific_by_hour_and_cluster, global_test, user_features_sub, venues_df, max_users=None, distance_threshold_meters=20, top_places_list=[1, 2, 3, 5, 10])
            
            combined = {**rec_metrics, **(poi_metrics if poi_metrics else {})}
            run_metrics_list.append(combined)
            
        avg_metrics = {key: np.mean([run.get(key, 0) for run in run_metrics_list]) for key in run_metrics_list[0].keys()}
        results.append({"top_k": top_k, **avg_metrics})
        
        pd.DataFrame(results).to_csv("statistics/top_k_results.csv", index=False)
        with open("statistics/top_k_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
    print("\n✅ Top-K Experiment abgeschlossen! Ergebnisse in 'statistics/' gespeichert.")

def main():
    # -- Preprocessing --------------------------------------------------------
    print("Start Preprocessing Pipeline...")    
    checkin_df = pipeline(CHECKINS_FILE, CATEGORIES_FILE)
    print("=== Preprocessing done successfully! 🎉 ===\n")
    
    # -- Logging initialisieren -----------------------------------------------
    unique_cats = checkin_df['venue_category_name'].dropna().unique()
    print(f"Schreibe Log-Datei mit {len(unique_cats)} verbleibenden Kategorien...")
    with open("log.md", "w", encoding="utf-8") as f:
        f.write("# Experiment Log\n\n")
        f.write("## Preprocessing Statistics\n")
        f.write(f"Anzahl spezifischer Kategorien nach dem Filtern: **{len(unique_cats)}**\n\n")
        f.write("<details>\n<summary>Klick hier für alle Kategorien</summary>\n\n")
        for cat in sorted(unique_cats):
            f.write(f"- {cat}\n")
        f.write("\n</details>\n\n")
    
    # Gefilterte Check-ins speichern, damit die API beim Validieren/Testen nicht über herausgefilterte Orte stolpert
    print(f"Speichere bereinigte Check-ins unter '{PREPROCESSED_CHECKINS_FILE}'...")
    checkin_df.to_parquet(PREPROCESSED_CHECKINS_FILE, index=False)
    
    # -- Visualisierung -------------------------------------------------------
    print("Start visualisation...")
    test_user_id = checkin_df['user_id'].iloc[0]
    plot_user_trajectory(df=checkin_df, user_id=test_user_id, output_html="nyc_user_map.html")

    # -- User Clustering (K-Means)  ------------------------------------------
    print("\n" + "="*70)
    print("START USER CLUSTERING MODEL TRAINING")
    print("="*70 + "\n")
    
    # 1. User Partitioning Modell trainieren
    user_clustering_model = UserPartitioningRecommender(k=U_USER_CLUSTERS, top_categories=9)
    user_features_df = user_clustering_model.fit(checkin_df)
    
    # 2. User-Features als Parquet speichern
    print(f"\nSpeichere User-Features unter '{USER_FEATURES_PATH}'...")
    user_features_df.to_parquet(USER_FEATURES_PATH, index=False)
    print(f"✅ User-Features gespeichert!")
    
    # 3. K-Means Modell speichern
    print(f"Speichere K-Means Modell unter '{KMEANS_MODEL_PATH}'...")
    save_model_dictionary(user_clustering_model, KMEANS_MODEL_PATH)
    print(f"✅ K-Means Modell gespeichert!")
    
    # 4. Centroid-Analyse ausgeben
    print("\nCluster-Centroids (durchschnittliche Kategorien-Verteilung pro Cluster):")
    print("-" * 70)
    centroids_df = user_clustering_model.get_cluster_centroids()
    print(centroids_df.to_string())
    
    # 5. Radar Chart plotten
    from src.visualize_clusters import plot_radar_chart
    os.makedirs("statistics", exist_ok=True)
    plot_radar_chart(centroids_df, output_path="statistics/cluster_radar_chart.png")

    # -- Time-Based Recommendations mit User-Clustering -----------------------
    print("\n" + "="*70)
    print("START TIME-BASED RECOMMENDER MODEL TRAINING")
    print("="*70 + "\n")
    
    # 1. Daten splitten
    train_df, val_df, test_df = split_data_chronologically(checkin_df, train_ratio=0.6, val_ratio=0.2)

    # 2. Modell initialisieren (mit User-Cluster Support)
    model = TimeBasedBaselineRecommender(top_k=K_PREDICTED_CATS, use_user_clusters=True)

    # 3. Modell auf TRAIN-Daten trainieren (mit User-Cluster Features!)
    model.fit(train_df, user_cluster_df=user_features_df)

    # 4. Modell auf TEST-Daten evaluieren (mit Cluster-basiertem Evaluation)
    rec_metrics = evaluate_recommender(model, test_df, user_features_df=user_features_df)

    # 5. Stündliche Empfehlungen ausgeben (Global + Pro Cluster)
    print("\n" + "="*70)
    model.print_hourly_recommendations()
    print("\nBeispiel: Empfehlungen für Cluster 7:")
    model.print_hourly_recommendations(user_cluster=7)
    print("\nBeispiel: Empfehlungen für Cluster 9:")
    model.print_hourly_recommendations(user_cluster=9)
    print("="*70)
    
    # 6. Das Dictionary aus dem Modell extrahieren und speichern
    trained_dict = model.popular_specific_by_hour_and_cluster 
    save_model_dictionary(trained_dict, MODEL_OUTPUT_PATH)

    # -- Local Venue Leave-One-Out Evaluation ------------------------------
    print("\n" + "="*70)
    print("START LOCAL POI RETRIEVAL EVALUATION")
    print("="*70 + "\n")
    
    # Datenbank dynamisch mit User-Clustern neu aufbauen
    print("Erstelle Venue-Datenbank mit Cluster-Visits neu...")
    venues_df = create_venue_db(checkin_df, user_features_df, output_path=VENUES_FILE)
    
    if venues_df is None:
        print("⚠️ Fehler beim Erstellen der Venue-Datenbank.")
    
    poi_metrics = evaluate_poi_retrieval(
        trained_dict=trained_dict, 
        test_df=test_df, 
        user_features_df=user_features_df, 
        venues_df=venues_df, 
        max_users=None,  # Keine Limitierung mehr! Alle Test-User verwenden.
        distance_threshold_meters=20,
        top_places_list=[N_PLACES_PER_CAT],
        alpha_cluster_weight=ALPHA_CLUSTER_WEIGHT
    )
    
    # Standard-Ergebnisse in den Statistics-Ordner schreiben
    os.makedirs("statistics", exist_ok=True)
    standard_results = {**rec_metrics, **(poi_metrics if poi_metrics else {})}
    with open("statistics/standard_evaluation.json", "w") as f:
        json.dump(standard_results, f, indent=4)
    print(f"\n✅ Standard-Evaluierung unter 'statistics/standard_evaluation.json' gespeichert.")

    # -------------------------------------------------------------------------
    # Visualisierungen: Performance Snapshot & Learning Curve
    # -------------------------------------------------------------------------
    from tools.plot_performance import plot_performance_snapshot, plot_learning_curve
    plot_performance_snapshot(output_path="statistics/performance_snapshot.png")
    
    print("\n" + "="*70)
    print("START LEARNING CURVE EXPERIMENT (Training Progression)")
    print("="*70)
    
    fractions = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0]
    cat_hits_history = []
    poi_hits_history = []
    global_cat_hits_history = []
    global_poi_hits_history = []
    
    for frac in fractions:
        print(f"\n▶ Trainiere Modell mit {int(frac*100)}% der verfügbaren Trainingsdaten...")
        
        if frac == 0.0:
            print("  Überspringe Training für 0% (Hardcoded auf 0 Hit Rate).")
            cat_hits_history.append(0)
            poi_hits_history.append(0)
            global_cat_hits_history.append(0)
            global_poi_hits_history.append(0)
            continue
            
        # Subset vom train_df nehmen (chronologisch von Anfang an)
        subset_len = int(len(train_df) * frac)
        train_subset = train_df.iloc[:subset_len]
        
        # Modell neu trainieren
        model_sub = TimeBasedBaselineRecommender(top_k=K_PREDICTED_CATS, use_user_clusters=True)
        model_sub.fit(train_subset, user_cluster_df=user_features_df)
        
        # Evaluieren auf dem GLEICHEN globalen test_df
        rec_metrics_sub = evaluate_recommender(model_sub, test_df, user_features_df=user_features_df, silent=True)
        poi_metrics_sub = evaluate_poi_retrieval(
            model_sub.popular_specific_by_hour_and_cluster, 
            test_df, user_features_df, venues_df, 
            max_users=None, distance_threshold_meters=20, top_places_list=[N_PLACES_PER_CAT], 
            alpha_cluster_weight=ALPHA_CLUSTER_WEIGHT, silent=True
        )
        
        # --- GLOBAL VERGLEICH ---
        # 1. Global POI Dictionary simulieren (jedes Cluster verweist auf Global)
        global_fallback_dict = {c: model_sub.popular_specific_by_hour for c in range(U_USER_CLUSTERS)}
        
        # 2. Global POI Evaluation (ohne Cluster Weight)
        poi_metrics_global = evaluate_poi_retrieval(
            global_fallback_dict, 
            test_df, user_features_df, venues_df, 
            max_users=None, distance_threshold_meters=20, top_places_list=[N_PLACES_PER_CAT], 
            alpha_cluster_weight=0.0, silent=True
        )
        
        # Abspeichern der Cluster-Werte
        cat_hits_history.append(rec_metrics_sub.get('cluster_hit_k_spec', rec_metrics_sub.get('global_hit_k_spec', 0)))
        poi_hits_history.append(poi_metrics_sub.get(f'local_poi_hit_rate_{N_PLACES_PER_CAT}_per_cat', 0) if poi_metrics_sub else 0)
        
        # Abspeichern der Global-Werte
        global_cat_hits_history.append(rec_metrics_sub.get('global_hit_k_spec', 0))
        global_poi_hits_history.append(poi_metrics_global.get(f'local_poi_hit_rate_{N_PLACES_PER_CAT}_per_cat', 0) if poi_metrics_global else 0)
        
    plot_learning_curve(fractions, cat_hits_history, poi_hits_history, global_cat_hits_history, global_poi_hits_history, output_path="statistics/learning_curve.png")

    # -- K OPTIMIZATION EXPERIMENT --------------------------------------------
    # run_k_optimization_experiment(checkin_df, venues_df)
    
    # -- TOP K EXPERIMENT -----------------------------------------------------
    # run_top_k_experiment(checkin_df, venues_df)
    
    print("\n🎉 TRAINING COMPLETE!")
    print("="*70)
    
    
if __name__ == "__main__":
    main()