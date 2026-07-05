import os
import json
from dotenv import load_dotenv
import pickle  # Neu importieren für das Speichern
import pandas as pd
import numpy as np
from src.evaluator import evaluate_recommender, split_data_chronologically, evaluate_poi_retrieval
from src.timebased_recommender import TimeBasedBaselineRecommender
from src.decentralized_recommender import DecentralizedRecommender
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
EPSILON_LDP = 1.0          # epsilon: Privacy budget for LDP (lower = more private, more noise)
# =============================================================================

def save_model_dictionary(model_dict, filepath):
    """Hilfsmethode zum Speichern des trainierten Dictionaries"""
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"=== Model successfully saved to '{filepath}' 💾 ===")

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
        print(f"\n\n--- K-Optimization Experiment with k={k} ({num_runs} runs) ---")
        
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
            
    print("\n✅ K-Optimization Experiment completed! Results saved to 'statistics/'.")

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
        print(f"\n\n--- Top-K Experiment with top_k={top_k} ({num_runs} runs) ---")
        
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
            
    print("\n✅ Top-K Experiment completed! Results saved to 'statistics/'.")

def main():
    # -- Preprocessing --------------------------------------------------------
    print("Start Preprocessing Pipeline...")    
    checkin_df = pipeline(CHECKINS_FILE, CATEGORIES_FILE)
    print("=== Preprocessing done successfully! 🎉 ===\n")
    
    # -- Logging initialisieren -----------------------------------------------
    unique_cats = checkin_df['venue_category_name'].dropna().unique()
    print(f"Writing log file with {len(unique_cats)} remaining categories...")
    with open("log.md", "w", encoding="utf-8") as f:
        f.write("# Experiment Log\n\n")
        f.write("## Preprocessing Statistics\n")
        f.write(f"Anzahl spezifischer Kategorien nach dem Filtern: **{len(unique_cats)}**\n\n")
        f.write("<details>\n<summary>Klick hier für alle Kategorien</summary>\n\n")
        for cat in sorted(unique_cats):
            f.write(f"- {cat}\n")
        f.write("\n</details>\n\n")
    
    # Gefilterte Check-ins speichern, damit die API beim Validieren/Testen nicht über herausgefilterte Orte stolpert
    print(f"Saving cleaned check-ins to '{PREPROCESSED_CHECKINS_FILE}'...")
    checkin_df.to_parquet(PREPROCESSED_CHECKINS_FILE, index=False)
    
    # -- Visualisierung -------------------------------------------------------
    print("Start visualisation...")
    test_user_id = checkin_df['user_id'].iloc[0]
    plot_user_trajectory(df=checkin_df, user_id=test_user_id, output_html="nyc_user_map.html")

    # -- User Clustering (K-Means)  ------------------------------------------
    print("\n" + "="*70)
    print("START USER CLUSTERING MODEL TRAINING")
    print("="*70 + "\n")
    
    # 1. User Partitioning Modell trainieren (Zentral!)
    user_clustering_model = UserPartitioningRecommender(k=U_USER_CLUSTERS, top_categories=9)
    user_features_df = user_clustering_model.fit(checkin_df)
    
    # 1b. User Partitioning Modell trainieren (Dezentral!)
    dec_clustering_model = UserPartitioningRecommender(k=U_USER_CLUSTERS, top_categories=9)
    dec_user_features_df = dec_clustering_model.fit_decentralized(checkin_df, gossip_rounds=15, epochs=5)
    
    # 2. User-Features als Parquet speichern
    print(f"\nSaving user features to '{USER_FEATURES_PATH}'...")
    user_features_df.to_parquet(USER_FEATURES_PATH, index=False)
    print("✅ User features saved!")
    
    # 3. K-Means Modell speichern
    print(f"Saving K-Means model to '{KMEANS_MODEL_PATH}'...")
    save_model_dictionary(user_clustering_model, KMEANS_MODEL_PATH)
    print("✅ K-Means model saved!")
    
    # 4. Centroid-Analyse ausgeben
    print("\nCluster-Centroids calculated successfully.")
    centroids_df = user_clustering_model.get_cluster_centroids()
    
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

    # 5. Skip verbose hourly recommendations
    print("Hourly recommendations calculated.")
    
    # 6. Das Dictionary aus dem Modell extrahieren und speichern
    trained_dict = model.popular_specific_by_hour_and_cluster 
    save_model_dictionary(trained_dict, MODEL_OUTPUT_PATH)

    # -- Local Venue Leave-One-Out Evaluation ------------------------------
    print("\n" + "="*70)
    print("START LOCAL POI RETRIEVAL EVALUATION")
    print("="*70 + "\n")
    
    # Datenbank dynamisch mit User-Clustern neu aufbauen
    print("Rebuilding venue database with cluster visits...")
    venues_df = create_venue_db(checkin_df, user_features_df, output_path=VENUES_FILE)
    
    if venues_df is None:
        print("⚠️ Error creating venue database.")
    
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
    print("\n✅ Standard evaluation saved to 'statistics/standard_evaluation.json'.")

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
    
    # Decentralized: one list per gossip_rounds value
    gossip_rounds_list = [5, 10, 20]
    dec_cat_dict = {r: [] for r in gossip_rounds_list}  # {rounds: [cat_hits]}
    dec_poi_dict = {r: [] for r in gossip_rounds_list}  # {rounds: [poi_hits]}
    
    for frac in fractions:
        print(f"\n▶ Trainiere Modell mit {int(frac*100)}% der verfügbaren Trainingsdaten...")
        
        if frac == 0.0:
            print("  Überspringe Training für 0% (Hardcoded auf 0 Hit Rate).")
            cat_hits_history.append(0)
            poi_hits_history.append(0)
            global_cat_hits_history.append(0)
            global_poi_hits_history.append(0)
            for r in gossip_rounds_list:
                dec_cat_dict[r].append(0)
                dec_poi_dict[r].append(0)
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
        
        # --- DECENTRALIZED (Gossip + LDP) mit verschiedenen Gossip Rounds ---
        for rounds in gossip_rounds_list:
            dec_model = DecentralizedRecommender(top_k=K_PREDICTED_CATS, use_user_clusters=True, epsilon=EPSILON_LDP, gossip_rounds=rounds)
            dec_model.fit(train_subset, user_cluster_df=dec_user_features_df)
            
            dec_rec_metrics = evaluate_recommender(dec_model, test_df, user_features_df=dec_user_features_df, silent=True)
            dec_poi_metrics = evaluate_poi_retrieval(
                dec_model.popular_specific_by_hour_and_cluster,
                test_df, dec_user_features_df, venues_df,
                max_users=None, distance_threshold_meters=20, top_places_list=[N_PLACES_PER_CAT],
                alpha_cluster_weight=ALPHA_CLUSTER_WEIGHT, silent=True
            )
            
            dec_cat_dict[rounds].append(dec_rec_metrics.get('cluster_hit_k_spec', dec_rec_metrics.get('global_hit_k_spec', 0)))
            dec_poi_dict[rounds].append(dec_poi_metrics.get(f'local_poi_hit_rate_{N_PLACES_PER_CAT}_per_cat', 0) if dec_poi_metrics else 0)
        
        # Abspeichern der Cluster-Werte
        cat_hits_history.append(rec_metrics_sub.get('cluster_hit_k_spec', rec_metrics_sub.get('global_hit_k_spec', 0)))
        poi_hits_history.append(poi_metrics_sub.get(f'local_poi_hit_rate_{N_PLACES_PER_CAT}_per_cat', 0) if poi_metrics_sub else 0)
        
        # Abspeichern der Global-Werte
        global_cat_hits_history.append(rec_metrics_sub.get('global_hit_k_spec', 0))
        global_poi_hits_history.append(poi_metrics_global.get(f'local_poi_hit_rate_{N_PLACES_PER_CAT}_per_cat', 0) if poi_metrics_global else 0)
        
    plot_learning_curve(fractions, cat_hits_history, poi_hits_history, global_cat_hits_history, global_poi_hits_history, dec_cat_dict, dec_poi_dict, output_path="statistics/learning_curve.png")

    # -- K OPTIMIZATION EXPERIMENT --------------------------------------------
    # run_k_optimization_experiment(checkin_df, venues_df)
    
    # # -- TOP K EXPERIMENT -----------------------------------------------------
    # run_top_k_experiment(checkin_df, venues_df)
    
    # # -- PLOT HYPERPARAMETERS -------------------------------------------------
    # from tools.plot_hyperparameters import load_data, plot_k_optimization, plot_top_k_optimization, plot_top_places_tradeoff
    # print("\nGenerating hyperparameter plots...")
    # plots_dir = "statistics/plots"
    # os.makedirs(plots_dir, exist_ok=True)
    
    # k_df = load_data("statistics/k_optimization_results.json")
    # if k_df is not None:
    #     plot_k_optimization(k_df, plots_dir)
    
    # top_k_df = load_data("statistics/top_k_results.json")
    # if top_k_df is not None:
    #     plot_top_k_optimization(top_k_df, plots_dir)
    #     plot_top_places_tradeoff(top_k_df, plots_dir)
    
    print("\n🎉 TRAINING COMPLETE!")
    print("="*70)
    
    
if __name__ == "__main__":
    main()