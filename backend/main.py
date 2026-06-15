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

CHECKINS_FILE = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\data\\foursquare_checkins_nyc.parquet"
CATEGORIES_FILE = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\data\\foursquare_categories.parquet"
MODEL_OUTPUT_PATH = "trained_model.pkl"  # Pfad für die Modelldatei
USER_FEATURES_PATH = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\data\\user_partitioning.parquet"  # User-Features Parquet
PREPROCESSED_CHECKINS_FILE = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\data\\preprocessed_checkins_nyc.parquet"
KMEANS_MODEL_PATH = "user_clustering_model.pkl"  # K-Means Modell

def save_model_dictionary(model_dict, filepath):
    """Hilfsmethode zum Speichern des trainierten Dictionaries"""
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"=== Modell erfolgreich unter '{filepath}' gespeichert! 💾 ===")

def run_scaling_experiment(checkin_df, api_key, offline_mode=True):
    """
    Trainiert und evaluiert das Modell iterativ mit steigender Anzahl an Usern,
    um den Lernfortschritt des Recommenders (Scaling Laws) zu messen.
    """
    print("\n" + "="*70)
    print("🚀 STARTING SCALING EXPERIMENT (Increasing User Counts)")
    print("="*70)
    
    os.makedirs("statistics", exist_ok=True)
    
    all_users = checkin_df['user_id'].unique()
    user_counts = [20, 50, 100, 150, 200, 300, 400, 600, len(all_users)]
    
    results = []
    
    for count in user_counts:
        # Dynamisches k: 10% der Nutzerzahl, aber mindestens 2 Cluster
        dynamic_k = max(2, count // 10)
        
        print(f"\n\n--- Evaluierung mit {count} Usern (k={dynamic_k}) ---")
        # 1. Sample users (fixierter random state für Reproduzierbarkeit)
        np.random.seed(42)
        sampled_users = np.random.choice(all_users, size=min(count, len(all_users)), replace=False)
        df_subset = checkin_df[checkin_df['user_id'].isin(sampled_users)].copy()
        
        # 2. Pipeline für dieses Subset durchlaufen
        user_clustering_model = UserPartitioningRecommender(k=dynamic_k, top_categories=9)
        user_features_subset = user_clustering_model.fit(df_subset)
        
        train_sub, val_sub, test_sub = split_data_chronologically(df_subset, train_ratio=0.6, val_ratio=0.2)
        
        model_sub = TimeBasedBaselineRecommender(top_k=3, use_user_clusters=True)
        model_sub.fit(train_sub, user_cluster_df=user_features_subset)
        
        rec_metrics = evaluate_recommender(model_sub, test_sub, user_features_df=user_features_subset)
        
        # 3. Foursquare Evaluation (max_users=None, da das Cache API Calls verhindert!)
        poi_metrics = evaluate_poi_retrieval(model_sub.popular_specific_by_hour_and_cluster, test_sub, user_features_subset, api_key, max_users=None, distance_threshold_meters=100, offline_mode=offline_mode)
        
        # 4. Speichern der Subset-Ergebnisse
        combined_metrics = {
            "num_users": count,
            "k_clusters": dynamic_k,
            "total_checkins": len(df_subset),
            **rec_metrics,
            **(poi_metrics if poi_metrics else {})
        }
        results.append(combined_metrics)
        
        pd.DataFrame(results).to_csv("statistics/scaling_results.csv", index=False)
        with open("statistics/scaling_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
    print("\n✅ Scaling Experiment abgeschlossen! Ergebnisse in 'statistics/' gespeichert.")

def main():
    # -- Preprocessing --------------------------------------------------------
    print("Start Preprocessing Pipeline...")    
    checkin_df = pipeline(CHECKINS_FILE, CATEGORIES_FILE)
    print("=== Preprocessing done successfully! 🎉 ===\n")
    
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
    user_clustering_model = UserPartitioningRecommender(k=10, top_categories=9)
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

    # -- Time-Based Recommendations mit User-Clustering -----------------------
    print("\n" + "="*70)
    print("START TIME-BASED RECOMMENDER MODEL TRAINING")
    print("="*70 + "\n")
    
    # 1. Daten splitten
    train_df, val_df, test_df = split_data_chronologically(checkin_df, train_ratio=0.6, val_ratio=0.2)

    # 2. Modell initialisieren (mit User-Cluster Support)
    model = TimeBasedBaselineRecommender(top_k=3, use_user_clusters=True)

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

    # -- Foursquare API Leave-One-Out Evaluation ------------------------------
    print("\n" + "="*70)
    print("START FOURSQUARE POI RETRIEVAL EVALUATION")
    print("="*70 + "\n")
    load_dotenv()
    api_key = os.getenv("FOURSQUARE_API_KEY")
    
    poi_metrics = evaluate_poi_retrieval(
        trained_dict=trained_dict, 
        test_df=test_df, 
        user_features_df=user_features_df, 
        api_key=api_key, 
        max_users=20,  # Begrenzt auf zufällige 20 User zum Schonen deines API Limits
        offline_mode=True # Geht sicher, dass auch der Einzel-Test keine API-Credits mehr frisst!
    )
    
    # Standard-Ergebnisse in den Statistics-Ordner schreiben
    os.makedirs("statistics", exist_ok=True)
    standard_results = {**rec_metrics, **(poi_metrics if poi_metrics else {})}
    with open("statistics/standard_evaluation.json", "w") as f:
        json.dump(standard_results, f, indent=4)
    print(f"\n✅ Standard-Evaluierung unter 'statistics/standard_evaluation.json' gespeichert.")

    # -- SCALING EXPERIMENT ---------------------------------------------------
    run_scaling_experiment(checkin_df, api_key, offline_mode=True)
    
    print("\n🎉 TRAINING COMPLETE!")
    print("="*70)
    
    
if __name__ == "__main__":
    main()