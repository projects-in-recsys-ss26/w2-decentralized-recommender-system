import os
import json
import pandas as pd
import numpy as np
import sys

# Ensure the parent directory is in the path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluator import evaluate_recommender, split_data_chronologically, evaluate_poi_retrieval
from src.timebased_recommender import TimeBasedBaselineRecommender
from src.user_clustering import UserPartitioningRecommender

CHECKINS_FILE = "../../../data/preprocessed_checkins_nyc.parquet"
VENUES_FILE = "../../../data/venues.parquet"

def run_user_scaling_experiment(num_runs=1):
    print("\n" + "="*70)
    print("🚀 STARTING USER SCALING EXPERIMENT")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkins_path = os.path.normpath(os.path.join(base_dir, CHECKINS_FILE))
    venues_path = os.path.normpath(os.path.join(base_dir, VENUES_FILE))
    stats_dir = os.path.normpath(os.path.join(base_dir, "..", "statistics"))
    
    os.makedirs(stats_dir, exist_ok=True)
    
    print(f"Lese Check-ins aus {checkins_path}...")
    checkin_df = pd.read_parquet(checkins_path)
    
    print(f"Lese Venues aus {venues_path}...")
    venues_df = pd.read_parquet(venues_path)
    
    # We want to evaluate on the test split of the ENTIRE user base.
    global_train, _, global_test = split_data_chronologically(checkin_df, train_ratio=0.6, val_ratio=0.2)
    
    fractions = [0.05, 0.10, 0.20, 0.25, 0.40, 0.50, 0.70, 0.80, 1.00]
    seeds = [42 + i for i in range(num_runs)]
    results = []
    
    unique_users = global_train['user_id'].unique()
    total_users = len(unique_users)
    
    k_clusters = 20
    top_k_cats = 5
    top_places_list = [2]
    
    for frac in fractions:
        print(f"\n\n--- User Scaling Experiment mit fraction={frac*100:.0f}% der Nutzer ---")
        
        run_metrics_list = []
        for run_idx, seed in enumerate(seeds):
            print(f"\n  ▶ Run {run_idx + 1}/{num_runs} (Seed {seed})")
            np.random.seed(seed)
            
            num_users_to_sample = max(1, int(total_users * frac))
            sampled_users = np.random.choice(unique_users, size=num_users_to_sample, replace=False)
            
            train_subset = global_train[global_train['user_id'].isin(sampled_users)].copy()
            print(f"    Trainings-Nutzer: {num_users_to_sample}/{total_users}")
            print(f"    Trainings-Checkins: {len(train_subset)}/{len(global_train)}")
            
            if len(train_subset) < k_clusters:
                print(f"⚠️ Zu wenige Daten für k={k_clusters}. Überspringe.")
                continue
                
            # Train User Clustering
            user_clustering_model = UserPartitioningRecommender(k=k_clusters, top_categories=9)
            user_features_sub = user_clustering_model.fit(train_subset)
            
            # Train Recommender
            model_sub = TimeBasedBaselineRecommender(top_k=top_k_cats, use_user_clusters=True)
            model_sub.fit(train_subset, user_cluster_df=user_features_sub)
            
            print("    Extrahiere Features für alle User im global_test für die Evaluation...")
            # Berechne Features für ALLE User im Testset mithilfe des trainierten Extractors
            all_user_features = user_clustering_model._extract_user_features(global_test, inference=True)
            
            X_all = all_user_features[user_clustering_model.feature_columns].values
            X_all_scaled = user_clustering_model.scaler.transform(X_all)
            all_user_features['cluster'] = user_clustering_model.kmeans.predict(X_all_scaled)
            
            print("    Evaluiere Recommender auf dem GESAMTEN Testset...")
            rec_metrics = evaluate_recommender(model_sub, global_test, user_features_df=all_user_features)
            
            print("    Evaluiere POI Retrieval auf dem GESAMTEN Testset...")
            poi_metrics = evaluate_poi_retrieval(
                trained_dict=model_sub.popular_specific_by_hour_and_cluster, 
                test_df=global_test, 
                user_features_df=all_user_features, 
                venues_df=venues_df, 
                max_users=None,  # Evaluierung für alle
                distance_threshold_meters=20, 
                top_places_list=top_places_list
            )
            
            combined = {**rec_metrics, **(poi_metrics if poi_metrics else {})}
            run_metrics_list.append(combined)
            
        if run_metrics_list:
            avg_metrics = {key: np.mean([run.get(key, 0) for run in run_metrics_list]) for key in run_metrics_list[0].keys()}
            results.append({"fraction": frac, "num_train_users": num_users_to_sample, **avg_metrics})
            
            pd.DataFrame(results).to_csv(os.path.join(stats_dir, "user_scaling_results.csv"), index=False)
            with open(os.path.join(stats_dir, "user_scaling_results.json"), "w") as f:
                json.dump(results, f, indent=4)
            
    print("\n✅ User Scaling Experiment abgeschlossen! Ergebnisse in 'statistics/' gespeichert.")

if __name__ == "__main__":
    run_user_scaling_experiment()
