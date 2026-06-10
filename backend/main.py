import pickle  # Neu importieren für das Speichern
from src.evaluator import evaluate_recommender, split_data_chronologically
from src.timebased_recommender import TimeBasedBaselineRecommender
from src.user_clustering import UserPartitioningRecommender
from src.preprocess_data import pipeline
from src.visualize_trajectory import plot_user_trajectory

CHECKINS_FILE = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\data\\foursquare_checkins_nyc.parquet"
CATEGORIES_FILE = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\data\\foursquare_categories.parquet"
MODEL_OUTPUT_PATH = "trained_model.pkl"  # Pfad für die Modelldatei
USER_FEATURES_PATH = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\data\\user_partitioning.parquet"  # User-Features Parquet
KMEANS_MODEL_PATH = "user_clustering_model.pkl"  # K-Means Modell

def save_model_dictionary(model_dict, filepath):
    """Hilfsmethode zum Speichern des trainierten Dictionaries"""
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"=== Modell erfolgreich unter '{filepath}' gespeichert! 💾 ===")

def main():
    # -- Preprocessing --------------------------------------------------------
    print("Start Preprocessing Pipeline...")    
    checkin_df = pipeline(CHECKINS_FILE, CATEGORIES_FILE)
    print("=== Preprocessing done successfully! 🎉 ===\n")
    
    # -- Visualisierung -------------------------------------------------------
    print("Start visualisation...")
    test_user_id = checkin_df['user_id'].iloc[0]
    plot_user_trajectory(df=checkin_df, user_id=test_user_id, output_html="nyc_user_map.html")

    # -- User Clustering (K-Means) - ZUERST trainieren ------------------------------------------
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

    # -- Time-Based Recommendations mit User-Clustering ------------------------------------------
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
    evaluate_recommender(model, test_df, user_features_df=user_features_df)

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

    # 7. export categories fpr visu
    import pandas as pd, json
    df = pd.read_parquet('../data/foursquare_categories.parquet')
    mapping = df.set_index('category_name')['level_1'].to_dict()
    with open('../data/category_level1_map.json', 'w') as f:
        json.dump(mapping, f)
    
    print("\n🎉 TRAINING COMPLETE!")
    print("="*70)
    
    
if __name__ == "__main__":
    main()