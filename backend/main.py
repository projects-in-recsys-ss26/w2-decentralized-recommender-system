import pickle  # Neu importieren für das Speichern
from src.evaluator import evaluate_recommender, split_data_chronologically
from src.timebased_recommender import TimeBasedBaselineRecommender
from src.preprocess_data import pipeline
from src.visualize_trajectory import plot_user_trajectory

CHECKINS_FILE = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\backend\\data\\foursquare_checkins_nyc.parquet"
CATEGORIES_FILE = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\backend\\data\\foursquare_categories.parquet"
MODEL_OUTPUT_PATH = "trained_model.pkl"  # Pfad für die Modelldatei

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

    # -- Recommendations-------------------------------------------------------
    # 1. Daten splitten
    train_df, val_df, test_df = split_data_chronologically(checkin_df, train_ratio=0.6, val_ratio=0.2)

    # 2. Modell initialisieren
    model = TimeBasedBaselineRecommender(top_k=3)

    # 3. Modell auf TRAIN-Daten trainieren
    model.fit(train_df)

    # 4. Modell auf TEST-Daten evaluieren
    evaluate_recommender(model, test_df)

    # 5. Stündliche Empfehlungen ausgeben
    model.print_hourly_recommendations()
    
    # NEU: 6. Das Dictionary aus dem Modell extrahieren und speichern
    # Ersetze 'dein_internes_dict' mit dem echten Attributnamen deiner Klasse
    trained_dict = model.popular_specific_by_hour 
    save_model_dictionary(trained_dict, MODEL_OUTPUT_PATH)
    
    
if __name__ == "__main__":
    main()