from src.evaluator import evaluate_recommender, split_data_chronologically
from src.timebased_recommender import TimeBasedBaselineRecommender
from src.preprocess_data import pipeline
from src.visualize_trajectory import plot_user_trajectory

CHECKINS_FILE = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\backend\\data\\foursquare_checkins_nyc.parquet"
CATEGORIES_FILE = "C:\\Users\\Mattes\\Studium\\Semester 10\\Projekt\\masterproject-decentralized-recommender-systems\\backend\\data\\foursquare_categories.parquet"

def main():
    # -- Preprocessing --------------------------------------------------------
    print("Start Preprocessing Pipeline...")    
    checkin_df = pipeline(CHECKINS_FILE, CATEGORIES_FILE)
    print("=== Preprocessing done successfully! 🎉 ===\n")
    
    # -- Visualisierung -------------------------------------------------------
    print("Start visualisation...")
    
    # Wir holen uns dynamisch die erste User_ID, die im Datensatz vorkommt
    test_user_id = checkin_df['user_id'].iloc[0]
    
    # Karte generieren
    plot_user_trajectory(
        df=checkin_df, 
        user_id=test_user_id, 
        output_html="nyc_user_map.html"
    )

    # -- Recommendations-------------------------------------------------------
    # 1. Daten splitten
    train_df, val_df, test_df = split_data_chronologically(checkin_df, train_ratio=0.6, val_ratio=0.2)

    # 2. Modell initialisieren (wir wollen z. B. die Top 5 Kategorien)
    model = TimeBasedBaselineRecommender(top_k=3)

    # 3. Modell auf TRAIN-Daten trainieren
    model.fit(train_df)

    # 4. Modell auf TEST-Daten evaluieren
    # (Validation nutzen wir aktuell noch nicht, die wird erst wichtig, wenn wir Hyperparameter anpassen)
    evaluate_recommender(model, test_df)

    # 5. Stündliche Empfehlungen ausgeben
    model.print_hourly_recommendations()
    
    
if __name__ == "__main__":
    main()