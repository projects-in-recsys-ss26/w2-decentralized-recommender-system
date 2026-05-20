from src.preprocess_data import pipeline
from src.visualize_trajectory import plot_user_trajectory

CHECKINS_FILE = "./data/foursquare_checkins_tky.parquet"
CATEGORIES_FILE = "./data/foursquare_categories.parquet"

def main():
    print("Hello from backend!")

    # -- Preprocessing --------------------------------------------------------
    print("Starte Preprocessing Pipeline...")    
    checkin_df = pipeline(CHECKINS_FILE, CATEGORIES_FILE)
    print("=== Preprocessing erfolgreich abgeschlossen! 🎉 ===")
    
    # -- Visualisierung -------------------------------------------------------
    print("\nStarte Visualisierung...")
    
    # Wir holen uns dynamisch einfach die erste User_ID, die im Datensatz vorkommt
    test_user_id = checkin_df['user_id'].iloc[0]
    
    # Karte generieren
    plot_user_trajectory(
        df=checkin_df, 
        user_id=test_user_id, 
        output_html="tokyo_user_map.html"
    )

if __name__ == "__main__":
    main()