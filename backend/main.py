from src.preprocess_data import pipeline


CHECKINS_FILE = "./data/foursquare_checkins_tky.parquet"        # Deine Tokyo oder NYC Datei
CATEGORIES_FILE = "./data/foursquare_categories.parquet"    # Die hierarchische Kategorie-Datei

def main():
    print("Hello from backend!")

    # -- Preprocessing --------------------------------------------------------
    print("Starte Preprocessing Pipeline...")    

    checkin_df = pipeline(CHECKINS_FILE, CATEGORIES_FILE)
    
    print("=== Preprocessing erfolgreich abgeschlossen! 🎉 ===")
    print(checkin_df.head(3))


if __name__ == "__main__":
    main()
