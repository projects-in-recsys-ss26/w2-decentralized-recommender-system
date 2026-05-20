import pandas as pd

def convert_tsv_to_parquet(input_filepath, output_filepath):
    print(f"Lese Daten von {input_filepath}...")
    
    # Die Spaltennamen laut Datensatz-Beschreibung
    column_names = [
        "user_id",
        "venue_id",
        "venue_category_id",
        "venue_category_name",
        "latitude",
        "longitude",
        "timezone_offset",
        "utc_time"
    ]
    
    # 1. Datei einlesen
    df = pd.read_csv(
        input_filepath,
        sep='\t',
        header=None,
        names=column_names,
        engine='python',
        encoding='latin-1'  # <-- Das hier hinzufügen!
    )
    
    print("Optimiere Datentypen...")
    
    # 2. Datetime-Konvertierung (WICHTIG für Zeitreihen-Vorhersagen)
    # Das Format ist z.B. "Tue Apr 03 18:00:09 +0000 2012"
    # pd.to_datetime ist clever genug, dieses Standard-Format meist automatisch zu parsen, 
    # aber wir setzen format='%a %b %d %H:%M:%S %z %Y' für maximale Geschwindigkeit und Präzision.
    df['utc_time'] = pd.to_datetime(df['utc_time'], format='%a %b %d %H:%M:%S %z %Y')
    
    # 3. Datentypen optimieren (spart RAM)
    df['user_id'] = df['user_id'].astype('int32')
    df['latitude'] = df['latitude'].astype('float32')
    df['longitude'] = df['longitude'].astype('float32')
    df['timezone_offset'] = df['timezone_offset'].astype('int16')
    
    # 4. Als Parquet speichern
    print(f"Speichere als Parquet in {output_filepath}...")
    df.to_parquet(output_filepath, engine='pyarrow', index=False)
    
    print("Fertig! 🎉")
    print(f"Anzahl der geladenen Zeilen: {len(df)}")
    print("Ein kleiner Einblick in die Daten:")
    print(df.head())

if __name__ == "__main__":
    # HIER DEINE DATEINAMEN EINTRAGEN
    # Zum Beispiel: 'dataset_TSMC2014_NYC.txt'
    INPUT_FILE = "./dataset_tsmc2014/dataset_TSMC2014_NYC.txt" 
    OUTPUT_FILE = "./dataset_tsmc2014/foursquare_checkins_nyc.parquet"
    
    convert_tsv_to_parquet(INPUT_FILE, OUTPUT_FILE)