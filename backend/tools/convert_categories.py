import pandas as pd

def process_categories(input_filepath, output_filepath):
    print(f"Lese Kategorie-Daten von {input_filepath}...")
    
    # 1. Daten einlesen
    df = pd.read_csv(
        input_filepath,
        sep='\t',
        header=0, 
        names=['category_id', 'category_name', 'hierarchy'],
        engine='python'
    )
    
    print("Säubere Daten...")
    df['category_id'] = df['category_id'].str.strip()
    df['category_name'] = df['category_name'].str.strip()
    df['hierarchy'] = df['hierarchy'].str.strip()
    
    print("Erstelle dynamische Hierarchie-Ebenen...")
    # 2. String am '>' zerschneiden
    # Die Regex \s*>\s* entfernt direkt die Leerzeichen vor und nach dem '>'
    # expand=True transformiert das Ergebnis automatisch in separate Spalten!
    levels_df = df['hierarchy'].str.split(r'\s*>\s*', expand=True)
    
    # 3. Den neuen Spalten saubere Namen geben (level_1, level_2, ...)
    # levels_df.shape[1] gibt uns automatisch die maximale Tiefe (z.B. 4)
    level_columns = [f'level_{i+1}' for i in range(levels_df.shape[1])]
    levels_df.columns = level_columns
    
    # 4. Die neuen Spalten an das bestehende DataFrame anhängen
    df = pd.concat([df, levels_df], axis=1)
    
    # Optional: Den alten, unhandlichen Hierarchy-String entfernen, 
    # da wir die Infos jetzt sauber getrennt haben
    df = df.drop(columns=['hierarchy'])
    
    print(f"Gefundene maximale Hierarchie-Tiefe: {levels_df.shape[1]}")
    
    # 5. Als Parquet speichern
    print(f"Speichere als Parquet in {output_filepath}...")
    df.to_parquet(output_filepath, engine='pyarrow', index=False)
    
    print("Fertig! 🎉")
    print("\nEin kleiner Einblick in die Struktur:")
    print(df.head(4).to_string())

if __name__ == "__main__":
    INPUT_FILE = ".\\dataset_tsmc2014\\categories.txt" 
    OUTPUT_FILE = ".\\dataset_tsmc2014\\foursquare_categories.parquet"
    
    process_categories(INPUT_FILE, OUTPUT_FILE)