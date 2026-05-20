import os
import pandas as pd

def read_data(filepath: str) -> pd.DataFrame:
    """Lädt eine Parquet-Datei effizient in den RAM."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Die Datei {filepath} wurde nicht gefunden!")
        
    print(f"Lade Daten aus: {filepath}...")
    df = pd.read_parquet(filepath)
    print(f"-> Erfolgreich geladen. Form: {df.shape} (Zeilen, Spalten)")
    return df


def merge_categories(checkins_df: pd.DataFrame, categories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Verknüpft die Check-ins mit den hierarchischen Kategorien.
    Nutzt einen Left-Join, um keine Check-ins zu verlieren.
    """
    print("Verknüpfe Check-ins mit Kategorie-Hierarchie...")
    
    # Da die Spalten in den beiden Dateien leicht unterschiedlich heißen:
    # checkins_df: 'venue_category_id'
    # categories_df: 'category_id'
    merged_df = pd.merge(
        checkins_df,
        categories_df,
        left_on='venue_category_id',
        right_on='category_id',
        how='left'  # Left Join sorgt dafür, dass alle Check-ins behalten werden
    )
    
    # Bereinigung: Die doppelte ID-Spalte 'category_id' entfernen
    merged_df = merged_df.drop(columns=['category_id'])
    
    # Optionale Qualitätskontrolle: Gab es IDs, die nicht im Mapping waren?
    missing_mask = merged_df['level_1'].isna()
    missing_count = missing_mask.sum()
    
    if missing_count > 0:
        print(f"⚠️ Warnung: Für {missing_count} Check-ins wurde keine Kategorie-Hierarchie gefunden.")
        print("Hier sind die betroffenen Einträge:")
        
        # Wir filtern die fehlerhaften Zeilen heraus und zeigen nur die relevanten Spalten
        missing_rows = merged_df[missing_mask][
            ['user_id', 'venue_id', 'venue_category_id', 'venue_category_name']
        ]
        
        # to_string() sorgt dafür, dass Pandas nicht abkürzt, sondern alles schön druckt
        print(missing_rows.to_string())
    else:
        print("✅ Alle Kategorien erfolgreich zugeordnet!")
        
    return merged_df

def fix_missing_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ersetzt veraltete oder fehlende Foursquare Category IDs durch gültige Alternativen.
    """
    print("Korrigiere bekannte fehlende Kategorie-IDs...")
    
    # Mapping von alter ID -> neuer ID
    replacements = {
        "4e51a0c0bd41d3446defbb2e": "4bf58dd8d48988d12d951735" # Ferry -> Boat and Ferry
    }
    
    # replace() sucht in der Spalte nach den alten Keys und ersetzt sie durch die neuen Values
    df['venue_category_id'] = df['venue_category_id'].replace(replacements)
    
    return df


def pipeline(checkins_path: str, categories_path: str):
    """Die Haupt-Pipeline, die alles steuert."""
    print("=== Starte Daten-Preprocessing ===")
    
    # 1. Daten laden
    df_checkins = read_data(checkins_path)
    df_categories = read_data(categories_path)

    # 2. Fehlende Kategorien korrigieren (z.B. die veraltete Ferry-ID)
    df_checkins = fix_missing_categories(df_checkins)
    
    # 3. Mergen
    df_final = merge_categories(df_checkins, df_categories)

    # 4. Return 
    return df_final