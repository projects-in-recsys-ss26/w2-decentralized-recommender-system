import pandas as pd

class TimeBasedBaselineRecommender:
    def __init__(self, top_k=5):
        """
        top_k: Anzahl der Empfehlungen, die das Modell pro Kategorie-Level zurückgeben soll.
        """
        self.top_k = top_k
        self.popular_specific_by_hour = {}
        self.global_popular_specific = []
        
        self.popular_level1_by_hour = {}
        self.global_popular_level1 = []

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wandelt UTC-Zeit in lokale Zeit um und extrahiert die Stunde."""
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['utc_time']):
            df['utc_time'] = pd.to_datetime(df['utc_time'])
            
        df['local_time'] = df['utc_time'] + pd.to_timedelta(df['timezone_offset'], unit='m')
        df['hour'] = df['local_time'].dt.hour
        return df

    def fit(self, df: pd.DataFrame):
        """Trainiert das Modell für spezifische Kategorien UND Level 1 Kategorien."""
        print("Trainiere Time-Based Baseline Modell (Dual-Level)...")
        df = self._prepare_data(df)
        
        # 1. Global beliebteste Kategorien als Fallback berechnen
        self.global_popular_specific = df['venue_category_name'].value_counts().head(self.top_k).index.tolist()
        self.global_popular_level1 = df['level_1'].value_counts().head(self.top_k).index.tolist()
        
        # 2. Beliebteste Kategorien pro Stunde berechnen
        for hour, group in df.groupby('hour'):
            self.popular_specific_by_hour[hour] = group['venue_category_name'].value_counts().head(self.top_k).index.tolist()
            self.popular_level1_by_hour[hour] = group['level_1'].value_counts().head(self.top_k).index.tolist()
            
        print("✅ Training abgeschlossen.")

    def recommend(self, hour: int) -> dict:
        """Gibt ein Dictionary mit Vorhersagen für beide Granularitätsstufen zurück."""
        rec_specific = self.popular_specific_by_hour.get(hour, self.global_popular_specific)
        rec_level1 = self.popular_level1_by_hour.get(hour, self.global_popular_level1)
        
        return {
            'specific': rec_specific,
            'level_1': rec_level1
        }
    
    def print_hourly_recommendations(self):
        """Gibt die stündlichen Empfehlungen (Specific) zweispaltig im Terminal aus."""
        if not self.popular_specific_by_hour:
            print("Das Modell wurde noch nicht trainiert (Dictionary ist leer).")
            return

        print("\nTop-Kategorien im Tagesverlauf:")
        print("-" * 75)
        
        # 12 Zeilen für 24 Stunden (Spalte 1: 0-11 Uhr, Spalte 2: 12-23 Uhr)
        for i in range(12):
            # Linke Spalte (00:00 bis 11:00)
            hour_left = i
            cat_left = self.popular_specific_by_hour.get(hour_left, [])
            cat_str_left = cat_left[0] if cat_left else "-"
            str_left = f"{hour_left:02d}:00  ➔  {cat_str_left}"
            
            # Rechte Spalte (12:00 bis 23:00)
            hour_right = i + 12
            cat_right = self.popular_specific_by_hour.get(hour_right, [])
            cat_str_right = cat_right[0] if cat_right else "-"
            str_right = f"{hour_right:02d}:00  ➔  {cat_str_right}"
            
            # Linke Spalte wird mit ljust() auf 35 Zeichen aufgefüllt, 
            # damit die rechte Spalte immer exakt an der gleichen Stelle beginnt.
            print(f"{str_left.ljust(35)} |   {str_right}")
            
        print("-" * 75)