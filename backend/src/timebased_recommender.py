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