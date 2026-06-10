import pandas as pd

class TimeBasedBaselineRecommender:
    def __init__(self, top_k=5, use_user_clusters=False):
        """
        top_k: Anzahl der Empfehlungen, die das Modell pro Kategorie-Level zurückgeben soll.
        use_user_clusters: Ob User-Cluster-basierte Recommendations verwendet werden sollen
        """
        self.top_k = top_k
        self.use_user_clusters = use_user_clusters
        
        # Global (ohne Cluster)
        self.popular_specific_by_hour = {}
        self.global_popular_specific = []
        self.popular_level1_by_hour = {}
        self.global_popular_level1 = []
        
        # Mit User-Clusters (cluster -> hour -> categories)
        self.popular_specific_by_hour_and_cluster = {}  # {cluster: {hour: [cats]}}
        self.popular_level1_by_hour_and_cluster = {}    # {cluster: {hour: [level1s]}}
        self.global_popular_specific_by_cluster = {}    # {cluster: [cats]}
        self.global_popular_level1_by_cluster = {}      # {cluster: [level1s]}

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wandelt UTC-Zeit in lokale Zeit um und extrahiert die Stunde."""
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['utc_time']):
            df['utc_time'] = pd.to_datetime(df['utc_time'])
            
        df['local_time'] = df['utc_time'] + pd.to_timedelta(df['timezone_offset'], unit='m')
        df['hour'] = df['local_time'].dt.hour
        return df

    def fit(self, df: pd.DataFrame, user_cluster_df: pd.DataFrame = None):
        """
        Trainiert das Modell für spezifische Kategorien UND Level 1 Kategorien.
        
        Args:
            df: DataFrame mit Checkins
            user_cluster_df: Optional - DataFrame mit 'user_id' und 'cluster' Spalten
        """
        print("Trainiere Time-Based Baseline Modell...")
        df = self._prepare_data(df)
        
        # ===== GLOBAL TRAINING (ohne Cluster) =====
        # 1. Global beliebteste Kategorien als Fallback berechnen
        self.global_popular_specific = df['venue_category_name'].value_counts().head(self.top_k).index.tolist()
        self.global_popular_level1 = df['level_1'].value_counts().head(self.top_k).index.tolist()
        
        # 2. Beliebteste Kategorien pro Stunde berechnen
        for hour, group in df.groupby('hour'):
            self.popular_specific_by_hour[hour] = group['venue_category_name'].value_counts().head(self.top_k).index.tolist()
            self.popular_level1_by_hour[hour] = group['level_1'].value_counts().head(self.top_k).index.tolist()
        
        # ===== CLUSTER-BASIERTES TRAINING (optional) =====
        if user_cluster_df is not None and self.use_user_clusters:
            print("Trainiere Cluster-basierte Empfehlungen...")
            
            # Merge Cluster-Zuweisungen in den DataFrame
            df = df.merge(user_cluster_df[['user_id', 'cluster']], on='user_id', how='left')
            
            # Finde alle einzigartigen Cluster
            clusters = df['cluster'].unique()
            
            for cluster in sorted(clusters):
                cluster_data = df[df['cluster'] == cluster]
                
                # Global beliebteste pro Cluster
                self.global_popular_specific_by_cluster[cluster] = cluster_data['venue_category_name'].value_counts().head(self.top_k).index.tolist()
                self.global_popular_level1_by_cluster[cluster] = cluster_data['level_1'].value_counts().head(self.top_k).index.tolist()
                
                # Pro Stunde pro Cluster
                self.popular_specific_by_hour_and_cluster[cluster] = {}
                self.popular_level1_by_hour_and_cluster[cluster] = {}
                
                for hour, hour_group in cluster_data.groupby('hour'):
                    self.popular_specific_by_hour_and_cluster[cluster][hour] = hour_group['venue_category_name'].value_counts().head(self.top_k).index.tolist()
                    self.popular_level1_by_hour_and_cluster[cluster][hour] = hour_group['level_1'].value_counts().head(self.top_k).index.tolist()
            
            print(f"✅ Cluster-basiertes Training für {len(clusters)} Cluster abgeschlossen.")
        else:
            print("⚠️ Kein User-Cluster DataFrame bereitgestellt oder use_user_clusters=False")
            
        print("✅ Training abgeschlossen.")

    def recommend(self, hour: int, user_cluster: int = None) -> dict:
        """
        Gibt ein Dictionary mit Vorhersagen für beide Granularitätsstufen zurück.
        
        Args:
            hour: Stunde (0-23)
            user_cluster: Optional - User-Cluster (0-9). Falls angegeben, werden cluster-spezifische Recs zurückgegeben.
            
        Returns:
            Dictionary mit 'specific' und 'level_1' Recommendations
        """
        if user_cluster is not None and self.use_user_clusters:
            # Cluster-spezifische Empfehlungen
            cluster_hour_dict = self.popular_specific_by_hour_and_cluster.get(user_cluster, {})
            rec_specific = cluster_hour_dict.get(hour, self.global_popular_specific_by_cluster.get(user_cluster, self.global_popular_specific))
            
            cluster_level1_dict = self.popular_level1_by_hour_and_cluster.get(user_cluster, {})
            rec_level1 = cluster_level1_dict.get(hour, self.global_popular_level1_by_cluster.get(user_cluster, self.global_popular_level1))
        else:
            # Global Empfehlungen
            rec_specific = self.popular_specific_by_hour.get(hour, self.global_popular_specific)
            rec_level1 = self.popular_level1_by_hour.get(hour, self.global_popular_level1)
        
        return {
            'specific': rec_specific,
            'level_1': rec_level1
        }
    
    def print_hourly_recommendations(self, user_cluster: int = None):
        """
        Gibt die stündlichen Empfehlungen (Specific) zweispaltig im Terminal aus.
        
        Args:
            user_cluster: Optional - Wenn angegeben, zeigt cluster-spezifische Empfehlungen
        """
        if user_cluster is not None and self.use_user_clusters:
            hour_dict = self.popular_specific_by_hour_and_cluster.get(user_cluster, {})
            title = f"Top-Kategorien im Tagesverlauf (Cluster {user_cluster}):"
        else:
            hour_dict = self.popular_specific_by_hour
            title = "Top-Kategorien im Tagesverlauf (Global):"
        
        if not hour_dict:
            print("Das Modell wurde noch nicht trainiert (Dictionary ist leer).")
            return

        print(f"\n{title}")
        print("-" * 75)
        
        # 12 Zeilen für 24 Stunden (Spalte 1: 0-11 Uhr, Spalte 2: 12-23 Uhr)
        for i in range(12):
            # Linke Spalte (00:00 bis 11:00)
            hour_left = i
            cat_left = hour_dict.get(hour_left, [])
            cat_str_left = cat_left[0] if cat_left else "-"
            str_left = f"{hour_left:02d}:00  ➔  {cat_str_left}"
            
            # Rechte Spalte (12:00 bis 23:00)
            hour_right = i + 12
            cat_right = hour_dict.get(hour_right, [])
            cat_str_right = cat_right[0] if cat_right else "-"
            str_right = f"{hour_right:02d}:00  ➔  {cat_str_right}"
            
            # Linke Spalte wird mit ljust() auf 35 Zeichen aufgefüllt, 
            # damit die rechte Spalte immer exakt an der gleichen Stelle beginnt.
            print(f"{str_left.ljust(35)} |   {str_right}")
            
        print("-" * 75)