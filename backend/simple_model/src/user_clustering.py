import os
# Unterdrücke Warnungen von joblib/loky auf Windows (wmic nicht gefunden) und KMeans Memory Leak
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore", message=".*memory leak.*")
warnings.filterwarnings("ignore", message=".*physical cores.*")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
class UserPartitioningRecommender:
    """
    Trainiert ein K-Means Clustering-Modell basierend auf der Verteilung der level_1 Kategorien pro User.
    Features: Prozentuale Verteilung der Top-9 level_1 Kategorien (normalisiert auf 0-1).
    """
    
    def __init__(self, k: int = 10, top_categories: int = 9):
        """
        k: Anzahl der Cluster (Centroids)
        top_categories: Anzahl der Top-Level-1 Kategorien als Features
        """
        self.k = k
        self.top_categories = top_categories
        self.kmeans = None
        self.scaler = StandardScaler()
        self.top_level_1_categories = []  # Die Top-9 Kategorien
        self.feature_columns = []  # Spalten-Namen der Features
        
    def _extract_user_features(self, df: pd.DataFrame, inference: bool = False) -> pd.DataFrame:
        """
        Extrahiert pro User die prozentuale Verteilung der level_1 Kategorien.
        
        Returns:
            DataFrame mit user_id und prozentuale Features für die Top-9 level_1 Kategorien
        """
        print(f"Extracting user features (Level-1 categories distribution, inference={inference})...")
        
        # Gruppiere nach user_id und zähle level_1 Kategorien
        user_category_counts = df.groupby(['user_id', 'level_1']).size().unstack(fill_value=0)
        
        if not inference:
            # Finde die Top-9 level_1 Kategorien insgesamt
            category_totals = df['level_1'].value_counts()
            self.top_level_1_categories = category_totals.head(self.top_categories).index.tolist()
            
            print(f"  Top-{self.top_categories} level_1 categories: {self.top_level_1_categories}")
        
        # Missing categories with 0
        for cat in self.top_level_1_categories:
            if cat not in user_category_counts.columns:
                user_category_counts[cat] = 0
                
        # Behalte nur die Top-9 Kategorien
        user_category_counts = user_category_counts[self.top_level_1_categories]
        
        # Normalisiere zu Prozenten (0-1)
        user_features = user_category_counts.div(user_category_counts.sum(axis=1), axis=0).fillna(0)
        
        # --- ZUSÄTZLICHE FEATURES (Temporal & Behavioral) ---
        df_temp = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_temp['utc_time']):
            df_temp['utc_time'] = pd.to_datetime(df_temp['utc_time'])
        df_temp['local_time'] = df_temp['utc_time'] + pd.to_timedelta(df_temp['timezone_offset'], unit='m')
        
        # 1. Tageszeit-Präferenzen (Morning, Afternoon, Evening, Night)
        df_temp['hour'] = df_temp['local_time'].dt.hour
        conditions = [
            (df_temp['hour'] >= 6) & (df_temp['hour'] < 12),
            (df_temp['hour'] >= 12) & (df_temp['hour'] < 18),
            (df_temp['hour'] >= 18) & (df_temp['hour'] < 24)
        ]
        df_temp['time_of_day'] = np.select(conditions, ['morning', 'afternoon', 'evening'], default='night')
        tod_counts = df_temp.groupby(['user_id', 'time_of_day']).size().unstack(fill_value=0)
        tod_prop = tod_counts.div(tod_counts.sum(axis=1), axis=0).fillna(0)
        
        # 2. Wochenend-Präferenz (Wochenend-Checkins / Total Checkins)
        df_temp['is_weekend'] = df_temp['local_time'].dt.dayofweek.isin([5, 6]).astype(int)
        wknd_counts = df_temp.groupby(['user_id', 'is_weekend']).size().unstack(fill_value=0)
        wknd_prop = wknd_counts.div(wknd_counts.sum(axis=1), axis=0).fillna(0)
        weekend_ratio = wknd_prop[1].rename('weekend_ratio') if 1 in wknd_prop.columns else pd.Series(0, index=user_features.index, name='weekend_ratio')
        
        # 3. Exploration Rate (Unique Venues / Total Checkins)
        exploration_rate = df_temp.groupby('user_id').apply(lambda x: x['venue_id'].nunique() / len(x)).rename('exploration_rate')
        
        # Alles zusammenführen
        user_features = pd.concat([user_features, tod_prop, weekend_ratio, exploration_rate], axis=1).fillna(0)
        
        if not inference:
            # Feature-Column-Namen für später merken (Reihenfolge ist wichtig!)
            self.feature_columns = list(user_features.columns)
        else:
            # Reorder columns to match training and handle any missing columns
            for col in self.feature_columns:
                if col not in user_features.columns:
                    user_features[col] = 0
            user_features = user_features[self.feature_columns]
        
        # Reset Index um user_id als Spalte zu bekommen
        user_features = user_features.reset_index()
        
        print(f"  ✅ Extracted {len(self.feature_columns)} features for {len(user_features)} unique users")
        print(f"  Shape: {user_features.shape}")
        
        return user_features
    
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trainiert das K-Means Modell auf den User-Features.
        
        Args:
            df: Input DataFrame mit Spalten wie 'user_id', 'level_1', etc.
            
        Returns:
            DataFrame mit user_id, Cluster-Labels und Feature-Spalten
        """
        print("Training User Partitioning Model (K-Means)...")
        
        # 1. User-Features extrahieren
        user_features_df = self._extract_user_features(df)
        
        # 2. Feature-Matrix vorbereiten (nur die Feature-Spalten)
        X = user_features_df[self.feature_columns].values
        
        # 3. Scaler trainieren und Features normalisieren
        X_scaled = self.scaler.fit_transform(X)
        
        # 4. K-Means trainieren
        print(f"Training K-Means with k={self.k} clusters...")
        self.kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # 5. Cluster-Labels zum DataFrame hinzufügen
        user_features_df['cluster'] = cluster_labels
        
        print("✅ K-Means training completed.")
        # Omit detailed cluster distribution as requested to reduce output flood
        
        return user_features_df
    
    def predict_user_cluster(self, user_category_proportions: dict) -> int:
        """
        Sagt den Cluster für einen neuen User vorher basierend auf seiner Kategorien-Verteilung.
        
        Args:
            user_category_proportions: Dict wie {'Retail': 0.3, 'Food': 0.5, ...}
            
        Returns:
            Cluster-Label (0-9)
        """
        if self.kmeans is None:
            raise ValueError("Modell wurde noch nicht trainiert!")
        
        # Erstelle Feature-Vektor in korrekter Reihenfolge
        feature_vector = np.array([
            user_category_proportions.get(cat, 0) 
            for cat in self.feature_columns
        ]).reshape(1, -1)
        
        # Skaliere und Vorhersage
        feature_vector_scaled = self.scaler.transform(feature_vector)
        cluster = self.kmeans.predict(feature_vector_scaled)[0]
        
        return cluster
    
    def get_cluster_centroids(self) -> pd.DataFrame:
        """
        Gibt die Cluster-Centroids als DataFrame zurück (für Analyse).
        """
        if self.kmeans is None:
            raise ValueError("Modell wurde noch nicht trainiert!")
        
        # Scaler inverse transform um zurück zu Original-Scale zu gehen
        centroids_scaled = self.kmeans.cluster_centers_
        centroids_original = self.scaler.inverse_transform(centroids_scaled)
        
        centroid_df = pd.DataFrame(
            centroids_original,
            columns=self.feature_columns
        )
        centroid_df['cluster'] = range(self.k)
        
        return centroid_df
