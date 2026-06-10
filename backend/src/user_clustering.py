import pandas as pd
import numpy as np
import pickle
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
        
    def _extract_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrahiert pro User die prozentuale Verteilung der level_1 Kategorien.
        
        Returns:
            DataFrame mit user_id und prozentuale Features für die Top-9 level_1 Kategorien
        """
        print("Extrahiere User-Features (Level-1 Kategorien Verteilung)...")
        
        # Gruppiere nach user_id und zähle level_1 Kategorien
        user_category_counts = df.groupby(['user_id', 'level_1']).size().unstack(fill_value=0)
        
        # Finde die Top-9 level_1 Kategorien insgesamt
        category_totals = df['level_1'].value_counts()
        self.top_level_1_categories = category_totals.head(self.top_categories).index.tolist()
        
        print(f"  Top-{self.top_categories} level_1 Kategorien: {self.top_level_1_categories}")
        
        # Behalte nur die Top-9 Kategorien
        user_category_counts = user_category_counts[self.top_level_1_categories]
        
        # Normalisiere zu Prozenten (0-1)
        user_category_proportions = user_category_counts.div(user_category_counts.sum(axis=1), axis=0).fillna(0)
        
        # Reset Index um user_id als Spalte zu bekommen
        user_category_proportions = user_category_proportions.reset_index()
        
        # Feature-Column-Namen für später
        self.feature_columns = self.top_level_1_categories
        
        print(f"  ✅ Extrahiert Features für {len(user_category_proportions)} unique Users")
        print(f"  Shape: {user_category_proportions.shape}")
        
        return user_category_proportions
    
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trainiert das K-Means Modell auf den User-Features.
        
        Args:
            df: Input DataFrame mit Spalten wie 'user_id', 'level_1', etc.
            
        Returns:
            DataFrame mit user_id, Cluster-Labels und Feature-Spalten
        """
        print("Trainiere User Partitioning Model (K-Means)...")
        
        # 1. User-Features extrahieren
        user_features_df = self._extract_user_features(df)
        
        # 2. Feature-Matrix vorbereiten (nur die Feature-Spalten)
        X = user_features_df[self.feature_columns].values
        
        # 3. Scaler trainieren und Features normalisieren
        X_scaled = self.scaler.fit_transform(X)
        
        # 4. K-Means trainieren
        print(f"Trainiere K-Means mit k={self.k} Clustern...")
        self.kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # 5. Cluster-Labels zum DataFrame hinzufügen
        user_features_df['cluster'] = cluster_labels
        
        print(f"✅ K-Means Training abgeschlossen.")
        print(f"  Cluster-Verteilung:\n{user_features_df['cluster'].value_counts().sort_index()}")
        
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
