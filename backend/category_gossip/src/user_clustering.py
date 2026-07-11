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

    def fit_decentralized(self, df: pd.DataFrame, gossip_rounds: int = 15, epochs: int = 5) -> pd.DataFrame:
        """
        Trainiert das K-Means Modell vollständig dezentral mit dem Gossip-Protokoll.
        
        Schritt 1: Dezentrale Skalierung (Gossip für Mean und Varianz)
        Schritt 2: Dezentrales K-Means (Gossip für Centroids)
        """
        print("🌐 Training Decentralized User Partitioning Model (Gossip K-Means)...")
        
        # 1. Lokale Features extrahieren (Jeder User ist ein unabhängiger Knoten)
        user_features_df = self._extract_user_features(df)
        X = user_features_df[self.feature_columns].values
        n_users, n_features = X.shape
        
        # --- PHASE 1: Decentralized Standard Scaling ---
        print(f"  🔄 Phase 1: Decentralized Standard Scaling ({gossip_rounds} gossip rounds)...")
        # Knoten für Skalierung initialisieren
        scale_nodes = []
        for i in range(n_users):
            scale_nodes.append({
                'sum': X[i].copy(),
                'sq_sum': (X[i]**2).copy()
            })
            
        # Gossip rounds
        for r in range(gossip_rounds):
            indices = np.random.permutation(n_users)
            for p in range(n_users // 2):
                i, j = indices[2*p], indices[2*p+1]
                avg_sum = (scale_nodes[i]['sum'] + scale_nodes[j]['sum']) / 2.0
                avg_sq_sum = (scale_nodes[i]['sq_sum'] + scale_nodes[j]['sq_sum']) / 2.0
                
                scale_nodes[i]['sum'] = avg_sum
                scale_nodes[i]['sq_sum'] = avg_sq_sum
                scale_nodes[j]['sum'] = avg_sum.copy()
                scale_nodes[j]['sq_sum'] = avg_sq_sum.copy()
                
        # Konvergierte Werte extrapolieren (von Knoten 0)
        global_mean = scale_nodes[0]['sum']
        global_sq_sum = scale_nodes[0]['sq_sum']
        global_var = global_sq_sum - global_mean**2
        global_std = np.sqrt(np.maximum(global_var, 1e-10))
        
        # Manuelle Skalierung (lokal bei jedem Knoten mit den nun bekannten globalen Parametern)
        self.scaler.mean_ = global_mean
        self.scaler.scale_ = global_std
        self.scaler.var_ = global_var
        self.scaler.n_features_in_ = n_features
        
        X_scaled = (X - global_mean) / global_std
        
        # --- PHASE 2: Decentralized K-Means ---
        print(f"  🔄 Phase 2: Decentralized Gossip K-Means (k={self.k}, epochs={epochs})...")
        
        # Initialisierung der Centroids (Einigung auf gemeinsame Startpunkte z.B. per Seed)
        np.random.seed(42)
        initial_indices = np.random.choice(n_users, self.k, replace=False)
        centroids = X_scaled[initial_indices].copy()
        
        for epoch in range(epochs):
            # 1. Lokales Assignment und Initialisierung für Gossip
            kmeans_nodes = []
            
            for i in range(n_users):
                distances = np.linalg.norm(centroids - X_scaled[i], axis=1)
                best_cluster = np.argmin(distances)
                
                node_sum = np.zeros((self.k, n_features))
                node_count = np.zeros(self.k)
                
                node_sum[best_cluster] = X_scaled[i]
                node_count[best_cluster] = 1.0
                
                kmeans_nodes.append({
                    'sum': node_sum,
                    'count': node_count
                })
                
            # 2. Gossip für die Centroids
            for r in range(gossip_rounds):
                indices = np.random.permutation(n_users)
                for p in range(n_users // 2):
                    i, j = indices[2*p], indices[2*p+1]
                    
                    avg_sum = (kmeans_nodes[i]['sum'] + kmeans_nodes[j]['sum']) / 2.0
                    avg_count = (kmeans_nodes[i]['count'] + kmeans_nodes[j]['count']) / 2.0
                    
                    kmeans_nodes[i]['sum'] = avg_sum
                    kmeans_nodes[i]['count'] = avg_count
                    kmeans_nodes[j]['sum'] = avg_sum.copy()
                    kmeans_nodes[j]['count'] = avg_count.copy()
                    
            # 3. Centroids updaten aus konvergiertem State (Knoten 0 ist repräsentativ)
            rep = kmeans_nodes[0]
            for c in range(self.k):
                if rep['count'][c] > 0:
                    centroids[c] = rep['sum'][c] / rep['count'][c]
                    
            print(f"    Epoch {epoch+1}/{epochs} completed.")
            
        # --- PHASE 3: Modell Finalisieren ---
        # Duck-Typing Klasse um sklearn.cluster.KMeans zu ersetzen
        class DecentralizedKMeansWrapper:
            def __init__(self, centers):
                self.cluster_centers_ = centers
            def predict(self, X):
                dist = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
                return np.argmin(dist, axis=1)
                
        self.kmeans = DecentralizedKMeansWrapper(centroids)
        
        # Finales Assignment für die Rückgabe
        final_labels = self.kmeans.predict(X_scaled)
        user_features_df['cluster'] = final_labels
        
        print("✅ Decentralized K-Means training completed.")
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
