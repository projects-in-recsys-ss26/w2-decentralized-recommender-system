from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pickle
import os
import sys

# Füge simple_model zum Suchpfad hinzu, damit pickle die Klassen aus 'src' findet
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simple_model'))

import uvicorn
import httpx
import pandas as pd
from typing import Optional
from dotenv import load_dotenv
import asyncio
import time
import numpy as np

# .env Datei laden
load_dotenv()

CATEGORY_RECOMMENDER_PATH = "./simple_model/trained_model.pkl"
FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")
KMEANS_MODEL_PATH = "./simple_model/user_clustering_model.pkl"
USER_FEATURES_PATH = "../data/user_partitioning.parquet"
CHECKINS_FILE = "../data/preprocessed_checkins_nyc.parquet"
VENUES_FILE = "../data/venues.parquet"

app = FastAPI()

# CORS Middleware für dein React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globaler State für das geladene Dictionary
category_model_dict = {}
kmeans_clustering_model = None
user_features_df = None
checkins_df = None
venues_df = None

@app.on_event("startup")
def load_model_on_startup():
    """
    Lädt das fertige Modell beim Starten der API einmalig aus der Datei.
    Es findet kein Training statt.
    """
    global category_model_dict, kmeans_clustering_model, user_features_df, checkins_df, venues_df
    print("Starting API Server...")
    
    if os.path.exists(CATEGORY_RECOMMENDER_PATH):
        with open(CATEGORY_RECOMMENDER_PATH, 'rb') as f:
            category_model_dict = pickle.load(f)
        print("=== Time-Based Model loaded successfully! 🚀 ===")
    else:
        print(f"⚠️ ERROR: '{CATEGORY_RECOMMENDER_PATH}' not found! Please run main.py first.")
        category_model_dict = {}
    
    if os.path.exists(KMEANS_MODEL_PATH):
        with open(KMEANS_MODEL_PATH, 'rb') as f:
            kmeans_clustering_model = pickle.load(f)
        print("=== K-Means Clustering Model loaded successfully! 🚀 ===")
    else:
        print(f"⚠️ WARNING: '{KMEANS_MODEL_PATH}' not found. Cluster-based recs unavailable.")
    
    if os.path.exists(USER_FEATURES_PATH):
        user_features_df = pd.read_parquet(USER_FEATURES_PATH)
        print(f"=== User Features loaded! ({len(user_features_df)} Users) ===")
    else:
        print(f"⚠️ WARNING: '{USER_FEATURES_PATH}' not found.")
    
    if os.path.exists(CHECKINS_FILE):
        checkins_df = pd.read_parquet(CHECKINS_FILE)
        print(f"=== Checkin data loaded! ({len(checkins_df)} Checkins) ===")
    else:
        print(f"⚠️ WARNING: '{CHECKINS_FILE}' not found.")
        
    if os.path.exists(VENUES_FILE):
        venues_df = pd.read_parquet(VENUES_FILE)
        print(f"=== Venue database loaded! ({len(venues_df)} Venues) ===")
    else:
        print(f"⚠️ WARNING: '{VENUES_FILE}' not found. Please run backend/simple_model/tools/create_venue_db.py.")


@app.get("/api/recommendations")
def get_recommendations(hour: int = None):
    """
    Nimmt die Stunde entgegen und gibt die Top 3 Kategorien zurück.
    """
    if hour is None:
        hour = datetime.now().hour

    # Inferenz ist jetzt ein extrem schneller Dictionary-Lookup O(1)
    top_categories = category_model_dict.get(hour, [])

    return {
        "requested_hour": hour,
        "top_3_categories": top_categories
    }


@app.get("/api/recommendations/cluster")
def get_recommendations_by_cluster(hour: int = None, cluster: int = None):
    """
    Nimmt die Stunde und User-Cluster entgegen und gibt die Top 3 Kategorien zurück.
    
    Args:
        hour: Stunde (0-23). Falls None, wird aktuelle Stunde verwendet.
        cluster: User-Cluster (0-9). Falls None, gibt Global-Recommendations.
        
    Returns:
        Dictionary mit Recommendations basierend auf Cluster + Hour
    """
    if hour is None:
        hour = datetime.now().hour
    
    if cluster is None:
        # Fallback auf globale Recommendations
        top_categories = category_model_dict.get(hour, [])
        return {
            "requested_hour": hour,
            "cluster": None,
            "top_3_categories": top_categories,
            "type": "global"
        }
    
    # Cluster-basierte Recommendations
    # Das Dictionary hat jetzt Struktur: {cluster: {hour: [categories]}}
    cluster_data = category_model_dict.get(cluster, {})
    top_categories = cluster_data.get(hour, []) if isinstance(cluster_data, dict) else []
    
    return {
        "requested_hour": hour,
        "cluster": cluster,
        "top_3_categories": top_categories,
        "type": "cluster_specific"
    }


@app.post("/api/user-cluster")
async def predict_user_cluster(user_features: dict):
    """
    Sagt den Cluster für einen User vorher basierend auf seiner Kategorien-Verteilung und Verhaltens-Features.
    
    Input Example:
    {
        "Retail": 0.3,
        "Dining and Drinking": 0.5,
        "Arts and Entertainment": 0.1,
        "exploration_rate": 0.7,
        "morning": 0.2,
        "weekend_ratio": 0.4
        ...
    }
    
    Returns:
        Dictionary mit predicted cluster (0-9)
    """
    if kmeans_clustering_model is None:
        return {"error": "K-Means Clustering Modell nicht geladen"}
    
    try:
        cluster = kmeans_clustering_model.predict_user_cluster(user_features)
        return {
            "predicted_cluster": int(cluster),
            "confidence": "high"  # Kann später erweitert werden
        }
    except Exception as e:
        print(f"❌ Exception in predict_user_cluster: {str(e)}")
        return {"error": str(e)}


@app.get("/api/example-user-recommendations")
def get_example_user_recommendations(user_index: int = 0, hour: int = None):
    """
    Nimmt einen Example-User aus den Daten (basierend auf user_index),
    berechnet seinen Cluster und gibt Cluster-basierte Recommendations.
    
    Args:
        user_index: Index des Users in der user_features_df (0-indexed)
        hour: Stunde (0-23). Falls None, wird aktuelle Stunde verwendet.
        
    Returns:
        Dictionary mit:
        - User-Info (user_id, Kategorien-Verteilung)
        - Predicted Cluster
        - Cluster-basierte Recommendations
    """
    if hour is None:
        hour = datetime.now().hour
    
    if user_features_df is None or len(user_features_df) == 0:
        return {"error": "User-Features nicht geladen"}
    
    if user_index < 0 or user_index >= len(user_features_df):
        return {"error": f"User-Index out of range. Max: {len(user_features_df)-1}"}
    
    try:
        # 1. User aus user_features_df holen
        user_row = user_features_df.iloc[user_index]
        user_id = user_row['user_id']
        predicted_cluster = int(user_row['cluster'])
        
        # 2. Alle Features des Users extrahieren
        feature_columns = [col for col in user_features_df.columns if col not in ['user_id', 'cluster']]
        all_features = {col: float(user_row[col]) for col in feature_columns}
        
        # In Kategorien und neue Features aufteilen (fürs Frontend)
        behavioral_keys = ['morning', 'afternoon', 'evening', 'night', 'weekend_ratio', 'exploration_rate']
        category_distribution = {k: v for k, v in all_features.items() if k not in behavioral_keys}
        behavioral_features = {k: v for k, v in all_features.items() if k in behavioral_keys}
        
        # 3. Recommendations basierend auf Cluster + Hour
        cluster_data = category_model_dict.get(predicted_cluster, {})
        top_categories = cluster_data.get(hour, []) if isinstance(cluster_data, dict) else []
        
        # 4. Stats des Users
        if checkins_df is not None:
            user_checkins = checkins_df[checkins_df['user_id'] == user_id]
            num_checkins = len(user_checkins)
            
            target_time_str = None
            # Letzten Check-in als "Live-Test" Position verwenden
            if num_checkins > 0:
                if 'utc_time' in user_checkins.columns:
                    last_checkin = user_checkins.sort_values('utc_time').iloc[-1]
                    
                    # Lokale Zeit des Check-ins berechnen
                    try:
                        utc_time = pd.to_datetime(last_checkin['utc_time'])
                        if utc_time.tzinfo is not None:
                            utc_time = utc_time.tz_convert(None) # Macht die Zeit zeitzonen-naiv (reines UTC)
                        tz_offset = int(last_checkin.get('timezone_offset', 0))
                        local_time = utc_time + pd.Timedelta(minutes=tz_offset)
                        target_time_str = local_time.isoformat()
                    except Exception as e:
                        print(f"Time parse error: {e}")
                else:
                    last_checkin = user_checkins.iloc[-1]
                    
                target_lat = float(last_checkin['latitude'])
                target_lng = float(last_checkin['longitude'])
                target_name = str(last_checkin.get('venue_category_name', 'Unknown Place'))
            else:
                target_lat, target_lng = 40.7128, -74.0060  # NYC Default Fallback
                target_name = "Unknown Place"
        else:
            num_checkins = "unknown"
            target_time_str = None
            target_lat, target_lng = 40.7128, -74.0060  # NYC Default Fallback
            target_name = "Unknown Place"
        
        return {
            "user_index": user_index,
            "user_id": int(user_id),
            "num_checkins": num_checkins,
            "location": {
                "lat": target_lat,
                "lng": target_lng,
                "name": target_name,
                "local_time": target_time_str
            },
            "category_distribution": category_distribution,
            "behavioral_features": behavioral_features,
            "all_features": all_features,
            "predicted_cluster": predicted_cluster,
            "requested_hour": hour,
            "recommendations": {
                "top_categories": top_categories,
                "cluster_data": cluster_data,
                "type": "cluster_specific"
            }
        }
    
    except Exception as e:
        print(f"❌ Exception in get_example_user_recommendations: {str(e)}")
        return {"error": str(e)}


@app.get("/api/example-users-list")
def get_example_users_list(limit: int = 10):
    """
    Gibt eine Liste von Example-Users mit ihren Cluster-Zuweisungen.
    
    Args:
        limit: Maximale Anzahl von Users in der Liste
        
    Returns:
        Liste von Users mit Index, ID und Cluster
    """
    if user_features_df is None or len(user_features_df) == 0:
        return {"error": "User-Features nicht geladen"}
    
    try:
        limit = min(limit, len(user_features_df))
        users_list = []
        
        for idx in range(limit):
            user_row = user_features_df.iloc[idx]
            users_list.append({
                "index": idx,
                "user_id": int(user_row['user_id']),
                "cluster": int(user_row['cluster']),
                "url": f"/api/example-user-recommendations?user_index={idx}"
            })
        
        return {
            "total_users": len(user_features_df),
            "users": users_list
        }
    
    except Exception as e:
        print(f"❌ Exception in get_example_users_list: {str(e)}")
        return {"error": str(e)}


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Berechnet die Distanz in Metern zwischen zwei Punkten auf der Erde.
    Unterstützt Numpy Arrays für lat2, lon2.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000 # Radius der Erde in Metern
    return c * r

@app.get("/api/venues/search")
async def search_venues(
    query: str,
    lat: float,
    lng: float,
    radius: int = 1500,
    limit: int = 3
):
    """
    Sucht Orte in der lokalen Venue-Datenbank basierend auf der Kategorie (query) und Distanz.
    Ersetzt die alte Foursquare API.
    """
    print(f"🔍 Local Venue Search: query='{query}', lat={lat}, lng={lng}, radius={radius}")
    
    if venues_df is None:
        print("❌ Venues database not loaded!")
        return {"error": "Venues Datenbank nicht geladen"}
    
    try:
        # 1. Filtern nach Kategorie (query)
        # Die Foursquare Kategorie entspricht in unserer DB der 'category' Spalte
        filtered_df = venues_df[venues_df['category'].str.lower() == query.lower()].copy()
        
        if filtered_df.empty:
            print(f"⚠️ No venues found for category '{query}'.")
            return {"results": []}
            
        # 2. Distanz berechnen
        distances = haversine_vectorized(lat, lng, filtered_df['latitude'].values, filtered_df['longitude'].values)
        filtered_df['distance'] = distances
        
        # 3. Filtern nach Radius und sortieren nach Popularität (checkin_count) statt Distanz
        nearby_df = filtered_df[filtered_df['distance'] <= radius].sort_values('checkin_count', ascending=False)
        
        # 4. Limit anwenden
        top_venues = nearby_df.head(limit)
        
        # Formatiere die Ergebnisse für das Frontend
        formatted_results = []
        for _, place in top_venues.iterrows():
            formatted_place = {
                "id": str(place['venue_id']),
                "name": str(place['category']), # Wir verwenden die Kategorie als Namen, da spezifische Namen fehlen
                "lat": float(place['latitude']),
                "lng": float(place['longitude']),
                "distance": float(place['distance']),
                "popularity": int(place.get('checkin_count', 0)),
                "address": "", # Historische Daten haben keine Adresse
                "website": "", # Historische Daten haben keine Website
            formatted_results.append(formatted_place)
            
        print(f"📤 Returning {len(formatted_results)} result(s) for '{query}'\n")
        return {"results": formatted_results}
    
    except Exception as e:
        print(f"❌ Exception in search_venues: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)