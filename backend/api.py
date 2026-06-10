from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pickle
import os
import uvicorn
import httpx
from typing import Optional
from dotenv import load_dotenv
import asyncio
import time

# .env Datei laden
load_dotenv()

MODEL_PATH = "trained_model.pkl"
FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")
KMEANS_MODEL_PATH = "user_clustering_model.pkl"

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
trained_model_dict = {}
kmeans_clustering_model = None

@app.on_event("startup")
def load_model_on_startup():
    """
    Lädt das fertige Modell beim Starten der API einmalig aus der Datei.
    Es findet kein Training statt.
    """
    global trained_model_dict, kmeans_clustering_model
    print("Starte API-Server...")
    
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            trained_model_dict = pickle.load(f)
        print("=== Time-Based Modell erfolgreich aus Datei geladen! 🚀 ===")
    else:
        print(f"⚠️ FEHLER: '{MODEL_PATH}' wurde nicht gefunden! Bitte führe zuerst main.py aus.")
        trained_model_dict = {}
    
    if os.path.exists(KMEANS_MODEL_PATH):
        with open(KMEANS_MODEL_PATH, 'rb') as f:
            kmeans_clustering_model = pickle.load(f)
        print("=== K-Means Clustering Modell erfolgreich aus Datei geladen! 🚀 ===")
    else:
        print(f"⚠️ WARNUNG: '{KMEANS_MODEL_PATH}' wurde nicht gefunden. Cluster-basierte Recs nicht verfügbar.")


@app.get("/api/recommendations")
def get_recommendations(hour: int = None):
    """
    Nimmt die Stunde entgegen und gibt die Top 3 Kategorien zurück.
    """
    if hour is None:
        hour = datetime.now().hour

    # Inferenz ist jetzt ein extrem schneller Dictionary-Lookup O(1)
    top_categories = trained_model_dict.get(hour, [])

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
        top_categories = trained_model_dict.get(hour, [])
        return {
            "requested_hour": hour,
            "cluster": None,
            "top_3_categories": top_categories,
            "type": "global"
        }
    
    # Cluster-basierte Recommendations
    # Das Dictionary hat jetzt Struktur: {cluster: {hour: [categories]}}
    cluster_data = trained_model_dict.get(cluster, {})
    top_categories = cluster_data.get(hour, []) if isinstance(cluster_data, dict) else []
    
    return {
        "requested_hour": hour,
        "cluster": cluster,
        "top_3_categories": top_categories,
        "type": "cluster_specific"
    }


@app.post("/api/user-cluster")
async def predict_user_cluster(user_categories: dict):
    """
    Sagt den Cluster für einen User vorher basierend auf seiner Kategorien-Verteilung.
    
    Input Example:
    {
        "Retail": 0.3,
        "Dining and Drinking": 0.5,
        "Arts and Entertainment": 0.1,
        ...
    }
    
    Returns:
        Dictionary mit predicted cluster (0-9)
    """
    if kmeans_clustering_model is None:
        return {"error": "K-Means Clustering Modell nicht geladen"}
    
    try:
        cluster = kmeans_clustering_model.predict_user_cluster(user_categories)
        return {
            "predicted_cluster": int(cluster),
            "confidence": "high"  # Kann später erweitert werden
        }
    except Exception as e:
        print(f"❌ Exception in predict_user_cluster: {str(e)}")
        return {"error": str(e)}


@app.get("/api/foursquare/search")
async def search_foursquare(
    query: str,
    lat: float,
    lng: float,
    radius: int = 1500,
    limit: int = 1
):
    """
    Sucht Orte auf Foursquare über ein Backend-Proxy um CORS-Probleme zu vermeiden.
    """
    print(f"🔍 Foursquare Search: query='{query}', lat={lat}, lng={lng}, radius={radius}")
    
    if not FOURSQUARE_API_KEY:
        print("❌ Foursquare API Key nicht konfiguriert!")
        return {"error": "Foursquare API Key nicht konfiguriert"}
    
    try:
        search_params = {
            "query": query,
            "ll": f"{lat},{lng}",
            "radius": str(radius),
            "limit": str(limit),
            "fields": "fsq_place_id,name,latitude,longitude,location,website"
        }
        
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {FOURSQUARE_API_KEY}",
            "X-Places-Api-Version": "2025-06-17",
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://places-api.foursquare.com/places/search",
                params=search_params,
                headers=headers,
                timeout=10.0
            )
        
        if response.status_code != 200:
            print(f"❌ Foursquare API error: {response.status_code} \n {response.text}")
            return {"error": f"Foursquare API error: {response.status_code}"}
            
        data = response.json()
        print(f"📍 Foursquare Response für '{query}':", data)
        
        # Formatiere die Ergebnisse für das Frontend
        formatted_results = []
        if data.get("results"):
            for place in data["results"]:
                formatted_place = {
                    "id": place.get("fsq_place_id"),
                    "name": place.get("name"),
                    "lat": place.get("latitude"),
                    "lng": place.get("longitude"),
                    "address": place.get("location", {}).get("formatted_address", "No address"),
                    "website": place.get("website"),
                }
                formatted_results.append(formatted_place)
                print(f"✅ Place found: {formatted_place['name']} ({formatted_place['id']})")
        else:
            print(f"⚠️  Keine Ergebnisse für '{query}' gefunden")
        
        print(f"📤 Returning {len(formatted_results)} result(s) for '{query}'\n")
        return {"results": formatted_results}
    
    except Exception as e:
        print(f"❌ Exception in search_foursquare: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)