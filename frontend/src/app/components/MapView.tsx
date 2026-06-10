import React, { useEffect, useState } from "react";
import { Navigation, MapPin, Settings, Bell, Lock, Clock } from "lucide-react";
import { useNavigate } from "react-router";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import { useNotification } from "./NotificationContext";

// Interface für die Foursquare Orte im State
interface FoursquarePlace {
  id: string;
  name: string;
  lat: number;
  lng: number;
  category: string;
  address?: string;
  website?: string;
}

// Create custom icons using DivIcon for Tailwind styling compatibility
const createUserIcon = () => {
  return L.divIcon({
    html: `
      <div class="relative w-8 h-8 flex items-center justify-center">
        <div class="absolute inset-0 bg-blue-500/30 rounded-full animate-ping"></div>
        <div class="absolute inset-1 bg-blue-500/20 rounded-full"></div>
        <div class="relative bg-blue-600 w-8 h-8 rounded-full border-2 border-white shadow-xl flex items-center justify-center">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="-rotate-45 ml-[-2px] mt-[2px]"><polygon points="3 11 22 2 13 21 11 13 3 11"></polygon></svg>
        </div>
      </div>
    `,
    className: 'custom-leaflet-icon',
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -16]
  });
};

const createRecommendationIcon = (label: string) => {
  // Dynamische Farbwahl basierend auf dem Kategorienamen
  const lowerLabel = label.toLowerCase();
  let bgColor = 'bg-indigo-500'; // Default
  
  if (lowerLabel.includes('mensa') || lowerLabel.includes('food') || lowerLabel.includes('restaurant')) {
    bgColor = 'bg-orange-500';
  } else if (lowerLabel.includes('cafe') || lowerLabel.includes('coffee') || lowerLabel.includes('kaffee')) {
    bgColor = 'bg-amber-600';
  } else if (lowerLabel.includes('mobility') || lowerLabel.includes('research') || lowerLabel.includes('uni')) {
    bgColor = 'bg-blue-600';
  }

  return L.divIcon({
    html: `
      <div class="relative flex flex-col items-center gap-1 cursor-pointer group transform transition-transform hover:scale-105">
        <div class="relative ${bgColor} w-7 h-7 rounded-full border-2 border-white shadow-md flex items-center justify-center text-white z-10">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m12 8-4 4h8z"/></svg>
        </div>
        <div class="bg-white/95 backdrop-blur-sm px-2 py-0.5 rounded-full shadow-sm border border-gray-100/50 whitespace-nowrap -mt-2 pt-2 pb-1">
          <span class="text-[10px] font-bold text-gray-700 leading-none">${label}</span>
        </div>
      </div>
    `,
    className: 'custom-leaflet-icon',
    iconSize: [120, 50],
    iconAnchor: [60, 25],
    popupAnchor: [0, -25]
  });
};

export function MapView() {
  const navigate = useNavigate();
  const { triggerNotification } = useNotification();
  const [currentHour, setCurrentHour] = useState(new Date().getHours());
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [foursquarePlaces, setFoursquarePlaces] = useState<FoursquarePlace[]>([]);
  const [loading, setLoading] = useState(true);

  // Garching Forschungszentrum Coordinates
  const userPos: [number, number] = [48.2650, 11.6702];

  useEffect(() => {
    // Inject Leaflet CSS if not already present
    if (!document.getElementById("leaflet-css")) {
      const link = document.createElement("link");
      link.id = "leaflet-css";
      link.rel = "stylesheet";
      link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
      document.head.appendChild(link);
    }

    // Fetch recommendations from backend
    const fetchRecommendations = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/example-user-recommendations?hour=${currentHour}`);
        const data = await response.json();
        console.log("📍 Backend Response:", data);
        console.log("🎯 Top 3 Kategorien:", data.recommendations["top_3_categories"]);
        setRecommendations(data.recommendations["top_3_categories"] || []);
      } catch (error) {
        console.error("❌ Fehler beim Abrufen der Empfehlungen:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
  }, [currentHour]);

  // Sucht Orte auf Foursquare basierend auf den erhaltenen Kategorien
  useEffect(() => {
    if (recommendations.length === 0) return;

    const fetchFoursquarePlaces = async () => {
      const fetchedPlaces: FoursquarePlace[] = [];

      // Für jede Kategorie aus dem localhost-Call senden wir eine Anfrage an den Backend-Proxy
      for (let index = 0; index < recommendations.length; index++) {
        const category = recommendations[index];
        
        try {
          // Kleine Verzögerung zwischen Requests um Rate Limiting zu vermeiden (500ms)
          if (index > 0) {
            await new Promise(resolve => setTimeout(resolve, 500));
          }
          
          const response = await fetch(
            `http://localhost:8000/api/foursquare/search?query=${encodeURIComponent(category)}&lat=${userPos[0]}&lng=${userPos[1]}&radius=1500&limit=1`,
            {
              method: "GET",
              headers: {
                Accept: "application/json",
              },
            }
          );

          if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
          
          const result = await response.json();
          
          if (result.results && result.results.length > 0) {
            const place = result.results[0];
            fetchedPlaces.push({
              id: place.id,
              name: place.name,
              lat: place.lat,
              lng: place.lng,
              category: category,
              address: place.address,
              website: place.website
            });
          }
        } catch (error) {
          console.error(`❌ Fehler beim Abrufen der Foursquare-Daten für '${category}':`, error);
        }
      }

      setFoursquarePlaces(fetchedPlaces);
    };

    fetchFoursquarePlaces();
  }, [recommendations]);

  return (
    <div className="relative w-full h-full bg-slate-50">
      
      {/* Interactive Map */}
      <div className="absolute inset-0 z-0 [&_.leaflet-control-attribution]:hidden">
        <MapContainer 
          center={userPos} 
          zoom={15} 
          zoomControl={false}
          style={{ height: "100%", width: "100%", zIndex: 0 }}
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          />
          
          {/* User Marker */}
          <Marker position={userPos} icon={createUserIcon()}>
            <Popup className="rounded-xl overflow-hidden shadow-xl border-none">
              <div className="text-center font-semibold text-gray-900 py-1">You are here<br/><span className="font-normal text-xs text-gray-500">TUM Campus Garching</span></div>
            </Popup>
          </Marker>

          {/* Dynamische Foursquare Marker */}
          {!loading && foursquarePlaces.map((place) => (
            <Marker 
              key={place.id} 
              position={[place.lat, place.lng]} 
              icon={createRecommendationIcon(place.category)}
            >
              <Popup className="!p-0 border-none overflow-hidden rounded-xl shadow-lg m-0 w-[180px]">
                <div className="flex flex-col">
                  <div className="p-3 bg-white">
                    <div className="flex justify-between items-start mb-1 gap-1">
                      <h3 className="font-bold text-gray-900 leading-tight text-sm break-words flex-1">{place.name}</h3>
                    </div>
                    <p className="text-[11px] text-gray-500 line-clamp-2">{place.address}</p>
                    {place.website && (
                      <a 
                        href={place.website} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="inline-block mt-2 text-[9px] bg-blue-100 text-blue-600 font-semibold px-2 py-0.5 rounded-full hover:bg-blue-200 transition-colors"
                      >
                        Website besuchen ↗
                      </a>
                    )}
                    <span className="inline-block mt-2 ml-2 text-[9px] bg-slate-100 text-slate-600 font-semibold px-2 py-0.5 rounded-full">
                      Kategorie: {place.category}
                    </span>
                  </div>
                </div>
              </Popup>
            </Marker>
          ))}

        </MapContainer>
      </div>

      {/* Header / Search Bar */}
      <div className="absolute top-12 inset-x-4 flex gap-2 z-[1000] pointer-events-none">
        <div className="flex-1 bg-white/90 backdrop-blur-md rounded-full px-4 py-3 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.1)] flex items-center gap-3 pointer-events-auto border border-gray-100/50">
          <MapPin className="w-5 h-5 text-gray-400" />
          <input 
            type="text" 
            placeholder="Search destination..." 
            defaultValue="Garching Forschungszentrum"
            className="bg-transparent border-none outline-none w-full text-sm font-medium placeholder:text-gray-400"
          />
        </div>
        
        {/* Current Hour Display */}
        <div className="bg-white/90 backdrop-blur-md rounded-full px-4 py-3 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.1)] flex items-center gap-2 pointer-events-auto border border-gray-100/50">
          <Clock className="w-5 h-5 text-blue-600" />
          <span className="text-sm font-bold text-gray-900">{currentHour.toString().padStart(2, '0')}:00</span>
        </div>
      </div>

      {/* Bottom Navigation */}
      <div className="absolute bottom-8 inset-x-6 z-[1000]">
        <div className="bg-zinc-900/95 backdrop-blur-lg text-white rounded-[32px] px-6 py-4 flex justify-between items-center shadow-2xl border border-zinc-800">
          <button className="flex flex-col items-center gap-1 text-white hover:text-blue-400 transition-colors">
            <MapPin className="w-6 h-6" />
          </button>
          
          <button className="w-14 h-14 bg-blue-600 rounded-full flex items-center justify-center shadow-[0_0_20px_rgba(37,99,235,0.5)] -mt-10 border-[6px] border-zinc-900 text-white transform transition-transform hover:scale-105 active:scale-95">
            <Navigation className="w-6 h-6 -rotate-45 ml-[-2px] mt-[2px]" />
          </button>

          <button 
            className="flex flex-col items-center gap-1 text-gray-400 hover:text-white transition-colors"
            onClick={() => navigate('/settings')}
          >
            <Settings className="w-6 h-6" />
          </button>
        </div>
      </div>
      
      {/* Dev / Prototype Controls */}
      <div className="absolute bottom-32 right-4 flex flex-col gap-2 z-[1000]">
        <button 
          onClick={() => triggerNotification({
            title: foursquarePlaces[0]?.name || "Notification",
            message: "You might like this based on your interests!",
            time: "now"
          })}
          className="w-12 h-12 bg-white/90 backdrop-blur-md rounded-full shadow-[0_4px_12px_rgba(0,0,0,0.1)] border border-gray-100 flex items-center justify-center text-amber-600 hover:bg-white active:scale-95 transition-all"
          title="Sample Notification"
        >
          <Bell className="w-5 h-5" />
        </button>
        <button 
          onClick={() => navigate('/lock')}
          className="w-12 h-12 bg-white/90 backdrop-blur-md rounded-full shadow-[0_4px_12px_rgba(0,0,0,0.1)] border border-gray-100 flex items-center justify-center text-gray-700 hover:bg-white active:scale-95 transition-all"
          title="Lock Screen"
        >
          <Lock className="w-5 h-5" />
        </button>
      </div>

    </div>
  );
}