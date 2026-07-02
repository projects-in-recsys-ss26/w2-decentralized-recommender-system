import React, { useEffect, useState, useMemo, useRef, useImperativeHandle, forwardRef } from "react";
import { MapPin, Bell, Lock, Clock, SkipForward } from "lucide-react";
import { useNavigate } from "react-router";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import { useNotification } from "./NotificationContext";
import { BottomNavBar } from "./BottomNavBar";
import categoryLevel1Map from '../../../../data/category_level1_map.json'

// Interface für die lokalen Orte im State
interface VenuePlace {
  id: string;
  name: string;
  lat: number;
  lng: number;
  categories: string[];
}

// Create custom icons using DivIcon for Tailwind styling compatibility
const createUserIcon = () => {
  return L.divIcon({
    html: `
      <div class="relative w-10 h-10 flex items-center justify-center">
        <div class="absolute inset-0 rounded-full animate-ping" style="background-color: rgba(59, 130, 246, 0.4);"></div>
        <div class="absolute inset-1 rounded-full" style="background-color: rgba(59, 130, 246, 0.2);"></div>
        <div class="relative w-8 h-8 rounded-full border-2 border-white shadow-xl flex items-center justify-center" style="background-color: #2563eb;">
          <span class="material-symbols-outlined text-white" style="font-size: 18px;">person</span>
        </div>
      </div>
    `,
    className: 'custom-leaflet-icon',
    iconSize: [40, 40],
    iconAnchor: [20, 20],
    popupAnchor: [0, -20]
  });
};

// Create custom icons using DivIcon for Tailwind styling compatibility
const createActualCheckinIcon = () => {
  return L.divIcon({
    html: `
      <div class="relative w-10 h-10 flex items-center justify-center">
        <div class="absolute inset-0 bg-sky-400/40 rounded-full animate-ping"></div>
        <div class="absolute inset-1 bg-sky-400/20 rounded-full"></div>
        <div class="relative bg-sky-500 w-8 h-8 rounded-full border-2 border-white shadow-xl flex items-center justify-center">
          <span class="material-symbols-outlined text-white" style="font-size: 18px;">my_location</span>
        </div>
      </div>
    `,
    className: 'custom-leaflet-icon',
    iconSize: [40, 40],
    iconAnchor: [20, 20],
    popupAnchor: [0, -20]
  });
};

const createRecommendationIcon = (category: string, isMatch: boolean = false) => {
  const cfg = getCategoryConfig(category)
  
  const bgColorMap: Record<string, { bg: string; border: string }> = {
    "Arts and Entertainment":             { bg: "#ede9fe", border: "#7c3aed" },
    "Business and Professional Services": { bg: "#e2e8f0", border: "#475569" },
    "Community and Government":           { bg: "#dbeafe", border: "#2563eb" },
    "Dining and Drinking":                { bg: "#ffedd5", border: "#ea580c" },
    "Event":                              { bg: "#fef9c3", border: "#ca8a04" },
    "Health and Medicine":                { bg: "#fee2e2", border: "#dc2626" },
    "Landmarks and Outdoors":             { bg: "#fef3e2", border: "#a16207" },
    "Nightlife Spot":                     { bg: "#e0e7ff", border: "#4f46e5" },
    "Retail":                             { bg: "#fce7f3", border: "#c026d3" },
    "Sports and Recreation":              { bg: "#ccfbf1", border: "#0d9488" },
    "Travel and Transportation":          { bg: "#e0f2fe", border: "#0284c7" },
  }

  const level1 = (categoryLevel1Map as Record<string, string>)[category]
  let colors = bgColorMap[level1] ?? { bg: "#fdf6e3", border: "#b8860b" } // Warm amber for everyday/non-tourist places
  
  if (isMatch) {
    colors = { bg: "#dcfce7", border: "#22c55e" } // Light green bg, bright green border
  }

  return L.divIcon({
    html: `
      <div style="display:flex; flex-direction:column; align-items:center; gap:4px; cursor:pointer;">
        <div style="
          width: 36px; height: 36px;
          background: ${colors.bg};
          border: 2px solid ${colors.border};
          border-radius: 50%;
          box-shadow: 0 2px 8px rgba(0,0,0,0.12);
          display: flex; align-items: center; justify-content: center;
          font-size: 18px;
          z-index: 10;
        "><span class="material-symbols-outlined" style="font-size: 20px; color: ${colors.border};">${cfg.icon}</span></div>
        <div style="
          background: white;
          border: 1px solid rgba(0,0,0,0.08);
          padding: 2px 8px;
          border-radius: 999px;
          box-shadow: 0 1px 4px rgba(0,0,0,0.08);
          margin-top: -6px;
          padding-top: 6px;
          white-space: nowrap;
        ">
          <span style="font-size: 10px; font-weight: 700; color: #374151; line-height: 1;">${category}</span>
        </div>
      </div>
    `,
    className: 'custom-leaflet-icon',
    iconSize: [120, 58],
    iconAnchor: [60, 29],
    popupAnchor: [0, -29]
  })
}

const LEVEL1_CONFIG = {
  "Arts and Entertainment":             { icon: "theater_comedy", bg: "bg-purple-50",  badge: "bg-purple-100 text-purple-800" },
  "Business and Professional Services": { icon: "business_center", bg: "bg-slate-50",   badge: "bg-slate-100 text-slate-700"  },
  "Community and Government":           { icon: "account_balance", bg: "bg-blue-50",    badge: "bg-blue-100 text-blue-800"    },
  "Dining and Drinking":                { icon: "restaurant", bg: "bg-orange-50",  badge: "bg-orange-100 text-orange-800"},
  "Event":                              { icon: "event", bg: "bg-yellow-50",  badge: "bg-yellow-100 text-yellow-800"},
  "Health and Medicine":                { icon: "local_hospital", bg: "bg-red-50",     badge: "bg-red-100 text-red-700"      },
  "Landmarks and Outdoors":             { icon: "park", bg: "bg-green-50",   badge: "bg-green-100 text-green-800"  },
  "Nightlife Spot":                     { icon: "nightlife", bg: "bg-indigo-50",  badge: "bg-indigo-100 text-indigo-800"},
  "Retail":                             { icon: "shopping_bag", bg: "bg-pink-50",    badge: "bg-pink-100 text-pink-800"    },
  "Sports and Recreation":              { icon: "sports_soccer", bg: "bg-teal-50",    badge: "bg-teal-100 text-teal-800"    },
  "Travel and Transportation":          { icon: "flight", bg: "bg-sky-50",     badge: "bg-sky-100 text-sky-800"      },
}

const FALLBACK_CONFIG = { icon: "home_work", bg: "bg-amber-50", badge: "bg-amber-100 text-amber-800" } // Everyday / non-tourist places

const level1Map = categoryLevel1Map as Record<string, string>

function getCategoryConfig(categoryName: string) {
  const level1 = level1Map[categoryName]
  return LEVEL1_CONFIG[level1 as keyof typeof LEVEL1_CONFIG] ?? FALLBACK_CONFIG
}

// Hilfskomponente, um die Karte so zu zentrieren, dass alle Marker sichtbar sind
function MapBoundsUpdater({ positions }: { positions: [number, number][] }) {
  const map = useMap();
  useEffect(() => {
    if (!positions || positions.length === 0) return;
    
    if (positions.length === 1) {
      map.setView(positions[0], 14);
    } else {
      const bounds = L.latLngBounds(positions);
      map.fitBounds(bounds, {
        paddingTopLeft: [50, 80],      // Platz für Searchbar & Timepicker oben
        paddingBottomRight: [50, 160], // Platz für Navigation unten
        maxZoom: 16,
        animate: true
      });
    }
  }, [positions, map]);
  return null;
}

// Komponente, die fitBounds über ein Ref nach außen exponiert
interface MapFitBoundsControlHandle {
  fitAllMarkers: () => void;
}

const MapFitBoundsControl = forwardRef<MapFitBoundsControlHandle, { positions: [number, number][] }>(
  ({ positions }, ref) => {
    const map = useMap();
    useImperativeHandle(ref, () => ({
      fitAllMarkers: () => {
        if (!positions || positions.length === 0) return;
        if (positions.length === 1) {
          map.setView(positions[0], 14, { animate: true });
        } else {
          const bounds = L.latLngBounds(positions);
          map.fitBounds(bounds, {
            paddingTopLeft: [50, 80],
            paddingBottomRight: [50, 160],
            maxZoom: 16,
            animate: true
          });
        }
      }
    }), [map, positions]);
    return null;
  }
);

export function MapView() {
  const navigate = useNavigate();
  const fitBoundsRef = useRef<MapFitBoundsControlHandle>(null);
  const { triggerNotification } = useNotification();
  
  // Neuer State für den User-Index, um durch die User zu navigieren
  const [userIndex, setUserIndex] = useState<number>(0);

  // Neuer State für die exakte Uhrzeit
  const [currentTime, setCurrentTime] = useState(new Date());
  
  // State für die simulierte User Location (Default: NYC Zentrum)
  const [userPos, setUserPos] = useState<[number, number]>([40.7128, -74.0060]);
  const [actualCheckinPos, setActualCheckinPos] = useState<[number, number] | null>(null);
  const [actualCheckinName, setActualCheckinName] = useState<string>("Unknown Place");
  const timeSyncedUserIndexRef = useRef<number | null>(null);

  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [foursquarePlaces, setFoursquarePlaces] = useState<VenuePlace[]>([]);
  const [loading, setLoading] = useState(true);
  const [recommendationModel, setRecommendationModel] = useState<"simple" | "federated">("simple");

  // Lese Modellauswahl aus localStorage (wird in PrivacySettings gesetzt)
  useEffect(() => {
    const savedModel = localStorage.getItem("recommendationModel");
    if (savedModel === "simple" || savedModel === "federated") {
      setRecommendationModel(savedModel);
    }

    // Auf Änderungen in anderen Tabs/Komponenten reagieren
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === "recommendationModel" && (e.newValue === "simple" || e.newValue === "federated")) {
        setRecommendationModel(e.newValue);
      }
    };
    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, []);

  // Berechnet die gerundete Stunde für die API (ab :30 aufrunden, sonst abrunden)
  const roundedHour = useMemo(() => {
    const minutes = currentTime.getMinutes();
    let hour = currentTime.getHours();
    
    if (minutes >= 30) {
      hour = (hour + 1) % 24; // Modulo 24, damit aus 23:30 -> 0 Uhr wird
    }
    return hour;
  }, [currentTime]);

  // Berechne alle Marker-Positionen, um die Karte darauf zu zentrieren
  const mapPositions = useMemo(() => {
    const positions: [number, number][] = [userPos];
    if (actualCheckinPos) {
      positions.push(actualCheckinPos);
    }
    foursquarePlaces.forEach((place) => {
      positions.push([place.lat, place.lng]);
    });
    return positions;
  }, [userPos, actualCheckinPos, foursquarePlaces]);

  useEffect(() => {
    // Inject Leaflet CSS if not already present
    if (!document.getElementById("leaflet-css")) {
      const link = document.createElement("link");
      link.id = "leaflet-css";
      link.rel = "stylesheet";
      link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
      document.head.appendChild(link);
    }

    // Inject Google Fonts if not already present
    if (!document.getElementById("google-fonts")) {
      const link = document.createElement("link");
      link.id = "google-fonts";
      link.rel = "stylesheet";
      link.href = "https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200";
      document.head.appendChild(link);
    }

    // Fetch recommendations from backend - nutzt jetzt roundedHour
    const fetchRecommendations = async () => {
      try {
        setLoading(true);
        setFoursquarePlaces([]); // Alte Orte löschen, bevor neue geladen werden!
        
        if (recommendationModel === "federated") {
          // ---- Federated Learning (FedKG) Model ----
          const response = await fetch(`http://localhost:8000/api/fedkg/predict?user_index=${userIndex}&topk=10`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();
          console.log("📍 FedKG Response:", data);

          if (data.error) {
            console.error("❌ FedKG Error:", data.error);
            setRecommendations([]);
            return;
          }

          // FedKG-Predictions haben bereits lat/lng → direkt als Marker verwenden
          const places: VenuePlace[] = [];
          if (data.predictions) {
            for (const pred of data.predictions) {
              if (pred.latitude && pred.longitude) {
                places.push({
                  id: String(pred.poi_id),
                  name: pred.category || `POI #${pred.poi_id}`,
                  lat: pred.latitude,
                  lng: pred.longitude,
                  categories: [pred.category || "Unknown"],
                });
              }
            }
          }
          console.log("🎯 FedKG Predictions als Marker:", places);

          // User-Position = letzter Check-in der Input-Sequence (wo der User gerade ist)
          if (data.last_checkin && data.last_checkin.latitude && data.last_checkin.longitude) {
            setUserPos([data.last_checkin.latitude, data.last_checkin.longitude]);
          }

          // Ground truth = tatsächlicher nächster Check-in (Vorhersage-Ziel)
          if (data.ground_truth && data.ground_truth.latitude && data.ground_truth.longitude) {
            const gt = data.ground_truth;
            setActualCheckinPos([gt.latitude, gt.longitude]);
            setActualCheckinName(gt.category || "Unknown Place");
            
            // Zeit auf den Test-Check-in (Ground Truth) synchronisieren
            if (gt.local_time && timeSyncedUserIndexRef.current !== userIndex) {
              setCurrentTime(new Date(gt.local_time));
              timeSyncedUserIndexRef.current = userIndex;
            }
          }

          // Direkt Marker setzen, kein Foursquare-Search nötig
          setFoursquarePlaces(places);
          setRecommendations([]); // Leer lassen → Foursquare-Search wird übersprungen
          
        } else {
          // ---- Simple (Cluster-based) Model ----
          const response = await fetch(`http://localhost:8000/api/example-user-recommendations?user_index=${userIndex}&hour=${roundedHour}`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();
          console.log("📍 Backend Response:", data);
          console.log("🎯 Top Kategorien:", data.recommendations["top_categories"]);
          
          if (data.location) {
            const lat = data.location.lat;
            const lng = data.location.lng;
            setActualCheckinPos([lat, lng]);
            setUserPos([lat - 0.0035, lng - 0.0035]); // Offset für die User-Position
            setActualCheckinName(data.location.name || "Unknown Place");
            
            // Zeit auf den echten Check-in synchronisieren (nur einmalig pro geladenem User)
            if (data.location.local_time && timeSyncedUserIndexRef.current !== userIndex) {
              setCurrentTime(new Date(data.location.local_time));
              timeSyncedUserIndexRef.current = userIndex;
            }
          }
          
          setRecommendations(data.recommendations["top_categories"] || []);
        }
      } catch (error) {
        console.error("❌ Fehler beim Abrufen der Empfehlungen:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
  }, [roundedHour, userIndex, recommendationModel]); // Abhängigkeit geändert auf roundedHour, userIndex und recommendationModel

  // Sucht lokale Orte basierend auf den erhaltenen Kategorien (nur für Simple Model)
  useEffect(() => {
    if (recommendations.length === 0 || recommendationModel === "federated") return;
    
    let isActive = true;

    const fetchFoursquarePlaces = async () => {
      const fetchedPlaces: VenuePlace[] = [];

      for (let index = 0; index < recommendations.length; index++) {
        if (!isActive) break;
        
        const category = recommendations[index];
        
        try {
          if (index > 0) {
            await new Promise(resolve => setTimeout(resolve, 500));
          }
          
          const response = await fetch(
            `http://localhost:8000/api/venues/search?query=${encodeURIComponent(category)}&lat=${userPos[0]}&lng=${userPos[1]}&radius=750&limit=2`,
            {
              method: "GET",
              headers: { Accept: "application/json" },
            }
          );

          if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
          
          const result = await response.json();
          
          // HIER GEÄNDERT: Schleife über alle gefundenen Orte (bis zu 3)
          if (isActive && result.results && result.results.length > 0) {
            result.results.forEach((place: any) => {
              const existingPlace = fetchedPlaces.find(p => p.id === place.id);
              
              if (existingPlace) {
                // Ort ist schon da! Füge nur die neue Kategorie zum Array hinzu
                if (!existingPlace.categories.includes(category)) {
                  existingPlace.categories.push(category);
                }
              } else {
                fetchedPlaces.push({
                  id: place.id,
                  name: place.name,
                  lat: place.lat,
                  lng: place.lng,
                  categories: [category]
                });
              }
            });
          }
        } catch (error) {
          console.error(`❌ Fehler beim Abrufen der Venues für '${category}':`, error);
        }
      }

      if (isActive) {
        setFoursquarePlaces(fetchedPlaces);
      }
    };

    fetchFoursquarePlaces();
    
    return () => {
      isActive = false;
    };
  }, [recommendations, userPos]);

  // Hilfsfunktion zum Formatieren der Zeit für den Input
  const timeString = `${currentTime.getHours().toString().padStart(2, '0')}:${currentTime.getMinutes().toString().padStart(2, '0')}`;

  // Prüfen, ob der tatsächliche Check-in in den vorhergesagten Orten enthalten ist
  const isPlaceMatch = (placeLat: number, placeLng: number) => {
    if (!actualCheckinPos) return false;
    // Toleranz von ca. 10 Metern
    return Math.abs(placeLat - actualCheckinPos[0]) < 0.0001 && Math.abs(placeLng - actualCheckinPos[1]) < 0.0001;
  };
  const hasMatchedPrediction = actualCheckinPos && foursquarePlaces.some(p => isPlaceMatch(p.lat, p.lng));

  return (
    <div className="relative w-full h-full bg-slate-50">
      
      {/* Interactive Map */}
      <div className="absolute inset-0 z-0 [&_.leaflet-control-attribution]:hidden">
        <MapContainer 
          center={userPos} 
          zoom={14} 
          zoomControl={false}
          style={{ height: "100%", width: "100%", zIndex: 0 }}
        >
          <MapBoundsUpdater positions={mapPositions} />
          <MapFitBoundsControl ref={fitBoundsRef} positions={mapPositions} />
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          />
          
          {/* User Marker (Simulierte aktuelle Position) */}
          <Marker position={userPos} icon={createUserIcon()}>
            <Popup className="rounded-xl overflow-hidden shadow-xl border-none">
              <div className="text-center font-semibold text-gray-900 py-1">You are here<br/><span className="font-normal text-xs text-gray-500">Simulated Location</span></div>
            </Popup>
          </Marker>

          {/* Actual Checkin Marker */}
          {actualCheckinPos && !hasMatchedPrediction && (
            <Marker position={actualCheckinPos} icon={createActualCheckinIcon()}>
              <Popup className="rounded-xl overflow-hidden shadow-xl border-none">
                <div className="text-center font-semibold text-gray-900 py-1">Actual Next Check-in<br/><span className="font-normal text-xs text-gray-500">{actualCheckinName}</span></div>
              </Popup>
            </Marker>
          )}

          {/* Dynamische Foursquare Marker */}
          {!loading && foursquarePlaces.map((place, index) => {
          const mainCategory = place.categories[0]; // Das Pin-Icon basiert auf der wichtigsten/ersten Kategorie
          const cfg = getCategoryConfig(mainCategory);
          const isMatch = isPlaceMatch(place.lat, place.lng);
          return (
            <Marker
              key={`${place.id}-${index}`}
              position={[place.lat, place.lng]}
              icon={createRecommendationIcon(mainCategory, isMatch)}
            >
              <Popup className="!p-0 border-none overflow-hidden rounded-xl shadow-lg m-0 w-[200px]">
                <div className="flex flex-col">
                  <div className={`flex items-center justify-center h-16 ${isMatch ? 'bg-green-50' : cfg.bg}`}>
                    <span className={`material-symbols-outlined text-[48px] ${isMatch ? 'text-green-600' : 'text-gray-500'}`}>{cfg.icon}</span>
                  </div>
                  <div className="p-3 bg-white">
                    <h3 className="font-semibold text-gray-900 text-sm leading-tight mb-2 flex items-center gap-1.5">
                      {place.name}
                      {isMatch && <span className="material-symbols-outlined text-green-500 text-[16px]" title="Correct Prediction!">check_circle</span>}
                    </h3>
                    <div className="flex flex-wrap items-center gap-1.5">
                      {/* Render alle zutreffenden Kategorien als kleine Badges */}
                      {place.categories.map((cat, i) => {
                        const catCfg = getCategoryConfig(cat);
                        return (
                          <span key={i} className={`inline-flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded-full ${catCfg.badge}`}>
                            <span className="material-symbols-outlined text-[12px]">{catCfg.icon}</span> {cat}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </Popup>
            </Marker>
          )
        })}

        </MapContainer>
      </div>

      {/* Header / Search Bar */}
      <div className="absolute top-12 inset-x-4 flex gap-2 z-[1000] pointer-events-none">
        <div className="flex-1 bg-white/90 backdrop-blur-md rounded-full px-4 py-3 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.1)] flex items-center gap-3 pointer-events-auto border border-gray-100/50">
          <MapPin className="w-5 h-5 text-gray-400" />
          <input 
            type="text" 
            placeholder="Search places in NYC..." 
            defaultValue="New York City"
            className="bg-transparent border-none outline-none w-full text-sm font-medium placeholder:text-gray-400"
          />
        </div>
        
        {/* Interactive Time Picker Display */}
        <div className="bg-white/90 backdrop-blur-md rounded-full px-4 py-3 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.1)] flex items-center gap-2 pointer-events-auto border border-gray-100/50 hover:bg-white transition-colors focus-within:ring-2 focus-within:ring-blue-500/50">
          <Clock className="w-5 h-5 text-blue-600" />
          <input
            type="time"
            value={timeString}
            onChange={(e) => {
              if (!e.target.value) return;
              const [hours, minutes] = e.target.value.split(':');
              const newDate = new Date(currentTime);
              newDate.setHours(parseInt(hours, 10));
              newDate.setMinutes(parseInt(minutes, 10));
              setCurrentTime(newDate);
            }}
            className="bg-transparent border-none outline-none text-sm font-bold text-gray-900 cursor-pointer w-[68px] [&::-webkit-calendar-picker-indicator]:cursor-pointer [&::-webkit-calendar-picker-indicator]:opacity-0 [&::-webkit-calendar-picker-indicator]:absolute [&::-webkit-calendar-picker-indicator]:w-full"
            title="Set simulation time"
          />
        </div>
      </div>

      {/* Bottom Navigation */}
      <BottomNavBar onCenterAction={() => fitBoundsRef.current?.fitAllMarkers()} />
      
      {/* Dev / Prototype Controls */}
      <div className="absolute bottom-32 right-4 flex flex-col gap-2 z-[1000]">
        <button 
          onClick={() => setUserIndex(prev => prev + 1)}
          className="w-12 h-12 bg-white/90 backdrop-blur-md rounded-full shadow-[0_4px_12px_rgba(0,0,0,0.1)] border border-gray-100 flex items-center justify-center text-blue-600 hover:bg-white active:scale-95 transition-all"
          title="Next Example User"
        >
          <SkipForward className="w-5 h-5" />
        </button>
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