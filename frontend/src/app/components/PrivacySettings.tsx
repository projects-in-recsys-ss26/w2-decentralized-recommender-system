import React, { useState, useEffect } from "react";
import { ChevronLeft, Shield, Eye, Map, Share2, Activity } from "lucide-react";
import { useNavigate } from "react-router";
import { Switch } from "./ui/Switch"; // Geht davon aus, dass dein Switch 'checked' und 'onCheckedChange' unterstützt

export function PrivacySettings() {
  const navigate = useNavigate();

  // State initialisieren: Standardmäßig an (true), es sei denn, es wurde explizit ausgeschaltet
  const [shareLocation, setShareLocation] = useState(true);

  useEffect(() => {
    // Beim Laden der Komponente den gespeicherten Wert abrufen
    const saved = localStorage.getItem("shareLocation");
    if (saved !== null) {
      setShareLocation(saved === "true");
    }
  }, []);

  const handleLocationToggle = (checked) => {
    // Wenn der User es AUSSCHALTEN will
    if (!checked) {
      const confirmDeactivation = window.confirm(
        "Achtung: Wenn du deinen Standort nicht mit unserem Server teilst, muss dein Smartphone die Suche nach Orten selbst übernehmen. Das kann deinen mobilen Datenverbrauch erhöhen. Möchtest du das wirklich?"
      );
      
      // Wenn der User im Alert auf "Abbrechen" drückt, brechen wir ab
      if (!confirmDeactivation) return;
    }

    // State aktualisieren und speichern
    setShareLocation(checked);
    localStorage.setItem("shareLocation", checked);
  };

  return (
    <div className="w-full h-full bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="pt-12 pb-4 px-6 bg-white flex items-center gap-4 shadow-sm z-10">
        <button 
          onClick={() => navigate('/')}
          className="p-2 hover:bg-gray-100 rounded-full transition-colors -ml-2 text-gray-700"
        >
          <ChevronLeft className="w-6 h-6" />
        </button>
        <h1 className="text-xl font-bold text-gray-900">Privacy & Data</h1>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-6 [&::-webkit-scrollbar]:hidden" style={{ scrollbarWidth: 'none' }}>
        
        <div className="flex items-center gap-3 mb-8">
          <div className="w-12 h-12 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center">
            <Shield className="w-6 h-6" />
          </div>
          <div>
            <h2 className="font-semibold text-gray-900 text-lg">Control Your Data</h2>
            <p className="text-sm text-gray-500">Decide what you share with others.</p>
          </div>
        </div>

        <div className="space-y-6">
          {/* Setting Group */}
          <div>
            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3 ml-1">Data Sharing</h3>
            <div className="bg-white rounded-2xl p-4 shadow-sm border border-gray-100 space-y-4">
              
              <div className="flex items-center justify-between">
                <div className="flex items-start gap-3">
                  <Map className="w-5 h-5 text-gray-400 mt-0.5" />
                  <div>
                    <p className="font-medium text-gray-900">Current Location</p>
                    <p className="text-xs text-gray-500 mt-0.5">Allow app to share your exact position</p>
                  </div>
                </div>
                {/* Switch an den State anbinden */}
                <Switch 
                  checked={shareLocation} 
                  onCheckedChange={handleLocationToggle} 
                />
              </div>

              <div className="h-px w-full bg-gray-100" />

              <div className="flex items-center justify-between">
                <div className="flex items-start gap-3">
                  <Eye className="w-5 h-5 text-gray-400 mt-0.5" />
                  <div>
                    <p className="font-medium text-gray-900">Personal Information</p>
                    <p className="text-xs text-gray-500 mt-0.5">Share details like age, sex, and more</p>
                  </div>
                </div>
                <Switch defaultChecked={false} />
              </div>

              <div className="h-px w-full bg-gray-100" />

              <div className="flex items-center justify-between">
                <div className="flex items-start gap-3">
                  <Activity className="w-5 h-5 text-gray-400 mt-0.5" />
                  <div>
                    <p className="font-medium text-gray-900">Location History</p>
                    <p className="text-xs text-gray-500 mt-0.5">Keep and share a record of places visited</p>
                  </div>
                </div>
                <Switch defaultChecked={false} />
              </div>

            </div>
          </div>

        </div>

      </div>
    </div>
  );
}