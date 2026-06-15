import React, { useState, useEffect } from "react";
import { ChevronLeft, Shield, Eye, Map, Share2, Activity } from "lucide-react";
import { useNavigate } from "react-router";
import { Switch } from "./ui/switch";

export function PrivacySettings() {
  const navigate = useNavigate();

  // States
  const [shareLocation, setShareLocation] = useState(true);
  const [showWarningModal, setShowWarningModal] = useState(false);

  useEffect(() => {
    // Beim Laden der Komponente den gespeicherten Wert abrufen
    const saved = localStorage.getItem("shareLocation");
    if (saved !== null) {
      setShareLocation(saved === "true");
    }
  }, []);

  const handleLocationToggle = (checked: boolean) => {
    if (!checked) {
      // Wenn der User es AUSSCHALTEN will -> Modal anzeigen, State noch nicht ändern
      setShowWarningModal(true);
    } else {
      // Wenn der User es EINSCHALTEN will -> Direkt ändern
      setShareLocation(true);
      localStorage.setItem("shareLocation", "true");
    }
  };

  // Wird aufgerufen, wenn im Modal "Turn off anyways" geklickt wird
  const confirmTurnOff = () => {
    setShareLocation(false);
    localStorage.setItem("shareLocation", "false");
    setShowWarningModal(false);
  };

  // Wird aufgerufen, wenn im Modal "Share location" geklickt wird
  const cancelTurnOff = () => {
    setShowWarningModal(false);
    // Switch bleibt auf "true", da der State nicht geändert wurde
  };

  return (
    <div className="w-full h-full bg-gray-50 flex flex-col relative">
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
                    <p className="font-medium text-gray-900">Share Exact Check-ins</p>
                    <p className="text-xs text-gray-500 mt-0.5">Allow saving your precise check-in locations instead of just the venue categories.</p>
                  </div>
                </div>
                <Switch defaultChecked={false} />
              </div>

            </div>
          </div>
        </div>
      </div>

      {/* Google-Style Custom Alert Modal */}
      {showWarningModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4 animate-in fade-in duration-200">
          <div className="bg-white rounded-2xl p-6 max-w-sm w-full shadow-2xl">
            <h3 className="text-[1.15rem] font-medium text-gray-900 mb-3">
              Turn off location sharing?
            </h3>
            <p className="text-sm text-gray-600 mb-6 leading-relaxed">
              Attention: Without sharing your location with our server, your device must perform location searches itself, potentially increasing your mobile data usage.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={confirmTurnOff}
                className="px-4 py-2.5 rounded-full text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 transition-colors"
              >
                Turn off anyways
              </button>
              <button
                onClick={cancelTurnOff}
                className="px-4 py-2.5 rounded-full text-sm font-medium text-white bg-[#1a73e8] hover:bg-blue-700 transition-colors"
              >
                Share location
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}