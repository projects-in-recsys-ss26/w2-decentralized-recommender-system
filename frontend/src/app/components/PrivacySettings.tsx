import React, { useState, useEffect } from "react";
import { ChevronLeft, Shield, Eye, Map, Share2, Activity, Brain, Network } from "lucide-react";
import { useNavigate } from "react-router";
import { Switch } from "./ui/switch";
import { BottomNavBar } from "./BottomNavBar";
import { PlaySquare } from "lucide-react";

export function PrivacySettings() {
  const navigate = useNavigate();

  // States
  const [shareLocation, setShareLocation] = useState(true);
  const [sharePersonal, setSharePersonal] = useState(false);
  const [shareExactCheckins, setShareExactCheckins] = useState(false);
  
  type ModalType = "location" | "personal" | "checkins" | null;
  const [activeModal, setActiveModal] = useState<ModalType>(null);
  
  const [recommendationModel, setRecommendationModel] = useState<"simple" | "federated">("simple");
  const [demoMode, setDemoMode] = useState(false);

  useEffect(() => {
    // Beim Laden der Komponente den gespeicherten Wert abrufen
    const savedLoc = localStorage.getItem("shareLocation");
    if (savedLoc !== null) setShareLocation(savedLoc === "true");

    const savedPers = localStorage.getItem("sharePersonal");
    if (savedPers !== null) setSharePersonal(savedPers === "true");

    const savedCheckins = localStorage.getItem("shareExactCheckins");
    if (savedCheckins !== null) setShareExactCheckins(savedCheckins === "true");

    const savedModel = localStorage.getItem("recommendationModel");
    if (savedModel === "simple" || savedModel === "federated") {
      setRecommendationModel(savedModel);
    }
    const savedDemo = localStorage.getItem("demoMode");
    if (savedDemo !== null) {
      setDemoMode(savedDemo === "true");
    }
  }, []);

  const handleLocationToggle = (checked: boolean) => {
    if (!checked) {
      setActiveModal("location");
    } else {
      setShareLocation(true);
      localStorage.setItem("shareLocation", "true");
    }
  };

  const handlePersonalToggle = (checked: boolean) => {
    if (!checked) {
      setActiveModal("personal");
    } else {
      setSharePersonal(true);
      localStorage.setItem("sharePersonal", "true");
    }
  };

  const handleCheckinsToggle = (checked: boolean) => {
    if (!checked) {
      setActiveModal("checkins");
    } else {
      setShareExactCheckins(true);
      localStorage.setItem("shareExactCheckins", "true");
    }
  };

  const handleDemoModeToggle = (checked: boolean) => {
    setDemoMode(checked);
    localStorage.setItem("demoMode", checked.toString());
  };

  const confirmTurnOff = () => {
    if (activeModal === "location") {
      setShareLocation(false);
      localStorage.setItem("shareLocation", "false");
    } else if (activeModal === "personal") {
      setSharePersonal(false);
      localStorage.setItem("sharePersonal", "false");
    } else if (activeModal === "checkins") {
      setShareExactCheckins(false);
      localStorage.setItem("shareExactCheckins", "false");
    }
    setActiveModal(null);
  };

  const cancelTurnOff = () => {
    setActiveModal(null);
  };

  const getModalContent = () => {
    switch (activeModal) {
      case "location":
        return {
          title: "Turn off location sharing?",
          text: "Please note that disabling background location access will prevent the app from sending you timely recommendations and notifications while you are not actively using it.",
          confirmBtn: "Turn off anyways",
          cancelBtn: "Keep sharing"
        };
      case "personal":
        return {
          title: "Stop sharing personal info?",
          text: "By withholding your personal information, you limit our ability to improve the recommendation algorithms for the broader community. However, the quality of your own recommendations will remain unaffected.",
          confirmBtn: "Turn off anyways",
          cancelBtn: "Keep sharing"
        };
      case "checkins":
        return {
          title: "Turn off exact check-ins?",
          text: "Disabling exact check-in sharing will not prevent you from receiving notifications, but it may significantly reduce the accuracy and relevance of your personalized recommendations.",
          confirmBtn: "Turn off anyways",
          cancelBtn: "Keep sharing"
        };
      default:
        return null;
    }
  };

  const modalContent = getModalContent();

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
      <div className="flex-1 overflow-y-auto px-6 py-6 pb-28 [&::-webkit-scrollbar]:hidden" style={{ scrollbarWidth: 'none' }}>
        
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
                <div className="flex items-start gap-4">
                  <Map className="w-5 h-5 text-gray-400 mt-0.5 shrink-0" />
                  <div>
                    <p className="font-medium text-gray-900">Background Location</p>
                    <p className="text-xs text-gray-500 mt-0.5">Allow app to track your position in the background</p>
                  </div>
                </div>
                <Switch 
                  checked={shareLocation} 
                  onCheckedChange={handleLocationToggle} 
                />
              </div>

              <div className="h-px w-full bg-gray-100" />

              <div className="flex items-center justify-between">
                <div className="flex items-start gap-4">
                  <Eye className="w-5 h-5 text-gray-400 mt-0.5 shrink-0" />
                  <div>
                    <p className="font-medium text-gray-900">Personal Information</p>
                    <p className="text-xs text-gray-500 mt-0.5">Share details like age, sex, and more</p>
                  </div>
                </div>
                <Switch 
                  checked={sharePersonal}
                  onCheckedChange={handlePersonalToggle} 
                />
              </div>

              <div className="h-px w-full bg-gray-100" />

              <div className="flex items-center justify-between">
                <div className="flex items-start gap-4">
                  <Activity className="w-5 h-5 text-gray-400 mt-0.5 shrink-0" />
                  <div>
                    <p className="font-medium text-gray-900">Share Exact Check-ins</p>
                    <p className="text-xs text-gray-500 mt-0.5">Allow saving your precise check-in locations instead of just the venue categories.</p>
                  </div>
                </div>
                <Switch 
                  checked={shareExactCheckins}
                  onCheckedChange={handleCheckinsToggle} 
                />
              </div>

            </div>
          </div>

          {/* Recommendation Model Setting Group */}
          <div className="mt-6">
            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3 ml-1">Recommendation Model</h3>
            <div className="bg-white rounded-2xl p-1 shadow-sm border border-gray-100 space-y-0.5">
              
              <button 
                onClick={() => { setRecommendationModel("simple"); localStorage.setItem("recommendationModel", "simple"); }}
                className={`w-full flex items-center justify-between gap-4 rounded-xl p-3 transition-colors ${
                  recommendationModel === "simple" ? "bg-blue-50 ring-2 ring-blue-500" : "hover:bg-gray-50"
                }`}
              >
                <div className="flex items-start gap-4">
                  <Brain className={`w-5 h-5 mt-0.5 shrink-0 ${recommendationModel === "simple" ? "text-blue-600" : "text-gray-400"}`} />
                  <div className="text-left">
                    <p className={`font-medium ${recommendationModel === "simple" ? "text-blue-900" : "text-gray-900"}`}>Category Gossip</p>
                    <p className="text-xs text-gray-500 mt-0.5">Cluster-based recommendations using time &amp; category patterns</p>
                  </div>
                </div>
                <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${
                  recommendationModel === "simple" ? "border-blue-600" : "border-gray-300"
                }`}>
                  {recommendationModel === "simple" && <div className="w-2.5 h-2.5 rounded-full bg-blue-600" />}
                </div>
              </button>

              <div className="h-px mx-3 bg-gray-100" />

              <button 
                onClick={() => { setRecommendationModel("federated"); localStorage.setItem("recommendationModel", "federated"); }}
                className={`w-full flex items-center justify-between gap-4 rounded-xl p-3 transition-colors ${
                  recommendationModel === "federated" ? "bg-blue-50 ring-2 ring-blue-500" : "hover:bg-gray-50"
                }`}
              >
                <div className="flex items-start gap-4">
                  <Network className={`w-5 h-5 mt-0.5 shrink-0 ${recommendationModel === "federated" ? "text-blue-600" : "text-gray-400"}`} />
                  <div className="text-left">
                    <p className={`font-medium ${recommendationModel === "federated" ? "text-blue-900" : "text-gray-900"}`}>Federated Learning</p>
                    <p className="text-xs text-gray-500 mt-0.5">Privacy-preserving FedKG model with sequential POI prediction</p>
                  </div>
                </div>
                <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${
                  recommendationModel === "federated" ? "border-blue-600" : "border-gray-300"
                }`}>
                  {recommendationModel === "federated" && <div className="w-2.5 h-2.5 rounded-full bg-blue-600" />}
                </div>
              </button>

            </div>
          </div>

          {/* App Settings Group */}
          <div className="mt-6">
            <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3 ml-1">App Settings</h3>
            <div className="bg-white rounded-2xl p-4 shadow-sm border border-gray-100 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-start gap-4">
                  <PlaySquare className="w-5 h-5 text-gray-400 mt-0.5 shrink-0" />
                  <div>
                    <p className="font-medium text-gray-900">Demo Mode</p>
                    <p className="text-xs text-gray-500 mt-0.5">Showcase specific example predictions in MapView</p>
                  </div>
                </div>
                <Switch 
                  checked={demoMode} 
                  onCheckedChange={handleDemoModeToggle} 
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Navigation */}
      <BottomNavBar />

      {/* Google-Style Custom Alert Modal */}
      {activeModal && modalContent && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 animate-in fade-in duration-200">
          <div className="bg-white rounded-3xl p-6 w-[82%] max-w-[320px] shadow-2xl">
            <h3 className="text-[1.15rem] font-medium text-gray-900 mb-3">
              {modalContent.title}
            </h3>
            <p className="text-sm text-gray-600 mb-6 leading-relaxed">
              {modalContent.text}
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={confirmTurnOff}
                className="px-4 py-2.5 rounded-full text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 transition-colors"
              >
                {modalContent.confirmBtn}
              </button>
              <button
                onClick={cancelTurnOff}
                className="px-4 py-2.5 rounded-full text-sm font-medium text-white bg-[#1a73e8] hover:bg-blue-700 transition-colors"
              >
                {modalContent.cancelBtn}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}