import React from "react";
import { MapPin, Navigation, Settings } from "lucide-react";
import { useNavigate, useLocation } from "react-router";

interface BottomNavBarProps {
  onCenterAction?: () => void;
}

export function BottomNavBar({ onCenterAction }: BottomNavBarProps) {
  const navigate = useNavigate();
  const location = useLocation();

  const isMap = location.pathname === "/";
  const isSettings = location.pathname === "/settings";

  return (
    <div className="absolute bottom-8 inset-x-6 z-[1000]">
      <div className="bg-zinc-900/95 backdrop-blur-lg text-white rounded-[32px] px-6 py-4 flex justify-between items-center shadow-2xl border border-zinc-800">
        <button
          className={`flex flex-col items-center gap-1 transition-colors ${isMap ? "text-white" : "text-gray-400 hover:text-white"}`}
          onClick={() => navigate("/")}
        >
          <MapPin className="w-6 h-6" />
        </button>

        <button
          className="w-14 h-14 bg-blue-600 rounded-full flex items-center justify-center shadow-[0_0_20px_rgba(37,99,235,0.5)] -mt-10 border-[6px] border-zinc-900 text-white transform transition-transform hover:scale-105 active:scale-95"
          onClick={() => {
            if (onCenterAction) {
              onCenterAction();
            } else {
              navigate("/");
            }
          }}
        >
          <Navigation className="w-6 h-6 -rotate-45 ml-[-2px] mt-[2px]" />
        </button>

        <button
          className={`flex flex-col items-center gap-1 transition-colors ${isSettings ? "text-white" : "text-gray-400 hover:text-white"}`}
          onClick={() => navigate("/settings")}
        >
          <Settings className="w-6 h-6" />
        </button>
      </div>
    </div>
  );
}
