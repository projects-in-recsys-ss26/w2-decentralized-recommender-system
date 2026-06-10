import React from "react";
import { NotificationBanner } from "./NotificationBanner";

export function PhoneLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-neutral-900 flex items-center justify-center p-4 sm:p-8">
      {/* iPhone 16 Style Outer Frame */}
      <div className="relative w-full max-w-[393px] aspect-[393/852] bg-zinc-950 rounded-[55px] shadow-[0_0_0_1px_#444,0_0_0_5px_#111,0_25px_50px_-12px_rgba(0,0,0,1)] flex items-center justify-center flex-shrink-0">

        {/* Action Button (Left Top) */}
        <div className="absolute left-[-5px] top-[110px] w-[5px] h-[30px] bg-zinc-800 rounded-l-lg shadow-inner"></div>
        {/* Volume Up */}
        <div className="absolute left-[-5px] top-[160px] w-[5px] h-[60px] bg-zinc-800 rounded-l-lg shadow-inner"></div>
        {/* Volume Down */}
        <div className="absolute left-[-5px] top-[230px] w-[5px] h-[60px] bg-zinc-800 rounded-l-lg shadow-inner"></div>

        {/* Power Button (Right) */}
        <div className="absolute right-[-5px] top-[180px] w-[5px] h-[90px] bg-zinc-800 rounded-r-lg shadow-inner"></div>

        {/* Screen */}
        <div className="relative w-[calc(100%-14px)] h-[calc(100%-14px)] bg-black rounded-[48px] overflow-hidden isolate border border-black">
          
          {/* Dynamic Island */}
          <div className="absolute top-2.5 inset-x-0 h-9 flex justify-center z-[9998]">
            <div className="w-[124px] h-[34px] bg-black rounded-[20px] shadow-sm flex items-center justify-end px-3 gap-2">
               {/* Sensors / Camera simulation */}
               <div className="w-3 h-3 bg-zinc-900 rounded-full border border-zinc-800"></div>
               <div className="w-2.5 h-2.5 bg-zinc-950 rounded-full border border-zinc-800 flex items-center justify-center">
                  <div className="w-1 h-1 bg-green-500 rounded-full shadow-[0_0_4px_#22c55e]"></div>
               </div>
            </div>
          </div>
          
          {/* iOS Notification Banner */}
          <NotificationBanner />
          
          {/* Screen Content */}
          <div className="relative w-full h-full bg-white overflow-hidden">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}