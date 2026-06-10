import { Bell, Lock, Camera, Flashlight } from "lucide-react";
import { useEffect, useState } from "react";
import { useNotification } from "./NotificationContext";
import { useNavigate } from "react-router";

export function LockScreen() {
  const { triggerNotification } = useNotification();
  const navigate = useNavigate();
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const handleSampleNotification = () => {
    triggerNotification({
      title: "MW Cafe",
      message: "You might like this",
      time: "now"
    });
  };

  return (
    <div className="relative w-full h-full bg-slate-900 overflow-hidden text-white">
      {/* Abstract Wallpaper Image */}
      <img 
        src="https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&q=80&w=400&h=850" 
        alt="Wallpaper"
        className="absolute inset-0 w-full h-full object-cover opacity-60 mix-blend-overlay"
      />
      
      {/* Top Section */}
      <div className="relative z-10 flex flex-col items-center pt-20">
        <Lock className="w-5 h-5 mb-2" />
        <div className="text-[80px] font-medium leading-none tracking-tighter mb-2 font-sans">
          {time.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' })}
        </div>
        <div className="text-lg font-medium opacity-80">
          {time.toLocaleDateString('de-DE', { weekday: 'long', day: 'numeric', month: 'long' })}
        </div>
      </div>

      {/* Development / Prototype Actions */}
      <div className="relative z-10 flex flex-col gap-3 items-center mt-20 px-8">
        <button 
          onClick={handleSampleNotification}
          className="w-full bg-white/20 hover:bg-white/30 backdrop-blur-md text-white font-medium py-3 px-6 rounded-2xl flex items-center justify-center gap-2 transition-all active:scale-95"
        >
          <Bell className="w-5 h-5" /> Trigger Notification
        </button>
        <button 
          onClick={() => navigate('/')}
          className="w-full bg-black/40 hover:bg-black/60 backdrop-blur-md text-white font-medium py-3 px-6 rounded-2xl transition-all active:scale-95"
        >
          Unlock (Go to Map)
        </button>
      </div>

      {/* Bottom Actions */}
      <div className="absolute bottom-12 inset-x-8 flex justify-between z-10">
        <button className="w-12 h-12 rounded-full bg-black/40 backdrop-blur-md flex items-center justify-center transition-all active:scale-90">
          <Flashlight className="w-5 h-5 text-white" />
        </button>
        <button className="w-12 h-12 rounded-full bg-black/40 backdrop-blur-md flex items-center justify-center transition-all active:scale-90">
          <Camera className="w-5 h-5 text-white" />
        </button>
      </div>
      
      {/* Swipe Bar */}
      <div className="absolute bottom-2 left-1/2 -translate-x-1/2 w-[130px] h-[5px] bg-white rounded-full z-10"></div>
    </div>
  );
}