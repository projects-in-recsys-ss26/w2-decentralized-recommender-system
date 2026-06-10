import { motion, AnimatePresence } from 'motion/react';
import { useNotification } from './NotificationContext';
import { Coffee } from 'lucide-react';

export function NotificationBanner() {
  const { notification, clearNotification } = useNotification();

  return (
    <div className="absolute top-2.5 left-0 right-0 z-[10000] px-2 pointer-events-none flex justify-center">
      <AnimatePresence>
        {notification && (
          <motion.div
            initial={{ y: -60, scale: 0.9, opacity: 0 }}
            animate={{ y: 0, scale: 1, opacity: 1 }}
            exit={{ y: -60, scale: 0.9, opacity: 0 }}
            transition={{ type: "spring", stiffness: 400, damping: 30 }}
            className="w-full max-w-[361px] bg-white/70 backdrop-blur-xl border border-white/50 shadow-[0_8px_30px_rgb(0,0,0,0.12)] rounded-[24px] p-3 pointer-events-auto cursor-pointer"
            onClick={clearNotification}
          >
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 rounded-[10px] bg-amber-600 flex flex-shrink-0 items-center justify-center text-white shadow-sm">
                {notification.icon || <Coffee className="w-6 h-6" />}
              </div>
              <div className="flex-1 min-w-0 pt-0.5">
                <div className="flex justify-between items-center mb-0.5">
                  <h4 className="font-semibold text-[15px] text-gray-900 leading-none">{notification.title}</h4>
                  <span className="text-xs text-gray-500 font-medium">{notification.time || 'now'}</span>
                </div>
                <p className="text-[14px] text-gray-700 leading-tight">{notification.message}</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}