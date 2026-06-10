import { createContext, useContext, useState, ReactNode, useEffect } from 'react';

type Notification = {
  id: number;
  title: string;
  message: string;
  icon?: ReactNode;
  time?: string;
};

type NotificationContextType = {
  notification: Notification | null;
  triggerNotification: (notif: Omit<Notification, 'id'>) => void;
  clearNotification: () => void;
};

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [notification, setNotification] = useState<Notification | null>(null);

  const triggerNotification = (notif: Omit<Notification, 'id'>) => {
    setNotification({ ...notif, id: Date.now() });
  };

  const clearNotification = () => setNotification(null);

  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => {
        clearNotification();
      }, 5000); // 5 seconds display
      return () => clearTimeout(timer);
    }
  }, [notification]);

  return (
    <NotificationContext.Provider value={{ notification, triggerNotification, clearNotification }}>
      {children}
    </NotificationContext.Provider>
  );
}

export function useNotification() {
  const context = useContext(NotificationContext);
  if (!context) throw new Error("useNotification must be used within NotificationProvider");
  return context;
}