import React from "react";
import { RouterProvider } from "react-router";
import { router } from "./routes";
import { NotificationProvider } from "./components/NotificationContext";

export default function App() {
  return (
    <NotificationProvider>
      <RouterProvider router={router} />
    </NotificationProvider>
  );
}