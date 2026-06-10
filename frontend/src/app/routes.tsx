import { createBrowserRouter, Outlet } from "react-router";
import { MapView } from "./components/MapView";
import { PrivacySettings } from "./components/PrivacySettings";
import { PhoneLayout } from "./components/PhoneLayout";
import { LockScreen } from "./components/LockScreen";

function Root() {
  return (
    <PhoneLayout>
      <Outlet />
    </PhoneLayout>
  );
}

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Root,
    children: [
      {
        index: true,
        Component: MapView,
      },
      {
        path: "settings",
        Component: PrivacySettings,
      },
      {
        path: "lock",
        Component: LockScreen,
      }
    ]
  }
]);