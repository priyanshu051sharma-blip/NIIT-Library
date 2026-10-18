"use client";

import { createContext, ReactNode, useContext, useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type CameraPurpose = "library" | "discussion" | "entrance";

export type OccupancyCounts = {
  people_count: number;
  occupied_seats: number;
  empty_seats: number;
  total_seats: number;
  occupancy_percentage: number;
  entries: number;
  exits: number;
  mode: string;
  connected_cameras: number;
  detected_cameras: number;
};

export type CameraFeed = {
  id: string;
  name: string;
  channel: number | null;
  purpose: CameraPurpose;
  status: string;
  stream_running: boolean;
  last_frame_at: string | null;
  last_detection_at: string | null;
  reconnect_attempts: number;
  error: string | null;
  seat_roi: [number, number][] | null;
  calibrated: boolean;
  occupancy: {
    people_count: number;
    occupied_seats: number;
    empty_seats: number;
    occupancy_percentage: number;
    entries: number;
    exits: number;
    mode: string;
  };
};

export type OccupancySnapshot = OccupancyCounts & {
  timestamp: string;
  status: string;
  privacy: string;
  camera_groups: Record<CameraPurpose, OccupancyCounts & { purpose: CameraPurpose }>;
  cameras: CameraFeed[];
};

type OccupancyContextValue = {
  snapshot: OccupancySnapshot | null;
  error: string | null;
};

const OccupancyContext = createContext<OccupancyContextValue | null>(null);

export function OccupancyProvider({ children }: { children: ReactNode }) {
  const [snapshot, setSnapshot] = useState<OccupancySnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!window.location.pathname.startsWith("/student/") && !window.location.pathname.startsWith("/admin/")) return;
    let active = true;
    let requestInProgress = false;

    const refresh = async () => {
      if (requestInProgress) return;
      requestInProgress = true;
      try {
        const response = await fetch(`${API}/api/occupancy/current`, {
          headers: { Authorization: "Bearer demo-student" },
          cache: "no-store",
        });
        if (!response.ok) throw new Error("Could not load shared live occupancy.");
        const nextSnapshot = (await response.json()) as OccupancySnapshot;
        if (active) {
          setSnapshot(nextSnapshot);
          setError(null);
        }
      } catch (requestError) {
        if (active) {
          setError(requestError instanceof Error ? requestError.message : "Could not load shared live occupancy.");
        }
      } finally {
        requestInProgress = false;
      }
    };

    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  return <OccupancyContext.Provider value={{ snapshot, error }}>{children}</OccupancyContext.Provider>;
}

export function useOccupancy() {
  const context = useContext(OccupancyContext);
  if (!context) throw new Error("useOccupancy must be used within an OccupancyProvider.");
  return context;
}
