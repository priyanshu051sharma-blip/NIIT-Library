import type { Metadata } from "next";
import { OccupancyProvider } from "./context/OccupancyContext";
import "./globals.css";

export const metadata: Metadata = { title: "SmartLib", description: "AI-powered library occupancy and space management" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body><OccupancyProvider>{children}</OccupancyProvider></body></html>; }
