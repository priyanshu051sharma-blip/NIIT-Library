"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type DetectionResult = {
  people_count: number;
  occupied_seats: number;
  empty_seats: number;
  occupancy_percentage: number;
  mode: string;
  annotated_image?: string | null;
};

function normalizeDetectionResult(occupancy: Partial<DetectionResult> | null | undefined): DetectionResult | null {
  if (!occupancy) return null;

  const people = Math.max(0, Number(occupancy.people_count ?? occupancy.occupied_seats ?? 0));
  const occupied = Math.max(0, Number(occupancy.occupied_seats ?? 0));
  const empty = Math.max(0, Number(occupancy.empty_seats ?? 0));
  const totalSeats = Math.max(1, occupied + empty || people || 1);

  let correctedOccupied = occupied;
  let correctedEmpty = empty;

  if (people > 0) {
    if (correctedOccupied > people || correctedOccupied >= totalSeats || correctedEmpty < 0) {
      correctedOccupied = Math.min(people, totalSeats);
      correctedEmpty = Math.max(totalSeats - correctedOccupied, 0);
    } else if (correctedOccupied < people) {
      correctedOccupied = Math.min(people, totalSeats);
      correctedEmpty = Math.max(totalSeats - correctedOccupied, 0);
    }
  } else if (totalSeats > 0) {
    correctedOccupied = 0;
    correctedEmpty = totalSeats;
  }

  return {
    people_count: people,
    occupied_seats: correctedOccupied,
    empty_seats: correctedEmpty,
    occupancy_percentage: totalSeats > 0 ? Number(((correctedOccupied / totalSeats) * 100).toFixed(1)) : 0,
    mode: occupancy.mode ?? "unknown",
    annotated_image: occupancy.annotated_image ?? null,
  };
}

export default function LiveDetection() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [sourceImage, setSourceImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => () => stopCamera(), []);

  async function analyze(dataUrl: string) {
    setBusy(true);
    setError(null);
    setSourceImage(dataUrl);
    try {
      const response = await fetch(`${API}/api/process/frame`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ frame_data: dataUrl }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "AI frame processing failed.");
      const normalized = normalizeDetectionResult(payload.occupancy);
      setResult(normalized);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "AI frame processing failed.");
    } finally {
      setBusy(false);
    }
  }

  async function startCamera() {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
      window.setTimeout(() => captureFrame(), 700);
    } catch {
      setError("Camera access was unavailable. Use Choose image to analyze a saved camera frame.");
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOn(false);
  }

  function captureFrame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.82);
    void analyze(dataUrl);
  }

  function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => void analyze(String(reader.result));
    reader.onerror = () => setError("Could not read that image.");
    reader.readAsDataURL(file);
  }

  return (
    <section style={{ background: "#fff", borderRadius: 16, padding: 24, border: "1px solid #eadfce" }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
        <button onClick={cameraOn ? stopCamera : startCamera} style={buttonStyle}>{cameraOn ? "Stop live camera" : "Start live camera"}</button>
        <label htmlFor="live-detection-upload" style={buttonStyle}>Choose image</label>
        <input id="live-detection-upload" type="file" accept="image/*" onChange={handleUpload} style={{ display: "none" }} />
        {cameraOn && <button onClick={captureFrame} disabled={busy} style={{ ...buttonStyle, background: "#1d1b19" }}>{busy ? "Detecting..." : "Detect now"}</button>}
      </div>
      <p style={{ color: "#645d58", marginBottom: 0 }}>The camera frame is sent to the local detector and returned with chair labels and bounding boxes.</p>
      {error && <p style={{ color: "#a63125" }}>{error}</p>}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 18, marginTop: 20 }}>
        <div>
          <h3>Live source</h3>
          {cameraOn ? <video ref={videoRef} autoPlay playsInline muted style={mediaStyle} /> : sourceImage ? <img src={sourceImage} alt="Camera source frame" style={mediaStyle} /> : <div style={emptyStyle}>Start the camera or choose an image.</div>}
        </div>
        <div>
          <h3>Detection result</h3>
          {result?.annotated_image ? <img src={`data:image/jpeg;base64,${result.annotated_image}`} alt="Camera frame with chair detection boxes" style={mediaStyle} /> : <div style={emptyStyle}>Detection boxes will appear here.</div>}
        </div>
      </div>

      {result && <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 10, marginTop: 18 }}><Metric label="Occupied" value={result.occupied_seats} /><Metric label="Empty" value={result.empty_seats} /><Metric label="Occupancy" value={`${result.occupancy_percentage}%`} /><Metric label="Mode" value={result.mode} /></div>}
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </section>
  );
}

const buttonStyle = { border: "none", borderRadius: 9, padding: "11px 14px", background: "#d9703f", color: "#fff", fontWeight: 700, cursor: "pointer" };
const mediaStyle = { width: "100%", aspectRatio: "16 / 9", objectFit: "cover" as const, display: "block", borderRadius: 10, background: "#eee7df" };
const emptyStyle = { ...mediaStyle, display: "grid", placeItems: "center", color: "#645d58", padding: 20, boxSizing: "border-box" as const };

function Metric({ label, value }: { label: string; value: string | number }) {
  return <div style={{ background: "#faf3ed", borderRadius: 10, padding: 12 }}><div style={{ color: "#645d58", fontSize: 13 }}>{label}</div><strong style={{ display: "block", marginTop: 4 }}>{value}</strong></div>;
}
