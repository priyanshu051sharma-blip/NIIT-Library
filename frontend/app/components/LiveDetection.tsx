"use client";

import { ChangeEvent, MouseEvent, useCallback, useEffect, useRef, useState } from "react";
import { CameraFeed, useOccupancy } from "../context/OccupancyContext";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const AI_API = process.env.NEXT_PUBLIC_AI_SERVICE_URL || "http://localhost:8001";

type DetectionResult = {
  people_count: number;
  occupied_seats: number;
  empty_seats: number;
  occupancy_percentage: number;
  mode: string;
  annotated_image?: string | null;
};

type SeatPoint = [number, number];

function normalizeDetectionResult(occupancy: Partial<DetectionResult> | null | undefined): DetectionResult | null {
  if (!occupancy) return null;

  const people = Math.max(0, Number(occupancy.people_count ?? occupancy.occupied_seats ?? 0));
  const occupied = Math.max(0, Number(occupancy.occupied_seats ?? 0));
  const empty = Math.max(0, Number(occupancy.empty_seats ?? 0));
  const totalSeats = occupied + empty;

  return {
    people_count: people,
    occupied_seats: occupied,
    empty_seats: empty,
    occupancy_percentage: totalSeats > 0 ? Number(((occupied / totalSeats) * 100).toFixed(1)) : 0,
    mode: occupancy.mode ?? "unknown",
    annotated_image: occupancy.annotated_image ?? null,
  };
}

export default function LiveDetection() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const busyRef = useRef(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [ipCameraOn, setIpCameraOn] = useState(true);
  const [ipCameraStatus, setIpCameraStatus] = useState("connecting");
  const [calibratingCameraId, setCalibratingCameraId] = useState<string | null>(null);
  const [calibrationPoints, setCalibrationPoints] = useState<SeatPoint[]>([]);
  const [calibrationBusy, setCalibrationBusy] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [sourceImage, setSourceImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { snapshot } = useOccupancy();
  const cameraFeeds = snapshot?.cameras ?? [];
  const librarySummary = snapshot?.camera_groups.library;
  const discussionSummary = snapshot?.camera_groups.discussion;
  const entranceSummary = snapshot?.camera_groups.entrance;
  const cameraFrameKey = snapshot?.timestamp ?? "waiting";
  const libraryTotal = librarySummary?.total_seats ?? 0;
  const discussionTotal = discussionSummary?.total_seats ?? 0;
  const cameraStatusText = ipCameraOn
    ? snapshot
      ? `${cameraFeeds.filter((feed) => feed.status === "connected").length} of ${cameraFeeds.length} cameras connected`
      : ipCameraStatus
    : ipCameraStatus;

  useEffect(() => () => stopCamera(), []);

  const analyze = useCallback(async (dataUrl: string) => {
    busyRef.current = true;
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
      busyRef.current = false;
      setBusy(false);
    }
  }, []);

  async function startCamera() {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      setCameraOn(true);
    } catch {
      setError("Camera access was unavailable. Use Choose image to analyze a saved camera frame.");
    }
  }

  async function startIpCamera() {
    try {
      setError(null);
      const response = await fetch(`${AI_API}/ai/stream/start`, { method: "POST" });
      if (!response.ok) throw new Error("Could not start the configured IP camera stream.");
      stopCamera();
      setIpCameraOn(true);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not start the IP camera stream.");
    }
  }

  async function stopIpCamera() {
    try {
      const response = await fetch(`${AI_API}/ai/stream/stop`, { method: "POST" });
      if (!response.ok) throw new Error("Could not stop the IP camera stream.");
      setIpCameraOn(false);
      setIpCameraStatus("stopped");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not stop the IP camera stream.");
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOn(false);
  }

  const captureFrame = useCallback(() => {
    if (busyRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.82);
    void analyze(dataUrl);
  }, [analyze]);

  useEffect(() => {
    if (!cameraOn) return;
    const video = videoRef.current;
    if (!video || !streamRef.current) return;

    video.srcObject = streamRef.current;
    const timer = window.setInterval(captureFrame, 1500);
    return () => window.clearInterval(timer);
  }, [cameraOn, captureFrame]);

  function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => void analyze(String(reader.result));
    reader.onerror = () => setError("Could not read that image.");
    reader.readAsDataURL(file);
  }

  function beginCalibration(feed: CameraFeed) {
    setCalibrationPoints(feed.seat_roi ?? []);
    setCalibratingCameraId(feed.id);
    setError(null);
  }

  function addCalibrationPoint(event: MouseEvent<HTMLDivElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    if (calibrationPoints.length >= 64) {
      setError("A seat area can contain at most 64 points.");
      return;
    }
    const x = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width));
    const y = Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height));
    setCalibrationPoints((current) => [...current, [x, y]]);
  }

  async function saveCalibration() {
    if (!calibratingCameraId || calibrationPoints.length < 3) return;
    setCalibrationBusy(true);
    setError(null);
    try {
      const response = await fetch(`${AI_API}/ai/cameras/${encodeURIComponent(calibratingCameraId)}/roi`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: calibrationPoints }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not save this camera's seat area.");
      setCalibratingCameraId(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not save this camera's seat area.");
    } finally {
      setCalibrationBusy(false);
    }
  }

  async function clearCalibration() {
    if (!calibratingCameraId) return;
    setCalibrationBusy(true);
    setError(null);
    try {
      const response = await fetch(`${AI_API}/ai/cameras/${encodeURIComponent(calibratingCameraId)}/roi`, {
        method: "DELETE",
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not remove this camera's seat area.");
      setCalibrationPoints([]);
      setCalibratingCameraId(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not remove this camera's seat area.");
    } finally {
      setCalibrationBusy(false);
    }
  }

  const calibratingFeed = cameraFeeds.find((feed) => feed.id === calibratingCameraId);

  return (
    <section style={{ background: "#fff", borderRadius: 16, padding: 24, border: "1px solid #eadfce" }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
        <button onClick={ipCameraOn ? stopIpCamera : startIpCamera} disabled={cameraOn} style={buttonStyle}>{ipCameraOn ? "Disconnect all cameras" : "Connect all cameras"}</button>
        <button onClick={cameraOn ? stopCamera : startCamera} disabled={ipCameraOn} style={buttonStyle}>{cameraOn ? "Stop local camera" : "Use local camera"}</button>
        <label htmlFor="live-detection-upload" style={buttonStyle}>Choose image</label>
        <input id="live-detection-upload" type="file" accept="image/*" onChange={handleUpload} style={{ display: "none" }} />
        {cameraOn && <button onClick={captureFrame} disabled={busy} style={{ ...buttonStyle, background: "#1d1b19" }}>{busy ? "Detecting..." : "Detect now"}</button>}
      </div>
      <p style={{ color: "#645d58", marginBottom: 0 }}>
        The seven RTSP feeds connect server-side; credentials are not sent to the browser.         Status: {cameraStatusText}
      </p>
      <p style={{ color: "#81756d", fontSize: 13 }}>
        People are person detections; occupied seats use seat classifications and people inside the calibrated seating area. People outside that area do not count as seated.
      </p>
      {error && <p style={{ color: "#a63125" }}>{error}</p>}

      {ipCameraOn ? (
        <>
          <h3 style={{ marginBottom: 8 }}>Live camera feeds and AI detection</h3>
          {cameraFeeds.length > 0 && (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 14, marginTop: 14 }}>
              <CameraGroupSummary
                title="Library seating"
                cameras="Library-1 · Library-2 · Library-4"
                summary={librarySummary}
              />
              <CameraGroupSummary
                title="Discussion rooms"
                cameras="Cabin-1 · Cabin-2"
                summary={discussionSummary}
              />
              <CameraGroupSummary
                title="Entrances"
                cameras="Library Main · Library-3"
                summary={entranceSummary}
                peopleOnly
              />
            </div>
          )}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 16, marginTop: 18 }}>
            {cameraFeeds.map((feed) => (
              <article key={feed.id} style={{ border: "1px solid #eadfce", borderRadius: 12, overflow: "hidden", minWidth: 0 }}>
                <img
                  src={`${AI_API}/ai/camera/frame.jpg?camera_id=${encodeURIComponent(feed.id)}&annotated=true&frame=${cameraFrameKey}`}
                  alt={`${feed.name} live camera feed with detection`}
                  style={mediaStyle}
                />
                <div style={{ padding: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" }}>
                    <strong>{feed.name}</strong>
                    <span style={{ color: feed.status === "connected" ? "#287a48" : "#a63125", fontSize: 13 }}>
                      {feed.status}
                    </span>
                  </div>
                  <span style={{ display: "inline-block", marginTop: 6, color: "#8a5b22", fontSize: 12 }}>
                    {feed.purpose === "library" ? "Library seating" : feed.purpose === "discussion" ? "Discussion room" : "Entrance"}
                  </span>
                  {feed.error && <p style={{ color: "#a63125", fontSize: 12, marginBottom: 0 }}>{feed.error}</p>}
                  <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginTop: 10, color: "#645d58", fontSize: 13 }}>
                    <span>People detected: {feed.occupancy.mode === "waiting" ? "Waiting" : feed.occupancy.people_count}</span>
                    {feed.purpose !== "entrance" && (
                      <>
                        <span>Occupied seats: {feed.occupancy.mode === "waiting" ? "Waiting" : feed.occupancy.occupied_seats}</span>
                        <span>Available seats: {feed.occupancy.mode === "waiting" ? "Waiting" : feed.occupancy.empty_seats}</span>
                      </>
                    )}
                    <span>Detection: {feed.purpose === "entrance" ? "people only" : feed.occupancy.mode}</span>
                  </div>
                  {feed.purpose !== "entrance" && (
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, marginTop: 12 }}>
                      <span style={{ color: feed.calibrated ? "#287a48" : "#8a5b22", fontSize: 12 }}>
                        {feed.calibrated ? "Seat area calibrated" : "Seat area not calibrated"}
                      </span>
                      <button onClick={() => beginCalibration(feed)} style={smallButtonStyle}>Calibrate seat area</button>
                    </div>
                  )}
                </div>
              </article>
            ))}
            {cameraFeeds.length === 0 && <div style={emptyStyle}>Waiting for the seven configured camera channels.</div>}
          </div>
        </>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 18, marginTop: 20 }}>
          <div>
            <h3>Live source</h3>
            {cameraOn ? <video ref={videoRef} autoPlay playsInline muted style={mediaStyle} /> : sourceImage ? <img src={sourceImage} alt="Camera source frame" style={mediaStyle} /> : <div style={emptyStyle}>Connect the seven IP cameras, use a local camera, or choose an image.</div>}
          </div>
          <div>
            <h3>Detection result</h3>
            {result?.annotated_image ? <img src={`data:image/jpeg;base64,${result.annotated_image}`} alt="Camera frame with chair detection boxes" style={mediaStyle} /> : <div style={emptyStyle}>Detection boxes will appear here.</div>}
          </div>
        </div>
      )}

      {calibratingFeed && (
        <div style={calibrationBackdropStyle}>
          <div role="dialog" aria-modal="true" aria-labelledby="camera-calibration-title" style={calibrationDialogStyle}>
            <h2 id="camera-calibration-title" style={{ marginTop: 0 }}>Calibrate {calibratingFeed.name}</h2>
            <p style={{ color: "#645d58" }}>
              Click at least three points around the seating area. Keep bookshelves and walkways outside the polygon.
              Seat detections outside this region will be ignored.
            </p>
            <div onClick={addCalibrationPoint} style={calibrationImageStyle}>
              <img
                src={`${AI_API}/ai/camera/frame.jpg?camera_id=${encodeURIComponent(calibratingFeed.id)}&annotated=false&frame=${cameraFrameKey}`}
                alt={`${calibratingFeed.name} calibration source`}
                style={mediaStyle}
              />
              <svg viewBox="0 0 1000 562.5" preserveAspectRatio="none" style={calibrationOverlayStyle}>
                {calibrationPoints.length >= 3 && (
                  <polygon
                    points={calibrationPoints.map(([x, y]) => `${x * 1000},${y * 562.5}`).join(" ")}
                    fill="rgba(45, 150, 90, 0.2)"
                    stroke="#35b66a"
                    strokeWidth="4"
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                {calibrationPoints.length === 2 && (
                  <polyline
                    points={calibrationPoints.map(([x, y]) => `${x * 1000},${y * 562.5}`).join(" ")}
                    fill="none"
                    stroke="#35b66a"
                    strokeWidth="4"
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                {calibrationPoints.map(([x, y], index) => (
                  <circle key={index} cx={x * 1000} cy={y * 562.5} r="9" fill="#35b66a" stroke="#fff" strokeWidth="3" />
                ))}
              </svg>
            </div>
            <p style={{ color: "#645d58", fontSize: 13 }}>Points: {calibrationPoints.length} / 64</p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              <button onClick={() => setCalibrationPoints([])} disabled={calibrationBusy || calibrationPoints.length === 0} style={smallButtonStyle}>Start over</button>
              <button onClick={() => setCalibrationPoints((points) => points.slice(0, -1))} disabled={calibrationBusy || calibrationPoints.length === 0} style={smallButtonStyle}>Undo point</button>
              <button onClick={() => void saveCalibration()} disabled={calibrationBusy || calibrationPoints.length < 3} style={buttonStyle}>{calibrationBusy ? "Saving..." : "Save seat area"}</button>
              {calibratingFeed.calibrated && <button onClick={() => void clearCalibration()} disabled={calibrationBusy} style={smallButtonStyle}>Remove calibration</button>}
              <button onClick={() => setCalibratingCameraId(null)} disabled={calibrationBusy} style={smallButtonStyle}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      {!ipCameraOn && result && <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 10, marginTop: 18 }}><Metric label="People in view" value={result.people_count} /><Metric label="Occupied" value={result.occupied_seats} /><Metric label="Available" value={result.empty_seats} /><Metric label="Occupancy" value={`${result.occupancy_percentage}%`} /><Metric label="Mode" value={result.mode} /></div>}
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </section>
  );
}

const buttonStyle = { border: "none", borderRadius: 9, padding: "11px 14px", background: "#d9703f", color: "#fff", fontWeight: 700, cursor: "pointer" };
const smallButtonStyle = { border: "1px solid #eadfce", borderRadius: 8, padding: "8px 10px", background: "#fff", color: "#514941", fontWeight: 600, cursor: "pointer" };
const mediaStyle = { width: "100%", aspectRatio: "16 / 9", objectFit: "cover" as const, display: "block", borderRadius: 10, background: "#eee7df" };
const emptyStyle = { ...mediaStyle, display: "grid", placeItems: "center", color: "#645d58", padding: 20, boxSizing: "border-box" as const };
const calibrationBackdropStyle = { position: "fixed" as const, inset: 0, zIndex: 1000, background: "rgba(25, 22, 20, 0.7)", display: "grid", placeItems: "center", padding: 20, overflowY: "auto" as const };
const calibrationDialogStyle = { width: "min(100%, 1000px)", maxHeight: "95vh", overflowY: "auto" as const, background: "#fff", borderRadius: 16, padding: 24, boxSizing: "border-box" as const };
const calibrationImageStyle = { position: "relative" as const, cursor: "crosshair", userSelect: "none" as const };
const calibrationOverlayStyle = { position: "absolute" as const, inset: 0, width: "100%", height: "100%", pointerEvents: "none" as const };

function Metric({ label, value }: { label: string; value: string | number }) {
  return <div style={{ background: "#faf3ed", borderRadius: 10, padding: 12 }}><div style={{ color: "#645d58", fontSize: 13 }}>{label}</div><strong style={{ display: "block", marginTop: 4 }}>{value}</strong></div>;
}

function CameraGroupSummary({
  title,
  cameras,
  summary,
  peopleOnly = false,
}: {
  title: string;
  cameras: string;
  summary?: {
    people_count: number;
    occupied_seats: number;
    empty_seats: number;
    total_seats: number;
    occupancy_percentage: number;
    detected_cameras: number;
  };
  peopleOnly?: boolean;
}) {
  const hasDetections = Boolean(summary?.detected_cameras);
  const values: { label: string; value: string | number }[] = [
    { label: "People", value: hasDetections ? summary?.people_count ?? 0 : "Waiting" },
  ];
  if (!peopleOnly) {
    values.push(
      { label: "Occupied seats", value: hasDetections ? summary?.occupied_seats ?? 0 : "Waiting" },
      { label: "Available seats", value: hasDetections ? summary?.empty_seats ?? 0 : "Waiting" },
      { label: "Seat load", value: hasDetections ? `${summary?.occupancy_percentage ?? 0}%` : "Waiting" },
    );
  }

  return (
    <section style={{ background: "#faf5ef", border: "1px solid #eadfce", borderRadius: 14, padding: 16 }}>
      <h4 style={{ margin: 0, color: "#1d1b19" }}>{title}</h4>
      <p style={{ margin: "5px 0 14px", color: "#81756d", fontSize: 12 }}>{cameras}</p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(110px, 1fr))", gap: 10 }}>
        {values.map(({ label, value }) => <Metric key={label} label={label} value={value} />)}
      </div>
      <p style={{ margin: "12px 0 0", color: "#81756d", fontSize: 12 }}>
        {summary ? `${summary.detected_cameras} camera feeds reporting` : "Waiting for camera data"}
      </p>
    </section>
  );
}
