# SmartLib

AI-powered library occupancy and space management. SmartLib counts people and seat states only; it does not use facial recognition or store identity from video.

## Services

- `frontend`: Next.js student and staff dashboards on port 3001.
- `backend`: FastAPI REST/WebSocket API on port 8000.
- `ai-service`: FastAPI + OpenCV + Ultralytics YOLO inference service on port 8001.
- PostgreSQL and Redis are included in `docker-compose.yml`.

## Run locally

1. Copy `.env.example` to `.env` and change secrets.
2. Run `docker compose up --build`.
3. Open http://localhost:3001.
4. Compose runs live occupancy inference with the configured models and cameras. The API offers `demo-student` and `demo-admin` tokens for local testing; if no camera is connected, the portal shows that live camera data is unavailable.

For a native Python run: `pip install -r backend/requirements.txt && uvicorn backend.app.main:app --reload --port 8000`. Run the vision service with `AI_MODE=real PERSON_MODEL=ai-service/runs/smartlib-person-smoke/weights/best.pt SEAT_MODEL=ai-service/runs/chair-smoke/weights/best.pt uvicorn app.main:app --reload --port 8001` from the `ai-service` directory. The Compose setup mounts these trained models and runs the AI service in real mode by default.

To use the Roboflow chair occupancy model shown in the project, set `ROBOFLOW_API_KEY` before starting Compose. The default model ID is `chair-occupancy-detection-2/3`; override `ROBOFLOW_MODEL_ID` when using another deployed version. Roboflow inference is used when the key is present, while the local YOLO model remains available when it is not.

## Real video mode

Set `AI_MODE=real`, install the AI requirements, configure `VIDEO_SOURCE` as a webcam index, local video path, or RTSP URL, and set `PERSON_MODEL` to a YOLO model. Seat occupancy uses configurable polygons from `config/cameras.example.json`; a trained model can be supplied with `SEAT_MODEL` for occupied/empty seat classes.

### Configure an IP camera

The AI service opens the IP camera server-side, so camera credentials are never sent to the browser. Copy `.env.example` to `.env` and set `CAMERA_HOST`, `CAMERA_PORT`, `CAMERA_PROTOCOL`, `CAMERA_STREAM_PATH`, `CAMERA_USERNAME`, and `CAMERA_PASSWORD` to values supported by your camera. `CAMERA_STREAM_URL` can be used instead when you already know its complete stream URL. The default HTTP MJPEG path is `/cgi-bin/mjpg/video.cgi`; replace it if your camera uses another endpoint. Leave `VIDEO_SOURCE=0` as-is to use these camera settings. A configured `CAMERA_STREAM_URL` or `CAMERA_HOST` takes precedence over legacy `VIDEO_SOURCE` and `RTSP_URL` values.

Start the application with `docker compose up --build`, then open the staff live-detection page and choose **Connect IP camera**. The AI service continuously detects from the configured stream and serves the annotated preview at `/ai/stream.mjpeg`; camera diagnostics are available at `/ai/camera/status`. Diagnostics return generic connection errors and never include the camera URL or password. The AI service's host port is bound to loopback only; do not expose it through a public reverse proxy. Do not put credentials in a frontend setting or commit a populated `.env` file. Protect camera traffic with a trusted network or VPN, and use HTTPS where supported.

The provided archive is a person dataset with CSV columns `filename,width,height,class,xmin,ymin,xmax,ymax`; its labels are `person`. The included `Chair Occupancy Detection 3.v1i.yolov9` dataset is a separate three-class chair model dataset: `empty chair`, `guest on chair`, and `occupied chair`. The current seat-model checkpoint scored 0.974 precision, 0.988 recall, 0.995 mAP50, and 0.952 mAP50-95 on its 13-image, 23-instance held-out test split. Treat these as dataset results, not a guarantee of live-camera accuracy: the supplied set is small and visually broad. The live engine uses the seat model's class names for occupied and empty counts, supplements missed seat boxes with detected generic chair boxes, and uses one-to-one person-chair matches to identify occupied seats when the seat-state model misses them. When a calibrated seating polygon is available, people whose lower-body anchor falls inside it can also recover missed occupied-seat detections; people outside the polygon do not increase occupied counts. When the generic person detector is available, occupied counts are bounded by people detected inside the seating area, preventing phantom occupied chairs in empty rooms. Per-camera people, occupied-seat, and seat-inventory values use a five-frame median to reduce single-frame detector flicker. Library-2 (`cam-2`) and Library-4 (`cam-4`) use a 0.15 person confidence threshold, calibrated against sampled feed frames; all other cameras use the shared `PERSON_CONFIDENCE` setting (0.25 by default). These thresholds improve precision/recall for those viewpoints but do not force fixed counts, and should be revalidated as lighting or camera positions change. Unmatched detected chairs count as available. The default `/ai/occupancy` and student/admin library load use Library-1 (`cam-1`), Library-2 (`cam-2`), and Library-4 (`cam-4`) only. Cabins 1 and 2 (`cam-6`, `cam-7`) are the discussion-room group; Library-3 (`cam-3`) and Library Main (`cam-5`) are entrance cameras and do not contribute seat counts. Query `/ai/occupancy?group=discussion` or `?group=entrance` for those groups, `?group=all` for all camera seat counts, or `/ai/occupancy?camera_id=cam-2` for one camera. Camera views should cover non-overlapping seating areas to avoid counting the same seat more than once. Retrain and validate against the library's actual camera viewpoints before deployment.

The student and staff dashboards use the same backend occupancy snapshot, including group and per-camera values. Room bookings are written to and read from the backend API, so student and staff portals share server-side booking data instead of separate browser-local records. People detections and occupied-seat counts remain separate measurements; they are not forced to match.

## Train and test the detector

The supplied person annotations can be converted and fine-tuned with CUDA:

```powershell
python ai-service/scripts/prepare_person_dataset.py
python ai-service/scripts/train_person.py --epochs 20 --device 0 --name smartlib-person
```

Test a video file, webcam (`--source 0`), or server-side RTSP URL with persistent tracking:

```powershell
python ai-service/scripts/test_video.py --source path/to/video.mp4 --model ai-service/runs/smartlib-person/smartlib-person/weights/best.pt
```

For the service, set `AI_MODE=real`, `PERSON_MODEL` to the trained person `best.pt`, `SEAT_MODEL` to the chair `best.pt`, and `VIDEO_SOURCE` to a local video, webcam index, or RTSP URL. Start the stream with `POST /ai/stream/start` and inspect results with `GET /ai/occupancy`; stop it with `POST /ai/stream/stop`. RTSP credentials remain server-side.

## Security and privacy

RTSP URLs remain server-side and are never returned to students. Replace the demo auth with JWT verification, wire Redis pub/sub for multi-instance WebSocket fan-out, enable HTTPS at the reverse proxy, apply rate limiting, and record staff CCTV access in `audit_logs` before production deployment. Configure video retention separately; the default system stores occupancy metadata only.

API docs are available from FastAPI at `/docs` on the backend and AI service.
