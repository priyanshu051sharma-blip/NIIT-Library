# SmartLib

AI-powered library occupancy and space management. SmartLib counts people and seat states only; it does not use facial recognition or store identity from video.

## Services

- `frontend`: Next.js student and staff dashboards on port 3000.
- `backend`: FastAPI REST/WebSocket API on port 8000.
- `ai-service`: FastAPI + OpenCV + Ultralytics YOLO inference service on port 8001.
- PostgreSQL and Redis are included in `docker-compose.yml`.

## Run locally

1. Copy `.env.example` to `.env` and change secrets.
2. Run `docker compose up --build`.
3. Open http://localhost:3000.
4. The default UI uses mock occupancy data. The API offers `demo-student` and `demo-admin` tokens for local testing.

For a native Python run: `pip install -r backend/requirements.txt && uvicorn backend.app.main:app --reload --port 8000`. Run the vision service with `AI_MODE=real PERSON_MODEL=ai-service/runs/smartlib-person-smoke/weights/best.pt SEAT_MODEL=ai-service/runs/chair-smoke/weights/best.pt uvicorn app.main:app --reload --port 8001` from the `ai-service` directory. The Compose setup mounts these trained models and runs the AI service in real mode by default.

## Real video mode

Set `AI_MODE=real`, install the AI requirements, configure `VIDEO_SOURCE` as a webcam index, local video path, or RTSP URL, and set `PERSON_MODEL` to a YOLO model. YOLO tracking uses persistent track IDs for people. Seat occupancy uses configurable polygons from `config/cameras.example.json`; a future trained model can be supplied with `SEAT_MODEL` for occupied/empty seat classes.

The provided archive is a person dataset with CSV columns `filename,width,height,class,xmin,ymin,xmax,ymax`; its labels are `person`. The included `Chair Occupancy Detection 3.v1i.yolov9` dataset is a separate three-class chair model dataset: `empty chair`, `guest on chair`, and `occupied chair`. The live engine counts `empty`/`available`/`vacant` classes as empty and `occupied`/`guest`/`person` classes as occupied, using the model's class names. If a frame has no chair detections, it reports zero model-counted seats rather than falsely marking the configured capacity empty. Because that chair set is small and visually broad, retrain and validate it on the library's actual camera viewpoints before deployment.

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
