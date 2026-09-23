$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\ai-service"
$env:AI_MODE = "real"
$env:PERSON_MODEL = "runs/smartlib-person-smoke/weights/best.pt"
$env:SEAT_MODEL = "runs/chair-smoke/weights/best.pt"
$env:ROBOFLOW_API_KEY = "MauBf5CLvmmWiaPUqRzL"
$env:ROBOFLOW_WORKSPACE = "khushi-kathuria"
$env:ROBOFLOW_WORKFLOW_ID = "general-segmentation-api-3"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
