$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."
$env:AI_SERVICE_URL = "http://localhost:8001"
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
