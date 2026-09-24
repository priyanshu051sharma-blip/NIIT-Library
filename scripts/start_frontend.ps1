$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\frontend"
npm run dev -- --hostname 0.0.0.0 --port 3001
