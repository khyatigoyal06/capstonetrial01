# One-click Windows launcher: run scripts\start_windows.bat from the repository.
$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
$log = Join-Path $repo "server.log"
Set-Location $repo
"=== $(Get-Date) starting ===" | Out-File $log -Encoding utf8

function Log($msg) { Write-Host $msg -ForegroundColor Cyan; $msg | Out-File $log -Append -Encoding utf8 }
function Run($cmdline) {
  Log "> $cmdline"
  cmd /c "$cmdline 2>&1" | ForEach-Object { Write-Host $_; "$_" | Out-File $log -Append -Encoding utf8 }
  return $LASTEXITCODE
}

# MediaPipe/PyTorch need Python 3.10-3.12.
$candidates = @("py -3.12", "py -3.11", "py -3.10", "python", "`"$env:USERPROFILE\anaconda3\python.exe`"")
$py = $null
foreach ($c in $candidates) {
  $v = cmd /c "$c -c `"import sys;print(sys.version[:4])`" 2>nul"
  if ($LASTEXITCODE -eq 0 -and "$v".Trim() -match '^3\.(10|11|12)$') { $py = $c; Log "Using $c (Python $v)"; break }
  elseif ($v) { Log "Skipping $c (Python $v)" }
}
if (-not $py) { Log "ERROR: need Python 3.10, 3.11 or 3.12. Install 3.11 from python.org and re-run."; Read-Host "Press Enter to close"; exit 1 }

$venvPy = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Run "$py -m venv .venv" | Out-Null }
# Reinstall whenever requirements.txt changes (its hash is stored in .venv\.deps_ok).
$reqHash = (Get-FileHash (Join-Path $repo "requirements.txt") -Algorithm SHA256).Hash
$marker = Join-Path $repo ".venv\.deps_ok"
$installed = if (Test-Path $marker) { (Get-Content $marker -Raw).Trim() } else { "" }
if ($installed -ne $reqHash) {
  Log "Installing dependencies (first run: several minutes, PyTorch is large)..."
  Run "`"$venvPy`" -m pip install --upgrade pip" | Out-Null
  $code = Run "`"$venvPy`" -m pip install -r requirements.txt"
  if ($code -ne 0) { Log "ERROR: pip install failed (exit $code). See log."; Read-Host "Press Enter to close"; exit 1 }
  Set-Content -Path $marker -Value $reqHash
}

# Rebuild the frontend when it is missing or any source file is newer than the build.
$distIndex = Join-Path $repo "frontend\dist\index.html"
$sources = @(Get-ChildItem (Join-Path $repo "frontend\src") -Recurse -File) + @(Get-Item (Join-Path $repo "frontend\package.json"), (Join-Path $repo "frontend\index.html"))
$newestSource = ($sources | Measure-Object -Property LastWriteTime -Maximum).Maximum
if (-not (Test-Path $distIndex) -or (Get-Item $distIndex).LastWriteTime -lt $newestSource) {
  if (Get-Command npm -ErrorAction SilentlyContinue) {
    Log "Building the React frontend..."
    Push-Location (Join-Path $repo "frontend")
    Run "npm ci" | Out-Null
    Run "npm run build" | Out-Null
    Pop-Location
  } else { Log "npm not found: React app skipped; the study page still works at /api/gaze/capture" }
}

$url = if (Test-Path (Join-Path $repo "frontend\dist\index.html")) { "http://localhost:8001" } else { "http://localhost:8001/api/gaze/capture" }
# A server left running from an earlier launch would keep serving old code on the same port.
$old = Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($procId in $old) { Log "Stopping the previous server (PID $procId)"; Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue }
if ($old) { Start-Sleep 2 }
Log "Starting server on http://localhost:8001 ... (close this window to stop it)"
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile -Command `"Start-Sleep 12; Start-Process '$url'`""
Run "`"$venvPy`" -m uvicorn api.main:app --host 127.0.0.1 --port 8001" | Out-Null
Log "Server stopped."
Read-Host "Press Enter to close"
