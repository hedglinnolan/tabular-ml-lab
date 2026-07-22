# Tabular ML Lab — Windows launcher engine (invoked by "Start Tabular ML Lab.bat").
#
# 1. Downloads uv (the environment manager) into .\.tools (first run, ~35 MB).
# 2. Creates a private Python 3.12 environment in .\.venv and installs the
#    app's libraries (first run, ~1-2 GB, several minutes).
# 3. Creates "Tabular ML Lab" shortcuts (Desktop + Start Menu) with the app
#    icon, so future launches are one double-click.
# 4. Starts the app; the default browser opens automatically.
#
# Everything lives inside the app folder. To uninstall: delete the folder
# and the two shortcuts.
$ErrorActionPreference = "Stop"

$Root  = Split-Path -Parent $PSScriptRoot
$Uv    = Join-Path $Root ".tools\uv.exe"
$Venv  = Join-Path $Root ".venv"
$Py    = Join-Path $Venv "Scripts\python.exe"
$Stamp = Join-Path $Root ".venv-stamp"
$Reqs  = Join-Path $Root "requirements.txt"

function Say($msg)  { Write-Host "`n$msg" -ForegroundColor Cyan }
function Note($msg) { Write-Host $msg -ForegroundColor Gray }

try {
    # -- 1. uv ------------------------------------------------------------
    if (-not (Test-Path $Uv)) {
        Say "First-time setup 1/2: fetching the environment manager (uv)..."
        Note "This is a one-time download (~35 MB)."
        $tools = Join-Path $Root ".tools"
        New-Item -ItemType Directory -Force -Path $tools | Out-Null
        $arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "aarch64" } else { "x86_64" }
        $zipUrl  = "https://github.com/astral-sh/uv/releases/latest/download/uv-$arch-pc-windows-msvc.zip"
        $zipPath = Join-Path $tools "uv.zip"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
        Expand-Archive -Path $zipPath -DestinationPath $tools -Force
        Remove-Item $zipPath -Force
        $found = Get-ChildItem -Path $tools -Recurse -Filter "uv.exe" | Select-Object -First 1
        if ($null -eq $found) { throw "uv.exe not found after download." }
        if ($found.FullName -ne $Uv) { Move-Item -Force $found.FullName $Uv }
    }

    # -- 2. Python environment -------------------------------------------
    $reqHash = (Get-FileHash -Algorithm SHA256 $Reqs).Hash
    $stampOk = (Test-Path $Stamp) -and ((Get-Content $Stamp -Raw).Trim() -eq $reqHash)
    if (-not (Test-Path $Py) -or -not $stampOk) {
        Say "First-time setup 2/2: installing Python and the analysis libraries..."
        Note "This downloads ~1-2 GB and takes a few minutes - ONE TIME ONLY."
        Note "Every launch after this one takes seconds and works offline."
        & $Uv venv --python 3.12 $Venv
        if ($LASTEXITCODE -ne 0) { throw "Python environment creation failed." }
        & $Uv pip install --python $Py -r $Reqs
        if ($LASTEXITCODE -ne 0) { throw "Library installation failed - check your internet connection and re-run." }
        Set-Content -Path $Stamp -Value $reqHash -NoNewline
        Say "Setup complete."
    }

    # -- 3. Shortcuts with the app icon ----------------------------------
    $batPath  = Join-Path $Root "Start Tabular ML Lab.bat"
    $iconPath = Join-Path $Root "launcher\icon.ico"
    $shell = New-Object -ComObject WScript.Shell
    $targets = @(
        (Join-Path ([Environment]::GetFolderPath("Desktop")) "Tabular ML Lab.lnk"),
        (Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs\Tabular ML Lab.lnk")
    )
    foreach ($lnkPath in $targets) {
        if (-not (Test-Path $lnkPath)) {
            $lnk = $shell.CreateShortcut($lnkPath)
            $lnk.TargetPath = $batPath
            $lnk.WorkingDirectory = $Root
            $lnk.IconLocation = $iconPath
            $lnk.WindowStyle = 7   # minimized console
            $lnk.Description = "Tabular ML Lab - local ML research workbench"
            $lnk.Save()
        }
    }
    Note "A 'Tabular ML Lab' shortcut is on your Desktop and Start Menu."

    # -- 4. Launch --------------------------------------------------------
    Say "Starting Tabular ML Lab - your browser will open automatically."
    Note "Keep this window open while you work; closing it quits the app."
    Set-Location $Root
    & $Py -m streamlit run (Join-Path $Root "app.py") --browser.gatherUsageStats false
    exit $LASTEXITCODE
}
catch {
    Write-Host "`nSetup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
