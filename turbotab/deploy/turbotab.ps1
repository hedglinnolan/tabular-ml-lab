# TurboTab: the one command on Windows.
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File turbotab\deploy\turbotab.ps1 [--port N] [--no-open] [--smoke]
#
# Finds a Python 3.12 or newer and hands over to launch.py, which makes TurboTab's environment
# once, starts it and opens the browser (see launch.py). Without a Python 3.12+ it fetches one
# with uv into %TURBOTAB_HOME%\tools (default %USERPROFILE%\.turbotab\tools), once. Set
# TURBOTAB_PYTHON to choose the interpreter.
# "Continue": Windows PowerShell turns a program's stderr into errors when its output is
# redirected (a log, CI), and the server logs to stderr. Each step checks its own result instead.
$ErrorActionPreference = "Continue"

$Here = $PSScriptRoot
$TTHome = if ($env:TURBOTAB_HOME) { $env:TURBOTAB_HOME } else { Join-Path $env:USERPROFILE ".turbotab" }
$Check = "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"

function Test-Python([string]$Exe, [string[]]$Pre) {
    try {
        & $Exe @Pre -c $Check 2>&1 | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

# Each candidate: the program, and the arguments that pick a version (the py launcher's -3.13).
# "python" may be the Microsoft Store's stand-in, which fails the check and is passed over.
function Candidate([string]$Exe, [string[]]$Pre = @()) {
    [pscustomobject]@{ Exe = $Exe; Pre = $Pre }
}
$Candidates = @()
if ($env:TURBOTAB_PYTHON) { $Candidates += Candidate $env:TURBOTAB_PYTHON }
$Candidates += Candidate "py" @("-3.13")
$Candidates += Candidate "py" @("-3.12")
$Candidates += Candidate "python"
$Candidates += Candidate "python3"
$Candidates += Candidate "py" @("-3")

$Exe = $null
$Pre = @()
foreach ($c in $Candidates) {
    if (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
    if (Test-Python $c.Exe $c.Pre) { $Exe = $c.Exe; $Pre = @($c.Pre); break }
}

if (-not $Exe) {
    Write-Host "TurboTab: no Python 3.12 or newer found; fetching one (once, about 60 MB)..."
    $Uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
    if (-not $Uv) {
        $Tools = Join-Path $TTHome "tools"
        $Uv = Join-Path $Tools "uv.exe"
        if (-not (Test-Path $Uv)) {
            New-Item -ItemType Directory -Force -Path $Tools | Out-Null
            $arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "aarch64" } else { "x86_64" }
            $zip = Join-Path $Tools "uv.zip"
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -UseBasicParsing -OutFile $zip -ErrorAction Stop `
                -Uri "https://github.com/astral-sh/uv/releases/latest/download/uv-$arch-pc-windows-msvc.zip"
            Expand-Archive -Path $zip -DestinationPath $Tools -Force -ErrorAction Stop
            Remove-Item $zip -Force
            $found = Get-ChildItem -Path $Tools -Recurse -Filter "uv.exe" | Select-Object -First 1
            if ($found -and $found.FullName -ne $Uv) { Move-Item -Force $found.FullName $Uv }
        }
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $TTHome "tools\python"
    & $Uv python install 3.12
    $Exe = "$(& $Uv python find 3.12)".Trim()
    $Pre = @()
    if (-not $Exe -or -not (Test-Python $Exe $Pre)) {
        Write-Host "TurboTab: could not get Python 3.12. Install it from https://www.python.org/downloads/"
        Write-Host "and start TurboTab again."
        exit 1
    }
}

& $Exe @Pre (Join-Path $Here "launch.py") @args
exit $LASTEXITCODE
