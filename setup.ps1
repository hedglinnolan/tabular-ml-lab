# Tabular ML Lab — Setup Script (Windows PowerShell)

Write-Host "🔬 Setting up Tabular ML Lab..." -ForegroundColor Cyan

# ── Check for cloud-synced folders (OneDrive, Dropbox, iCloud) ────
$cwd = (Get-Location).Path
$cloudPatterns = @("OneDrive", "Dropbox", "iCloudDrive", "Google Drive")
$inCloudFolder = $false
foreach ($pattern in $cloudPatterns) {
    if ($cwd -match [regex]::Escape($pattern)) {
        $inCloudFolder = $true
        break
    }
}
if ($inCloudFolder) {
    Write-Host ""
    Write-Host "⚠️  This folder appears to be inside a cloud-synced directory ($pattern)." -ForegroundColor Yellow
    Write-Host "   This can cause slow git operations, file locking issues, and unnecessary" -ForegroundColor Yellow
    Write-Host "   syncing of thousands of virtual environment files." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   Recommended: clone the repo to a local folder instead, e.g.:" -ForegroundColor Yellow
    Write-Host "     cd C:\dev" -ForegroundColor White
    Write-Host "     git clone https://github.com/hedglinnolan/tabular-ml-lab.git" -ForegroundColor White
    Write-Host ""
    Write-Host "   Continuing anyway..." -ForegroundColor Gray
    Write-Host ""
}

# ── Try uv first (handles Python version automatically) ──────────
$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv) {
    Write-Host "📦 Found uv — setting up with Python 3.12..." -ForegroundColor Yellow
    uv venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create virtual environment with uv." -ForegroundColor Red
        exit 1
    }
    & ".venv\Scripts\Activate.ps1"

    # --link-mode=copy avoids hardlink errors on cloud-synced and cross-filesystem paths
    Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
    uv pip install --link-mode=copy -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install core dependencies." -ForegroundColor Red
        exit 1
    }

    # Install optional packages that need Python <=3.12
    Write-Host "📥 Installing optional packages (TDA, UMAP)..." -ForegroundColor Yellow
    uv pip install --link-mode=copy giotto-tda umap-learn 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  Optional packages (giotto-tda, umap-learn) failed — TDA/UMAP features will be unavailable." -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "✅ Setup complete!" -ForegroundColor Green
    Write-Host "Run: .\run.ps1" -ForegroundColor Cyan

# ── Fallback to pip ───────────────────────────────────────────────
} else {
    Write-Host "📦 uv not found — using pip" -ForegroundColor Yellow
    Write-Host "For best experience, install uv: irm https://astral.sh/uv/install.ps1 | iex" -ForegroundColor Gray

    # Check Python
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-Not $py) {
        Write-Host "❌ Python not found! Install Python 3.10+ from https://python.org" -ForegroundColor Red
        exit 1
    }
    $pyVersion = python --version 2>&1
    Write-Host "Found: $pyVersion" -ForegroundColor Gray

    # Create virtual environment
    if (-Not (Test-Path "venv")) {
        Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Failed to create virtual environment." -ForegroundColor Red
            exit 1
        }
    }
    & "venv\Scripts\Activate.ps1"

    Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
    pip install --upgrade pip 2>$null
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "⚠️  Some dependencies failed to install." -ForegroundColor Yellow
        Write-Host "Core features still work. Optional packages (giotto-tda, umap-learn) require Python <=3.12." -ForegroundColor Yellow
        Write-Host "Install uv (irm https://astral.sh/uv/install.ps1 | iex) to automatically use the right Python version." -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "✅ Setup complete!" -ForegroundColor Green
    Write-Host "Run: .\run.ps1" -ForegroundColor Cyan
}
