# Tabular ML Lab — Setup Script (Windows PowerShell)

Write-Host "🔬 Setting up Tabular ML Lab..." -ForegroundColor Cyan

# Check Python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-Not $py) {
    Write-Host "❌ Python not found! Install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Check Python version
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

# Activate
& "venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip 2>$null
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "⚠️  Some dependencies failed to install." -ForegroundColor Yellow
    Write-Host "The app may still work — optional packages (giotto-tda, umap-learn) require Python <=3.12." -ForegroundColor Yellow
    Write-Host "Core features work on Python 3.10-3.13." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host "Run: .\run.ps1" -ForegroundColor Cyan
