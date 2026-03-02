# Tabular ML Lab — Setup Script (Windows PowerShell)

Write-Host "🔬 Setting up Tabular ML Lab..." -ForegroundColor Cyan

# Check Python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-Not $py) {
    Write-Host "❌ Python not found! Install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment
if (-Not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate
& "venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host "Run: .\run.ps1" -ForegroundColor Cyan
