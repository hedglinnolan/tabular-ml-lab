# Tabular ML Lab — Run Script (Windows PowerShell)

Write-Host "🔬 Starting Tabular ML Lab..." -ForegroundColor Cyan

# Find virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    & ".venv\Scripts\Activate.ps1"
} else {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Run .\setup.ps1 first to set up the environment." -ForegroundColor Yellow
    exit 1
}

Write-Host "🌐 Opening at http://localhost:8501" -ForegroundColor Green
streamlit run app.py --server.headless true
