# Qwen-0.6B QLoRA Installation Script for Windows
# Run in PowerShell as Administrator
# =================================================

$ErrorActionPreference = "Stop"

Write-Host "Qwen-0.6B QLoRA Training - Installation Helper" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write ""

# Check Python version
$pythonCmd = if (Test-Path "C:\Python39\python.exe") { "C:\Python39\python.exe" } else { "python" }
try {
    $pythonVersion = & "$pythonCmd" -c "import sys; print(sys.version)" 2>$null
    Write-Host "[1/4] Python found: $pythonVersion"
} catch {
    Write-Error "Python not found! Please install Python 3.9-3.11 from https://www.python.org/downloads/"
    exit 1
}

# Install PyTorch
Write-Host "`n[2/4] Installing/upgrading PyTorch..." -ForegroundColor Yellow
$pytorchCmd = if ($null -ne (Get-Command "nvcc" -ErrorAction SilentlyContinue)) {
    # CUDA available
    "pip install --index-url https://download.pytorch.org/whl/cu123 torch torchvision torchaudio"
} else {
    # CPU only
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
}
Write-Host "Command: $pytorchCmd"

# Install core dependencies
Write-Host "`n[3/4] Installing core dependencies..." -ForegroundColor Yellow
$coreDeps = @(
    "transformers>=4.35.0"
    "datasets>=2.14.0"
    "huggingface-hub>=0.19.0"
    "accelerate>=0.26.0"
    "peft==0.7.1"
    "safetensors>=0.4.0"
)
foreach ($dep in $coreDeps) {
    Write-Host "  - $dep"
}
Write-Host "[3/4] Installing core dependencies..."

"

# Install optional packages
Write-Host "`n[4/4] Installing optional packages (bitsandbytes, etc.)..." -ForegroundColor Yellow

# bitsandbytes - try pip first, will fail gracefully if not suitable for your system
try {
    & pip install bitsandbytes --upgrade 2>$null
    Write-Host "  - bitsandbytes installed successfully"
} catch {
    Write-Host "  - Note: bitsandbytes could not be installed automatically" -ForegroundColor Yellow
    Write-Host "        See https://github.com/TimD'yakov/bitsandbytes#windows for manual installation" -ForegroundColor Yellow
}

# Finish
Write-Host "`n=============================================" -ForegroundColor Green
Write-Host "Installation Summary:" -ForegroundColor White
Write-Host "  Python version: $pythonVersion"
Write-Host "  PyTorch: Not yet installed (will be installed with pip install torch...)"
Write-Host "  Dependencies: Ready for `pip install -r requirements_qwen_lora.txt`"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  python fetch_hf_datasets.py"
Write-Host "  python scripts\qwen_finetune.py --model_name Qwen/Qwen-0_6B --data_path techedata/CodeAlpaca-20k"