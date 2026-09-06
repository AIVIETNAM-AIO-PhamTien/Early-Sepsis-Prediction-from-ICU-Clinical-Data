# Download full sepsyd-data from Kaggle into data/raw/
# Requires: pip install kaggle && kaggle auth login (or KAGGLE_API_TOKEN)
# Dataset: https://www.kaggle.com/datasets/nguyenhoangthaotrinh/sepsyd-data

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OutDir = Join-Path $ProjectRoot "data\raw"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Downloading nguyenhoangthaotrinh/sepsyd-data to $OutDir ..."
python -m kaggle datasets download -d nguyenhoangthaotrinh/sepsyd-data -p $OutDir --unzip

Write-Host "Done. Expected layout:"
Write-Host "  data/raw/training_setA/*.psv"
Write-Host "  data/raw/training_setB/*.psv"
