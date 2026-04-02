# Build script for Z-Image API (PowerShell)

Write-Host "Building base image..." -ForegroundColor Green
docker build -f Dockerfile.base -t zimage-base .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Base image build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Building application image..." -ForegroundColor Green
docker build -t zimage-api .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Application image build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Build complete!" -ForegroundColor Green
Write-Host "To run: docker run --gpus all -p 8000:8000 zimage-api" -ForegroundColor Cyan

