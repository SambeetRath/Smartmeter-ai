
# ============================================================
# SmartMeter AI - Complete Automated Project Verification
# Run everything using: .\check_project.ps1
# ============================================================

$ErrorActionPreference = "Stop"

# Project configuration
$ProjectPath = "C:\Users\sambe\SmartMeter AI"
$Python = Join-Path $ProjectPath ".venv\Scripts\python.exe"

# Track execution
$StartTime = Get-Date
$FailedChecks = @()
$TotalChecks = 5
$CurrentCheck = 0

# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------

function Invoke-ProjectCheck {
    param (
        [string]$Name,
        [scriptblock]$Command
    )

    $script:CurrentCheck++

    Write-Host "`n[$script:CurrentCheck/$script:TotalChecks] $Name" -ForegroundColor Yellow
    Write-Host ("-" * 60) -ForegroundColor DarkGray

    try {
        & $Command

        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE"
        }

        Write-Host "`n[PASS] $Name" -ForegroundColor Green
    }
    catch {
        Write-Host "`n[FAIL] $Name" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        $script:FailedChecks += $Name
    }
}

# ------------------------------------------------------------
# Start
# ------------------------------------------------------------

Set-Location $ProjectPath

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "             SMARTMETER AI - PROJECT CHECKS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "Project: $ProjectPath"
Write-Host "Python : $Python"
Write-Host "Started: $StartTime"

# ------------------------------------------------------------
# Environment validation
# ------------------------------------------------------------

if (-not (Test-Path $Python)) {
    Write-Host "`n[ERROR] Virtual environment Python not found!" -ForegroundColor Red
    exit 1
}

# ------------------------------------------------------------
# 1. Automated tests
# ------------------------------------------------------------

Invoke-ProjectCheck "Running complete Pytest suite" {
    & $Python -m pytest -v
}

# ------------------------------------------------------------
# 2. Ruff linting
# ------------------------------------------------------------

Invoke-ProjectCheck "Running Ruff lint checks" {
    & $Python -m ruff check src tests
}

# ------------------------------------------------------------
# 3. Ruff formatting verification
# ------------------------------------------------------------

Invoke-ProjectCheck "Checking Python code formatting" {
    & $Python -m ruff format --check src tests
}

# ------------------------------------------------------------
# 4. Dependency consistency
# ------------------------------------------------------------

Invoke-ProjectCheck "Checking installed dependencies" {
    & $Python -m pip check
}

# ------------------------------------------------------------
# 5. Git verification
# ------------------------------------------------------------

Invoke-ProjectCheck "Checking Git changes and whitespace" {
    git diff --check

    if ($LASTEXITCODE -ne 0) {
        throw "Git diff whitespace check failed"
    }

    Write-Host "`nCurrent Git status:" -ForegroundColor Cyan
    git status --short

    if ($LASTEXITCODE -ne 0) {
        throw "Git status command failed"
    }
}

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "                  VERIFICATION SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "Total checks : $TotalChecks"
Write-Host "Passed       : $($TotalChecks - $FailedChecks.Count)" -ForegroundColor Green
Write-Host "Failed       : $($FailedChecks.Count)" -ForegroundColor $(if ($FailedChecks.Count -gt 0) { "Red" } else { "Green" })
Write-Host "Duration     : $($Duration.ToString('hh\:mm\:ss'))"

if ($FailedChecks.Count -gt 0) {
    Write-Host "`nFailed checks:" -ForegroundColor Red

    foreach ($Check in $FailedChecks) {
        Write-Host " - $Check" -ForegroundColor Red
    }

    Write-Host "`nPROJECT VERIFICATION FAILED" -ForegroundColor Red
    exit 1
}

Write-Host "`nALL PROJECT CHECKS PASSED!" -ForegroundColor Green
Write-Host "Your project is ready for the next development step." -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Cyan

exit 0
