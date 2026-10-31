$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$EnvFile = Join-Path $ProjectRoot ".env"
if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
    Write-Host "ERROR: .env was not found in the project root: $ProjectRoot" -ForegroundColor Red
    Write-Host "Create .env from .env.example and configure DATABASE_URL before starting the API."
    exit 1
}

try {
    $EnvLines = Get-Content -LiteralPath $EnvFile -ErrorAction Stop
}
catch {
    Write-Host "ERROR: Could not read the project .env file." -ForegroundColor Red
    exit 1
}

$DatabaseUrl = $null
foreach ($Line in $EnvLines) {
    if ($Line -match '^\s*DATABASE_URL\s*=(.*)$') {
        $DatabaseUrl = $Matches[1].Trim()
        break
    }
}

if ($null -eq $DatabaseUrl) {
    Write-Host "ERROR: DATABASE_URL is missing from .env." -ForegroundColor Red
    Write-Host "Add a DATABASE_URL=... entry using the project's PostgreSQL connection settings."
    exit 1
}

if ($DatabaseUrl -match '^([''"])(.*)\1(?:\s+#.*)?$') {
    $DatabaseUrl = $Matches[2]
}
elseif ($DatabaseUrl -match '^[''"]|[''"]$') {
    Write-Host "ERROR: DATABASE_URL has unmatched surrounding quotes in .env." -ForegroundColor Red
    exit 1
}
else {
    $DatabaseUrl = ($DatabaseUrl -replace '\s+#.*$', '').TrimEnd()
}

if ([string]::IsNullOrWhiteSpace($DatabaseUrl)) {
    Write-Host "ERROR: DATABASE_URL is empty in .env." -ForegroundColor Red
    exit 1
}

$env:DATABASE_URL = $DatabaseUrl

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Write-Host "ERROR: The project virtual environment was not found at .venv\Scripts\python.exe." -ForegroundColor Red
    Write-Host "Create or restore the project virtual environment before starting the API."
    exit 1
}

$PreviousErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $Python -c "import uvicorn" 2>$null | Out-Null
    $UvicornImportExitCode = $LASTEXITCODE
}
catch {
    $UvicornImportExitCode = 1
}
finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
}

if ($UvicornImportExitCode -ne 0) {
    Write-Host "ERROR: Uvicorn is not installed or cannot be imported in the project virtual environment." -ForegroundColor Red
    Write-Host "Install the project's declared dependencies manually, then run this launcher again."
    exit 1
}

Write-Host "Starting SmartMeter AI API"
Write-Host "Project: $ProjectRoot"
Write-Host "Address: http://127.0.0.1:8000"
Write-Host "Swagger UI: http://127.0.0.1:8000/docs"
Write-Host "Health: http://127.0.0.1:8000/health"
Write-Host "Stop the server with Ctrl+C."

& $Python -m uvicorn smartmeter.api.main:app --host 127.0.0.1 --port 8000
$UvicornExitCode = $LASTEXITCODE
exit $UvicornExitCode
