param(
    [Parameter(Mandatory=$true)]
    [string]$FilePath
)

# 1. Setup Logging
$LogDir = "quality_logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = "$LogDir\safe_audit_$Timestamp.txt"

function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $msg = "[$(Get-Date -Format 'HH:mm:ss')] $Message"
    Write-Host $msg -ForegroundColor $Color
    $msg | Out-File -FilePath $LogFile -Append
}

Write-Log "--- STARTING SAFE CLEANUP: $FilePath ---" "Cyan"
"------------------------------------------------" | Out-File $LogFile -Append

# --- STEP 1: RUFF FORMAT (100% SAFE) ---
Write-Log "Step 1: Professional Formatting..." "Gray"
ruff format $FilePath *>> $LogFile
Write-Log "  [OK] Formatting Complete." "Green"

# --- STEP 2: SAFE LINTING (100% SAFE) ---
Write-Log "Step 2: Cleaning unused code & sorting imports..." "Gray"
# I: Sorts Imports, F401: Unused imports, F841: Unused variables
ruff check --select I,F401,F841 --fix $FilePath *>> $LogFile
Write-Log "  [OK] Safe-Fixes applied." "Green"

# --- STEP 3: LOGIC AUDIT (READ-ONLY) ---
Write-Log "Step 3: Running Logic Audit (Mypy)..." "Gray"
$MypyOut = mypy $FilePath --ignore-missing-imports --pretty
$MypyOut | Out-File -FilePath $LogFile -Append

if ($LASTEXITCODE -eq 0) {
    Write-Log "  [SUCCESS] LOGIC CHECK: 100% Clean." "Cyan"
} else {
    Write-Log "  [NOTICE] LOGIC CHECK: Issues found. See log for details." "Yellow"
}

Write-Log "Cleanup Finished. Your results are safe." "Green"
Write-Log "Full report: $LogFile" "Gray"