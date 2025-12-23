param(
    [Parameter(Mandatory=$true)]
    [string]$FilePath
)

# 1. Setup Logging Folder and Files
$LogDir = "quality_logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir }

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$CleanFileName = [System.IO.Path]::GetFileNameWithoutExtension($FilePath)
$LogFile = "$LogDir\log_${CleanFileName}_$Timestamp.txt"

# Helper function to log and print at the same time
function Write-Log {
    param([string]$Message, [string]$Color = "White")
    $TimestampedMessage = "[$(Get-Date -Format "HH:mm:ss")] $Message"
    Write-Host $TimestampedMessage -ForegroundColor $Color
    $TimestampedMessage | Out-File -FilePath $LogFile -Append
}

# Start the Audit
"============================================================" | Out-File -FilePath $LogFile
"  QUALITY AUDIT REPORT: $FilePath"                           | Out-File -FilePath $LogFile
"  TIMESTAMP: $(Get-Date)"                                     | Out-File -FilePath $LogFile
"============================================================" | Out-File -FilePath $LogFile

Write-Log "🚀 Starting Gold Standard Cleanup for $FilePath" "Cyan"

# --- Step 1: Autotyping ---
Write-Log "--- Step 1: Injecting Type Hints (Autotyping) ---" "Gray"
python -m autotyping $FilePath --none-return --scalar-return --int-param --float-param --str-param *>> $LogFile
if ($LASTEXITCODE -eq 0) { Write-Log "  ✅ Autotyping complete." "Green" } else { Write-Log "  ⚠️ Autotyping reported issues (check log)." "Yellow" }

# --- Step 2: Ruff Check ---
Write-Log "--- Step 2: Fixing Linting & Docstrings (Ruff) ---" "Gray"
ruff check --fix --unsafe-fixes --show-fixes $FilePath *>> $LogFile
if ($LASTEXITCODE -eq 0) { Write-Log "  ✅ Ruff Check complete." "Green" } else { Write-Log "  ⚠️ Ruff found items that need manual review." "Yellow" }

# --- Step 3: Ruff Format ---
Write-Log "--- Step 3: Global Formatting (Ruff) ---" "Gray"
ruff format $FilePath *>> $LogFile
Write-Log "  ✅ Formatting applied." "Green"

# --- Step 4: Mypy ---
Write-Log "--- Step 4: Static Analysis Check (Mypy) ---" "Gray"
# Running Mypy and capturing the exact error count
$MypyOut = mypy $FilePath --ignore-missing-imports --install-types --non-interactive 2>&1
$MypyOut | Out-File -FilePath $LogFile -Append
if ($LASTEXITCODE -eq 0) {
    Write-Log "  ✅ MYPY PASSED: 100% Type Safety." "Cyan"
} else {
    Write-Log "  ❌ MYPY FAILED: Type errors detected. See log for details." "Red"
}

Write-Log "`n✨ Audit Finished. Full report saved to: $LogFile" "Green"