$ErrorActionPreference = 'Continue'
$root = 'D:\codes\Mustard'
$logDir = 'D:\codes\Mustard\.openclaw\logs'
$reportDir = 'D:\codes\Mustard\.openclaw\reports'
$logPath = Join-Path $logDir 'claw_team_autorun.log'
$runTs = Get-Date -Format 'yyyyMMdd_HHmmss'
$reportPath = Join-Path $reportDir ("cycle_report_{0}.md" -f $runTs)

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

function Write-Log($msg) {
  $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'
  Add-Content -Path $logPath -Value "[$ts] $msg"
}

Write-Log '===== AUTO CYCLE START ====='
Write-Log "pwd=$root"

$gitStatusBefore = @()
$runOutput = @()
$lastCommit = @()
$statusOut = @()

try {
  Set-Location $root
  Write-Log 'git status --short'
  $gitStatusBefore = git status --short 2>&1
  if ($gitStatusBefore) { Add-Content -Path $logPath -Value $gitStatusBefore } else { Write-Log '(clean)' }

  Write-Log 'run: python -m scripts.claw_team_control run --auto-sync-git'
  $runOutput = python -m scripts.claw_team_control run --auto-sync-git 2>&1
  if ($runOutput) { Add-Content -Path $logPath -Value $runOutput }

  Write-Log 'last commit:'
  $lastCommit = git log --oneline -n 1 2>&1
  if ($lastCommit) { Add-Content -Path $logPath -Value $lastCommit }

  Write-Log 'next: python -m scripts.claw_team_control status'
  $statusOut = python -m scripts.claw_team_control status 2>&1
  if ($statusOut) { Add-Content -Path $logPath -Value $statusOut }
}
catch {
  Write-Log ("ERROR: " + $_.Exception.Message)
}
finally {
  Write-Log '===== AUTO CYCLE END ====='

  $report = @(
    "# Mustard Auto Cycle Report - $runTs",
    "",
    "- generated_at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "- script: scripts/auto_team_cycle.ps1",
    "",
    "## Git Status (Before)",
    '```text',
    $(if ($gitStatusBefore) { ($gitStatusBefore -join "`n") } else { '(clean)' }),
    '```',
    "",
    "## Cycle Run Output",
    '```text',
    $(if ($runOutput) { ($runOutput -join "`n") } else { '(no output)' }),
    '```',
    "",
    "## Last Commit",
    '```text',
    $(if ($lastCommit) { ($lastCommit -join "`n") } else { '(no commit info)' }),
    '```',
    "",
    "## Team Status",
    '```text',
    $(if ($statusOut) { ($statusOut -join "`n") } else { '(no status output)' }),
    '```'
  )

  Set-Content -Path $reportPath -Value ($report -join "`r`n") -Encoding UTF8
  Write-Log "report=$reportPath"
}
