$ErrorActionPreference = 'Continue'
$root = 'D:\codes\Mustard'
$logDir = 'D:\codes\Mustard\.openclaw\logs'
$logPath = Join-Path $logDir 'claw_team_autorun.log'

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log($msg) {
  $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'
  Add-Content -Path $logPath -Value "[$ts] $msg"
}

Write-Log '===== AUTO CYCLE START ====='
Write-Log "pwd=$root"

try {
  Set-Location $root
  Write-Log 'git status --short'
  $gitStatus = git status --short 2>&1
  if ($gitStatus) { Add-Content -Path $logPath -Value $gitStatus } else { Write-Log '(clean)' }

  Write-Log 'run: python -m scripts.claw_team_control run --auto-commit --auto-push'
  $output = python -m scripts.claw_team_control run --auto-commit --auto-push 2>&1
  if ($output) { Add-Content -Path $logPath -Value $output }

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
}
