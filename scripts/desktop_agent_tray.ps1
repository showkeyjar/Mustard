Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = "python"
$pythonGuiExe = "pythonw"

function Invoke-CarmControl {
    param(
        [string[]]$ArgsList
    )
    & $pythonExe "-m" "scripts.desktop_agent_control" @ArgsList
}

function Show-StatusBalloon {
    param(
        [System.Windows.Forms.NotifyIcon]$NotifyIcon
    )
    $statusJson = Invoke-CarmControl @("status", "--json")
    if (-not $statusJson) {
        return
    }
    $payload = $statusJson | ConvertFrom-Json
    $title = if ($payload.running) { "CARM Desktop Agent: Running" } else { "CARM Desktop Agent: Stopped" }
    $goalText = if ($payload.current_goal) { "目标: $($payload.current_goal)" } else { "目标: 未确认" }
    $proactiveText = if ($payload.proactive_status) { "状态: $($payload.proactive_status)" } else { "状态: 无" }
    $budgetText = "预算: $($payload.proactive_budget_remaining)"
    $body = if ($payload.running) {
        "PID: $($payload.pid)`n$goalText`n$proactiveText`n$budgetText"
    } else {
        "Agent is not running.`n$goalText`n$proactiveText`n$budgetText"
    }
    $tooltip = if ($payload.current_goal) {
        "CARM: $($payload.current_goal)"
    } elseif ($payload.running) {
        "CARM Desktop Agent (运行中)"
    } else {
        "CARM Desktop Agent"
    }
    $NotifyIcon.Text = if ($tooltip.Length -gt 63) { $tooltip.Substring(0, 63) } else { $tooltip }
    $NotifyIcon.BalloonTipTitle = $title
    $NotifyIcon.BalloonTipText = $body
    $NotifyIcon.ShowBalloonTip(3000)
}

$notifyIcon = New-Object System.Windows.Forms.NotifyIcon
$notifyIcon.Icon = [System.Drawing.SystemIcons]::Information
$notifyIcon.Text = "CARM Desktop Agent"
$notifyIcon.Visible = $true

$menu = New-Object System.Windows.Forms.ContextMenuStrip

$startItem = $menu.Items.Add("Start Agent")
$startItem.Add_Click({
    Invoke-CarmControl @("start") | Out-Null
    Show-StatusBalloon -NotifyIcon $notifyIcon
})

$stopItem = $menu.Items.Add("Stop Agent")
$stopItem.Add_Click({
    Invoke-CarmControl @("stop") | Out-Null
    Show-StatusBalloon -NotifyIcon $notifyIcon
})

$statusItem = $menu.Items.Add("Show Status")
$statusItem.Add_Click({
    Show-StatusBalloon -NotifyIcon $notifyIcon
})

$snapshotItem = $menu.Items.Add("Show Snapshot")
$snapshotItem.Add_Click({
    $snapshotText = Invoke-CarmControl @("snapshot")
    if (-not $snapshotText) {
        return
    }
    [System.Windows.Forms.MessageBox]::Show($snapshotText, "CARM 状态快照") | Out-Null
})

$chatItem = $menu.Items.Add("Open Bridge Chat")
$chatItem.Add_Click({
    Start-Process $pythonGuiExe -ArgumentList "-m scripts.desktop_bridge_chat" -WorkingDirectory $repoRoot
})

$openDataItem = $menu.Items.Add("Open Data Folder")
$openDataItem.Add_Click({
    Start-Process explorer.exe (Join-Path $repoRoot "data")
})

$startupItem = $menu.Items.Add("Install Startup")
$startupItem.Add_Click({
    Invoke-CarmControl @("install-startup") | Out-Null
    $notifyIcon.BalloonTipTitle = "CARM Desktop Agent"
    $notifyIcon.BalloonTipText = "Startup shortcut installed."
    $notifyIcon.ShowBalloonTip(3000)
})

$removeStartupItem = $menu.Items.Add("Remove Startup")
$removeStartupItem.Add_Click({
    Invoke-CarmControl @("remove-startup") | Out-Null
    $notifyIcon.BalloonTipTitle = "CARM Desktop Agent"
    $notifyIcon.BalloonTipText = "Startup shortcut removed."
    $notifyIcon.ShowBalloonTip(3000)
})

$menu.Items.Add("-") | Out-Null
$exitItem = $menu.Items.Add("Exit Tray")
$exitItem.Add_Click({
    $notifyIcon.Visible = $false
    [System.Windows.Forms.Application]::Exit()
})

$notifyIcon.ContextMenuStrip = $menu
$notifyIcon.Add_DoubleClick({
    Show-StatusBalloon -NotifyIcon $notifyIcon
})

[System.Windows.Forms.Application]::Run()
