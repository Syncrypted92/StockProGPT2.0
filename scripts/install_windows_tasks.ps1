# Register Windows Task Scheduler jobs for StockPro paper trading.
# Run:
#   powershell -ExecutionPolicy Bypass -File scripts\install_windows_tasks.ps1
#
# Schedules are defined in US Eastern (market time) and converted to YOUR
# PC's local clock automatically.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Runner = Join-Path $Root "scripts\run_session.py"

if (-not (Test-Path $Python)) {
    throw "Python venv not found at $Python. Create .venv and pip install -e . first."
}

function Convert-EasternTimeToLocal {
    param([string]$TimeHhMm)
    $et = [System.TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
    $localTz = [System.TimeZoneInfo]::Local
    $todayEt = [System.TimeZoneInfo]::ConvertTimeFromUtc((Get-Date).ToUniversalTime(), $et).Date
    $parts = $TimeHhMm.Split(":")
    $etDt = $todayEt.AddHours([int]$parts[0]).AddMinutes([int]$parts[1])
    $utc = [System.TimeZoneInfo]::ConvertTimeToUtc($etDt, $et)
    $localDt = [System.TimeZoneInfo]::ConvertTimeFromUtc($utc, $localTz)
    return $localDt.ToString("HH:mm")
}

function Unregister-IfExists {
    param([string]$Name)
    $existing = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
    }
}

function New-WeeklyTriggerAtLocal {
    param([string]$LocalTime)
    return New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $LocalTime
}

function Register-StockProTask {
    param(
        [string]$Name,
        [string]$Phase,
        [string[]]$EasternTimes
    )
    Unregister-IfExists -Name $Name

    $arg = '"' + $Runner + '" ' + $Phase + ' --submit'
    $action = New-ScheduledTaskAction -Execute $Python -Argument $arg -WorkingDirectory $Root

    $triggers = @()
    $localList = @()
    foreach ($etTime in $EasternTimes) {
        $localTime = Convert-EasternTimeToLocal -TimeHhMm $etTime
        $localList += ($etTime + "ET->" + $localTime)
        $triggers += New-WeeklyTriggerAtLocal -LocalTime $localTime
    }

    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew
    $desc = "StockPro Alpaca paper " + $Phase + " [" + ($localList -join ", ") + "]"
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $triggers -Settings $settings -Description $desc | Out-Null
    Write-Host ("Registered " + $Name + " phase=" + $Phase + " times=" + ($EasternTimes -join ","))
}

function Register-RepeatingWeekdayTask {
    param(
        [string]$Name,
        [string]$Phase,
        [string]$EasternStart,
        [int]$IntervalMinutes,
        [int]$DurationHours
    )
    Unregister-IfExists -Name $Name
    $localStart = Convert-EasternTimeToLocal -TimeHhMm $EasternStart
    $arg = '"' + $Runner + '" ' + $Phase + ' --submit'
    $action = New-ScheduledTaskAction -Execute $Python -Argument $arg -WorkingDirectory $Root

    # Base weekly trigger at start, then repeat
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $localStart
    $once = New-ScheduledTaskTrigger -Once -At $localStart `
        -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
        -RepetitionDuration (New-TimeSpan -Hours $DurationHours)
    $trigger.Repetition = $once.Repetition

    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew
    $desc = "StockPro $Phase every ${IntervalMinutes}m from ${EasternStart}ET for ${DurationHours}h (local start $localStart)"
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings -Description $desc | Out-Null
    Write-Host ("Registered " + $Name + " every " + $IntervalMinutes + "m from " + $EasternStart + "ET")
}

# Remove legacy task names if present
Unregister-IfExists -Name "StockPro-Afternoon"
Unregister-IfExists -Name "StockPro-Exits-Hourly"
Unregister-IfExists -Name "StockPro-ZeroDTE"

# Swing morning scan (deprioritized; still available)
Register-StockProTask -Name "StockPro-Morning" -Phase "morning" -EasternTimes @("09:50")

# Primary: SPY 0DTE day lane — scan every 5 minutes 09:35–15:30 ET (~6h)
Register-RepeatingWeekdayTask -Name "StockPro-SpyDay-Scan" -Phase "spy-day" -EasternStart "09:35" -IntervalMinutes 5 -DurationHours 6

# Exits every 5 minutes during session (covers spy_day TP/SL/force-flat)
Register-RepeatingWeekdayTask -Name "StockPro-SpyDay-Exits" -Phase "afternoon" -EasternStart "09:40" -IntervalMinutes 5 -DurationHours 6

# End-of-day grade + spy-day report
Register-StockProTask -Name "StockPro-EOD" -Phase "eod" -EasternTimes @("16:20")

# Weekly 5m bar refresh (Sunday evening ET) — includes extended/premarket cache
Unregister-IfExists -Name "StockPro-RefreshBars"
$localSun = Convert-EasternTimeToLocal -TimeHhMm "18:00"
$refreshArg = '"' + (Join-Path $Root "scripts\refresh_spy_bars.py") + '" --full'
$refreshAction = New-ScheduledTaskAction -Execute $Python -Argument $refreshArg -WorkingDirectory $Root
$refreshTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At $localSun
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName "StockPro-RefreshBars" -Action $refreshAction -Trigger $refreshTrigger -Settings $settings -Description "Refresh SPY 5m Alpaca bars (RTH+extended)" | Out-Null
Write-Host "Registered StockPro-RefreshBars Sunday 18:00ET->$localSun"

# Weekday premarket refresh before the open (07:00 ET ~ 06:00 CT)
Unregister-IfExists -Name "StockPro-Premarket-Refresh"
$localPm = Convert-EasternTimeToLocal -TimeHhMm "07:00"
$pmArg = '"' + (Join-Path $Root "scripts\refresh_spy_bars.py") + '"'
$pmAction = New-ScheduledTaskAction -Execute $Python -Argument $pmArg -WorkingDirectory $Root
$pmTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $localPm
Register-ScheduledTask -TaskName "StockPro-Premarket-Refresh" -Action $pmAction -Trigger $pmTrigger -Settings $settings -Description "Incremental SPY bars incl. premarket before open" | Out-Null
Write-Host "Registered StockPro-Premarket-Refresh 07:00ET->$localPm"

Write-Host ""
Write-Host "Done. Verify with: Get-ScheduledTask -TaskName StockPro-*"
Write-Host "Kill switch: set TRADING_HALTED=true in .env"
Write-Host "Spy-day submit requires artifacts/spy_day_backtest_latest.json gate_ok=true"
Write-Host "Re-run this script after DST changes if triggers look off."
