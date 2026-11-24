# run_sft_pipeline.ps1 - Option A: generate SFT data, clean it, merge pretrain, train SFT.
# RESUMABLE: each run reads the files on disk (plus data\pipeline_state.json and data\pipeline.log)
# to see which steps are already done, skips them, and continues where the last run stopped.
#
#   .\run_sft_pipeline.ps1                  # continue where it left off
#   .\run_sft_pipeline.ps1 -Status          # only show what is done / pending, run nothing
#   .\run_sft_pipeline.ps1 -From train      # force re-running from a step onwards
#   .\run_sft_pipeline.ps1 -Yes             # don't ask y/N questions (unattended)

param(
    [switch]$Status,
    [ValidateSet('pull', 'test', 'generate', 'dedup', 'validate', 'merge', 'train')]
    [string]$From,
    [switch]$Yes,
    [string]$Teacher = 'qwen2.5-coder:7b',
    [int]$General = 400
)

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
. .\.venv\Scripts\Activate.ps1

$Steps     = @('pull', 'test', 'generate', 'dedup', 'validate', 'merge', 'train')
$StateFile = 'data\pipeline_state.json'
$LogFile   = 'data\pipeline.log'
$Gen       = 'data\sft_generated.jsonl'
$GenState  = 'data\sft_generated.state.json'
$Clean     = 'data\sft_clean.jsonl'
$Adapter   = 'output\pretrain\adapter_model.safetensors'
$SftOut    = 'output\sft'
$SftFinal  = 'output\sft\adapter_model.safetensors'

# ------------------------------------------------------------------ helpers
function Step($title) { Write-Host "`n=== $title ===" -ForegroundColor Cyan }

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

function Check($what) {
    if ($LASTEXITCODE -ne 0) {
        Log "$what FAILED (exit code $LASTEXITCODE)"
        Write-Host "FAILED: $what (exit code $LASTEXITCODE). Fix it and run this script again: it resumes here." -ForegroundColor Red
        exit 1
    }
}

function Confirm($question) {
    if ($Yes) { return }
    $answer = Read-Host "$question [y/N]"
    if ($answer -notmatch '^(y|yes|s|si)$') {
        Log "stopped by user at: $question"
        Write-Host "Stopped. Run the script again to continue from here."
        exit 0
    }
}

function Forced($step) {
    if (-not $From) { return $false }
    return ([array]::IndexOf($Steps, $step) -ge [array]::IndexOf($Steps, $From))
}

function Time-Of($path) {
    if (Test-Path $path) { return (Get-Item $path).LastWriteTime }
    return [datetime]::MinValue
}

function Count-Rows($path) {
    if (Test-Path $path) { return @(Get-Content $path -Encoding UTF8).Count }
    return 0
}

function Load-State {
    $h = @{}
    if (Test-Path $StateFile) {
        $obj = Get-Content $StateFile -Raw | ConvertFrom-Json
        foreach ($p in $obj.PSObject.Properties) { $h[$p.Name] = $p.Value }
    }
    return $h
}

function Save-State($key) {
    $h = Load-State
    $h[$key] = (Get-Date).ToString('o')
    $h | ConvertTo-Json | Set-Content $StateFile -Encoding UTF8
}

function State-Time($key) {
    $h = Load-State
    if ($h.ContainsKey($key)) { return [datetime]::Parse($h[$key]) }
    return [datetime]::MinValue
}

function Ollama-Has($model) {
    try {
        $list = & ollama list 2>$null | Out-String
        return ($list -match [regex]::Escape($model))
    } catch { return $false }
}

function Gen-Progress {
    $out = & python generate_sft.py --general $General --dry-run 2>&1 | Out-String
    if ($out -match '(\d+) total, (\d+) already done, (\d+) to do') {
        return @{ Total = [int]$Matches[1]; Done = [int]$Matches[2]; Todo = [int]$Matches[3] }
    }
    Write-Host $out
    Write-Host "Could not read generation progress (see output above)." -ForegroundColor Red
    exit 1
}

function Merged-Model {
    if (-not (Test-Path 'merged')) { return $null }
    return Get-ChildItem 'merged' -Filter '*.safetensors' | Sort-Object LastWriteTime | Select-Object -Last 1
}

function Latest-Checkpoint {
    if (-not (Test-Path $SftOut)) { return $null }
    return Get-ChildItem $SftOut -Directory -Filter 'checkpoint-*' |
        Sort-Object { [int]($_.Name -replace '^checkpoint-', '') } | Select-Object -Last 1
}

function Train-Done {
    $m = Merged-Model
    if (-not $m -or -not (Test-Path $SftFinal)) { return $false }
    $t = Time-Of $SftFinal
    return (($t -ge (Time-Of $Clean)) -and ($t -ge $m.LastWriteTime))
}

# ------------------------------------------------------------------ status, read from disk
function Get-Status {
    $g = Gen-Progress
    $genDone = ($g.Todo -eq 0)
    $s = [ordered]@{}

    if ($genDone) { $s['pull'] = @('not needed', 'generation already complete') }
    elseif (Ollama-Has $Teacher) { $s['pull'] = @('done', "$Teacher is in Ollama") }
    else { $s['pull'] = @('pending', "$Teacher not pulled (or Ollama not running)") }

    if ((Test-Path $GenState) -or ((State-Time 'test') -gt [datetime]::MinValue)) {
        $s['test'] = @('done', 'generation already started')
    } else { $s['test'] = @('pending', '') }

    if ($genDone) { $s['generate'] = @('done', "$($g.Total) chunks, $(Count-Rows $Gen) rows") }
    else { $s['generate'] = @('pending', "$($g.Done)/$($g.Total) chunks done, $($g.Todo) to do, $(Count-Rows $Gen) rows so far") }

    if ($genDone -and (Test-Path $Clean) -and ((Time-Of $Clean) -ge (Time-Of $Gen))) {
        $s['dedup'] = @('done', "$(Count-Rows $Clean) rows")
    } else { $s['dedup'] = @('pending', '') }

    if ((Test-Path $Clean) -and ((State-Time 'validate') -ge (Time-Of $Clean))) { $s['validate'] = @('done', '') }
    else { $s['validate'] = @('pending', '') }

    $m = Merged-Model
    if ($m -and ($m.LastWriteTime -ge (Time-Of $Adapter))) { $s['merge'] = @('done', "merged\$($m.Name)") }
    else { $s['merge'] = @('pending', '') }

    $ck = Latest-Checkpoint
    if (Train-Done) { $s['train'] = @('done', $SftFinal) }
    elseif ($ck) { $s['train'] = @('partial', "can resume from $SftOut\$($ck.Name)") }
    else { $s['train'] = @('pending', '') }

    return @{ Steps = $s; Gen = $g }
}

function Show-Status($st) {
    Write-Host "`nPipeline status (read from disk):" -ForegroundColor Cyan
    foreach ($k in $st.Steps.Keys) {
        $v = $st.Steps[$k]
        $color = 'White'
        if ($v[0] -eq 'done') { $color = 'Green' }
        elseif ($v[0] -eq 'not needed') { $color = 'DarkGray' }
        elseif ($v[0] -eq 'partial') { $color = 'Yellow' }
        Write-Host ("  {0,-9} {1,-11} {2}" -f $k, $v[0], $v[1]) -ForegroundColor $color
    }
    if (Test-Path $LogFile) {
        Write-Host "`nLast log entries ($LogFile):" -ForegroundColor Cyan
        Get-Content $LogFile -Tail 8 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
    }
}

# ------------------------------------------------------------------ main
$st = Get-Status
Show-Status $st
if ($Status) { exit 0 }
$steps = $st.Steps
if ($From) {
    Write-Host "`nForcing re-run from step: $From" -ForegroundColor Yellow
    Log "run started (from $From)"
} else { Log "run started" }

# 1. Teacher model
if ((Forced 'pull') -or ($steps['pull'][0] -eq 'pending')) {
    Step "1. Pull teacher model"
    ollama pull $Teacher; Check "pull"
    Log "pull done"
}

# 2. Short test, only before the first generation run
if ((Forced 'test') -or ($steps['test'][0] -eq 'pending')) {
    Step "2. 8-chunk test (separate file)"
    Remove-Item data\sft_test.* -ErrorAction SilentlyContinue
    python generate_sft.py --out data\sft_test.jsonl --limit 8; Check "test"
    python generate_sft.py --out data\sft_test.jsonl --show 10; Check "test show"
    Save-State 'test'
    Log "test done"
    Write-Host "`nRead the samples above: does the code use real constructors/methods from your packages?" -ForegroundColor Yellow
    Confirm "Continue with the full generation?"
}

# 3. Generation (generate_sft.py itself resumes chunk by chunk)
if ((Forced 'generate') -or ($steps['generate'][0] -ne 'done')) {
    Step "3. Generation ($($st.Gen.Todo) chunks to do)"
    Log "generate started ($($st.Gen.Done)/$($st.Gen.Total) chunks already done)"
    python generate_sft.py --general $General; Check "generate"
    $g = Gen-Progress
    if ($g.Todo -gt 0) {
        Log "generate incomplete: $($g.Todo) chunks failed"
        Write-Host "$($g.Todo) chunks failed (Ollama errors). Run the script again to retry them." -ForegroundColor Red
        exit 1
    }
    Log "generate done ($(Count-Rows $Gen) rows)"
}

# 4. Dedup (re-runs whenever the generated file is newer than the clean one)
if ((Forced 'dedup') -or ((Time-Of $Clean) -lt (Time-Of $Gen))) {
    Step "4. Dedup"
    soup data dedup $Gen --threshold 0.85 -o $Clean; Check "dedup"
    Log "dedup done ($(Count-Rows $Clean) rows)"
}

# 5. Validate (re-runs whenever the clean file changed since the last validation)
if ((Forced 'validate') -or ((State-Time 'validate') -lt (Time-Of $Clean))) {
    Step "5. Validate"
    soup data validate $Clean --format chatml; Check "validate"
    Save-State 'validate'
    Log "validate done"
}

# From here on the GPU must be free: unload the teacher model from VRAM
try { & ollama stop $Teacher 2>$null | Out-Null } catch { }

# 6. Merge pretrain adapter into the base model
$m = Merged-Model
if ((Forced 'merge') -or (-not $m) -or ($m.LastWriteTime -lt (Time-Of $Adapter))) {
    Step "6. Merge pretrain adapter"
    if (Test-Path 'merged') {
        Confirm "Delete the existing .\merged (about 6 GB) and merge again?"
        Remove-Item 'merged' -Recurse -Force
    }
    Log "merge started"
    soup merge --adapter .\output\pretrain --output .\merged; Check "merge"
    Log "merge done"
}

# 7. Train SFT (resumes from the latest checkpoint if a previous run was interrupted)
if ((Forced 'train') -or -not (Train-Done)) {
    Step "7. Train SFT"
    $ck = Latest-Checkpoint
    if ((Forced 'train') -and (Test-Path $SftOut)) {
        $keep = "output\sft_old_$(Get-Date -Format 'yyyyMMdd_HHmm')"
        Move-Item $SftOut $keep
        Write-Host "Previous training run moved to $keep" -ForegroundColor Yellow
        $ck = $null
    }
    if ($ck) {
        Write-Host "Found an interrupted run: $SftOut\$($ck.Name)" -ForegroundColor Yellow
        Confirm "Resume training from $($ck.Name)? (N = stop; use -From train to start over)"
        Log "train resumed from $($ck.Name)"
        soup train --config soup.sft.yaml --resume $ck.FullName --yes; Check "train"
    } else {
        Confirm "Start SFT training (several hours)?"
        Log "train started"
        soup train --config soup.sft.yaml --yes; Check "train"
    }
    Log "train done"
}

Log "pipeline complete"
Write-Host "`nAll steps done. SFT adapter in .\$SftOut" -ForegroundColor Green
Write-Host "Next: soup export --model .\$SftOut --format gguf --quant q8_0" -ForegroundColor Green
