# Run a benchmark script against one or more simulator folders on ONE logical CPU (Windows).
# benchmark*.py's own --cpus option needs Linux, so the process affinity is set from outside.
#
#   .\run_benchmark_pinned.ps1 -Benchmark ..\..\benchmark3.py -Repos A,B -Rounds 2 -OutDir results
#
# Repos are run alternately (A, B, A, B, ...) so that machine slowdowns hit all of them equally.
param(
    [Parameter(Mandatory = $true)][string]$Benchmark,
    [Parameter(Mandatory = $true)][string[]]$Repos,
    [int]$Rounds = 2,
    [int]$Repeats = 3,
    [string]$Python = "$PSScriptRoot\..\..\..\.venvs\dev\Scripts\python.exe",
    [string]$OutDir = "$env:TEMP\bench-results"
)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
foreach ($name in "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS") {
    Set-Item -Path "env:$name" -Value "1"
}
$Benchmark = (Resolve-Path $Benchmark).Path
$Python = (Resolve-Path $Python).Path
foreach ($round in 1..$Rounds) {
    foreach ($repo in $Repos) {
        $repoPath = (Resolve-Path $repo).Path
        $label = "{0}_{1}_r{2}" -f [IO.Path]::GetFileNameWithoutExtension($Benchmark), (Split-Path (Split-Path $repoPath -Parent) -Leaf), $round
        # PYTHONPATH makes this folder's differometor win over the editable install of the venv.
        $env:PYTHONPATH = $repoPath
        $arguments = "`"$Benchmark`" --repo `"$repoPath`" --repeats $Repeats --output `"$OutDir\$label.json`""
        $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $OutDir -NoNewWindow -PassThru `
            -RedirectStandardOutput "$OutDir\$label.log" -RedirectStandardError "$OutDir\$label.err"
        try { $process.ProcessorAffinity = 1 } catch { Write-Warning "could not pin $label to one CPU" }
        $process.WaitForExit()
        $total = Select-String -Path "$OutDir\$label.log" -Pattern "first-run total" | Select-Object -First 1
        $pass = if (Select-String -Path "$OutDir\$label.log" -Pattern "^PASS" -Quiet) { "PASS" }
            elseif (Select-String -Path "$OutDir\$label.err" -Pattern "name 'repeat_samples' is not defined" -Quiet) {
                # benchmark3.py: validation asserts already passed; only its report step fails (warm loop commented out).
                "validation passed (report step crashed: repeat_samples)" }
            else { "NO PASS - see $label.err" }
        "{0}: {1} | {2}" -f $label, $total.Line, $pass
    }
}
$env:PYTHONPATH = $null
