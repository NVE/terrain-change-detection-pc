# Benchmark script - times each workflow execution
# Runs run_workflow.py with different configs and measures execution time

$configs = @(
    @{Name="DoD Sequential"; Config="config/profiles/bench_dod_seq.yaml"},
    @{Name="DoD Parallel"; Config="config/profiles/bench_dod_par.yaml"},
    @{Name="C2C Sequential"; Config="config/profiles/bench_c2c_seq.yaml"},
    @{Name="C2C Parallel"; Config="config/profiles/bench_c2c_par.yaml"},
    @{Name="M3C2 Sequential"; Config="config/profiles/bench_m3c2_seq.yaml"},
    @{Name="M3C2 Parallel"; Config="config/profiles/bench_m3c2_par.yaml"}
)

$results = @()

foreach ($test in $configs) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Running: $($test.Name)" -ForegroundColor Cyan
    Write-Host "Config: $($test.Config)" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    $start = Get-Date
    
    # Run workflow and capture output
    $output = uv run .\scripts\run_workflow.py --config $test.Config 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    
    $end = Get-Date
    $elapsed = ($end - $start).TotalSeconds
    
    Write-Host "`nCompleted in $([math]::Round($elapsed, 2)) seconds" -ForegroundColor Green
    Write-Host "Exit code: $exitCode`n"
    
    # Extract method-specific timing from output
    $methodTime = $null
    $tiles = $null
    
    if ($output -match "(\d+)\s+tiles\s+in\s+(\d+\.?\d*)\s*s") {
        $tiles = $Matches[1]
        $methodTime = $Matches[2]
    }
    
    $results += [PSCustomObject]@{
        Test = $test.Name
        TotalTime = [math]::Round($elapsed, 2)
        MethodTime = $methodTime
        Tiles = $tiles
        ExitCode = $exitCode
    }
}

# Display results table
Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "BENCHMARK RESULTS" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Yellow

$results | Format-Table -AutoSize

# Calculate speedups
Write-Host "`n========================================" -ForegroundColor Yellow
Write-Host "SPEEDUP ANALYSIS" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Yellow

$methods = @("DoD", "C2C", "M3C2")

foreach ($method in $methods) {
    $seq = $results | Where-Object { $_.Test -eq "$method Sequential" }
    $par = $results | Where-Object { $_.Test -eq "$method Parallel" }
    
    if ($seq -and $par) {
        $speedup = [math]::Round($seq.TotalTime / $par.TotalTime, 2)
        $efficiency = [math]::Round(($speedup / 11) * 100, 1)  # Assuming 11 workers
        
        Write-Host "$method Performance:" -ForegroundColor Cyan
        Write-Host "  Sequential: $($seq.TotalTime)s"
        Write-Host "  Parallel:   $($par.TotalTime)s"
        Write-Host "  Speedup:    ${speedup}x"
        Write-Host "  Efficiency: ${efficiency}% (11 workers)"
        Write-Host ""
    }
}

# Save to JSON
$results | ConvertTo-Json | Out-File "benchmark_results.json"
Write-Host "Results saved to benchmark_results.json`n" -ForegroundColor Green
