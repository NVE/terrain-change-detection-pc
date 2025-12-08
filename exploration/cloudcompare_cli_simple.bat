@echo off
REM ============================================================================
REM Simple CloudCompare CLI Pipeline: Load -> ICP -> M3C2
REM ============================================================================
REM 
REM A minimal example showing the core CloudCompare CLI workflow.
REM Edit the paths below to match your data.
REM
REM For the full-featured version with argument parsing, see:
REM   cloudcompare_cli_pipeline.bat
REM ============================================================================

REM --- EDIT THESE PATHS ---
set "REF=data\synthetic\synthetic_area\2015\data\synthetic_tile_01.laz"
set "MOV=data\synthetic\synthetic_area\2020\data\synthetic_tile_01.laz"
set "OUT=data\synthetic\synthetic_area\outputs\2020_aligned_simple.laz"
set "M3C2_PARAMS=data\synthetic\synthetic_area\outputs\m3c2_params.txt"

REM --- CloudCompare executable ---
set "CC=C:\Program Files\CloudCompare\CloudCompare.exe"

REM --- Create output directory if needed ---
for %%D in ("%OUT%") do mkdir "%%~dpD" 2>nul

REM --- Build command file ---
set "CMDFILE=%TEMP%\cc_simple_%RANDOM%.txt"

(
    echo # Simple CloudCompare Pipeline
    echo -VERBOSITY 2
    echo -AUTO_SAVE OFF
    echo -NO_TIMESTAMP
    echo.
    echo # Load clouds: moving first, then reference
    echo -O "%~dp0..\%MOV%"
    echo -O "%~dp0..\%REF%"
    echo.
    echo # ICP Registration
    echo -ICP -ITER 60 -OVERLAP 80 -RANDOM_SAMPLING_LIMIT 60000
    echo.
    echo # M3C2 Distance
    echo -M3C2 "%~dp0..\%M3C2_PARAMS%"
    echo.
    echo # Export
    echo -C_EXPORT_FMT LAS
    echo -EXT laz
    echo -SELECT_ENTITIES -FIRST 1 -CLOUD
    echo -SAVE_CLOUDS FILE "%~dp0..\%OUT%"
    echo.
    echo -CLEAR
) > "%CMDFILE%"

echo.
echo === CloudCompare Simple Pipeline ===
echo Reference: %REF%
echo Moving:    %MOV%
echo Output:    %OUT%
echo.

REM --- Execute ---
"%CC%" -SILENT -COMMAND_FILE "%CMDFILE%"

REM --- Cleanup ---
del "%CMDFILE%" 2>nul

echo.
if exist "%~dp0..\%OUT%" (
    echo [SUCCESS] Output saved: %OUT%
) else (
    echo [ERROR] Output file not created
)
