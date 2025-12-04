@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM CloudCompare CLI Pipeline: Load -> ICP -> M3C2
REM ============================================================================
REM 
REM This batch script demonstrates terrain change detection using
REM CloudCompare's command-line interface. It performs:
REM 
REM   1. Load reference (T1) and moving (T2) point clouds
REM   2. ICP Registration (align moving cloud to reference)
REM   3. M3C2 Distance computation (robust multi-scale change detection)
REM   4. Export aligned cloud with M3C2 scalar fields
REM 
REM Usage:
REM   cloudcompare_cli_pipeline.bat <REF.laz> <MOV.laz> <OUT.laz> /M3C2:params.txt [OPTIONS]
REM 
REM Options:
REM   /M3C2:file     - M3C2 parameter file (REQUIRED)
REM   /ITER:N        - ICP max iterations (default: 60)
REM   /OVERLAP:P     - ICP expected overlap percent 10-100 (default: 80)
REM   /SAMPLE:K      - ICP random sampling limit (default: 60000)
REM   /VERB:V        - Verbosity 0-4 (default: 2)
REM   /SOR           - Enable Statistical Outlier Removal
REM   /NOTS          - Disable timestamp in output filename
REM 
REM Examples:
REM   cloudcompare_cli_pipeline.bat ref.laz mov.laz out.laz /M3C2:params.txt
REM   cloudcompare_cli_pipeline.bat ref.laz mov.laz out.laz /M3C2:params.txt /ITER:100
REM 
REM Environment:
REM   Set CLOUDCOMPARE_BIN to CloudCompare.exe path if not on PATH
REM 
REM Note: M3C2 parameter file can be created via:
REM   CloudCompare GUI: Tools -> Distances -> M3C2 -> Save parameters
REM 
REM Author: Terrain Change Detection Team
REM Date: December 2024
REM ============================================================================

REM Check minimum required arguments
if "%~1"=="" goto :usage
if "%~2"=="" goto :usage
if "%~3"=="" goto :usage

REM Store required arguments (with full paths)
set "REFPATH=%~f1"
set "MOVPATH=%~f2"
set "OUTPATH=%~f3"

REM Default parameters
set "ITER=60"
set "OVERLAP=80"
set "SAMPLE=60000"
set "VERBOSITY=2"
set "ENABLE_SOR="
set "NO_TIMESTAMP=1"
set "M3C2_PARAMS="

REM Parse optional switches (shift past the 3 required args first)
shift & shift & shift

:parse_args
if "%~1"=="" goto :validate
set "ARG=%~1"

REM Parse each option
if /I "!ARG:~0,6!"=="/M3C2:"     ( set "M3C2_PARAMS=!ARG:~6!"  & shift & goto :parse_args )
if /I "!ARG:~0,6!"=="/ITER:"     ( set "ITER=!ARG:~6!"     & shift & goto :parse_args )
if /I "!ARG:~0,9!"=="/OVERLAP:"  ( set "OVERLAP=!ARG:~9!"  & shift & goto :parse_args )
if /I "!ARG:~0,8!"=="/SAMPLE:"   ( set "SAMPLE=!ARG:~8!"   & shift & goto :parse_args )
if /I "!ARG:~0,6!"=="/VERB:"     ( set "VERBOSITY=!ARG:~6!" & shift & goto :parse_args )
if /I "!ARG!"=="/SOR"            ( set "ENABLE_SOR=1"      & shift & goto :parse_args )
if /I "!ARG!"=="/NOTS"           ( set "NO_TIMESTAMP=1"    & shift & goto :parse_args )
if /I "!ARG!"=="/?"              ( goto :usage )
if /I "!ARG!"=="/HELP"           ( goto :usage )

echo [WARN] Unknown option: !ARG!
shift
goto :parse_args

:validate
REM Validate input files exist
if not exist "!REFPATH!" (
    echo [ERROR] Reference file not found: !REFPATH!
    exit /b 1
)
if not exist "!MOVPATH!" (
    echo [ERROR] Moving file not found: !MOVPATH!
    exit /b 1
)

REM Validate M3C2 parameter file
if not defined M3C2_PARAMS (
    echo [ERROR] M3C2 parameter file required! Use /M3C2:filename.txt
    echo         Create one via CloudCompare GUI: Tools -^> Distances -^> M3C2 -^> Save parameters
    exit /b 1
)
if not exist "!M3C2_PARAMS!" (
    echo [ERROR] M3C2 parameter file not found: !M3C2_PARAMS!
    exit /b 1
)

REM Extract output directory and ensure it exists
for %%D in ("!OUTPATH!") do set "OUTDIR=%%~dpD"
if not exist "!OUTDIR!" (
    echo [INFO] Creating output directory: !OUTDIR!
    mkdir "!OUTDIR!" >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Failed to create output directory
        exit /b 1
    )
)

REM Determine export format from output extension
for %%E in ("!OUTPATH!") do set "OUTEXT=%%~xE"
set "OUTEXT=!OUTEXT:~1!"

REM Set format based on extension
set "FMT=PLY"
set "EXTOPT="
if /I "!OUTEXT!"=="laz" ( set "FMT=LAS" & set "EXTOPT=-EXT laz" )
if /I "!OUTEXT!"=="las" ( set "FMT=LAS" & set "EXTOPT=-EXT las" )
if /I "!OUTEXT!"=="ply" ( set "FMT=PLY" )
if /I "!OUTEXT!"=="e57" ( set "FMT=E57" )
if /I "!OUTEXT!"=="bin" ( set "FMT=BIN" )

:build_commands
REM Create temporary command file
set "CMDFILE=%TEMP%\cc_m3c2_pipeline_%RANDOM%%RANDOM%.txt"

echo [INFO] Building CloudCompare command file...
echo [INFO] Workflow: Load -^> ICP -^> M3C2

(
    echo # CloudCompare CLI Pipeline: Load -^> ICP -^> M3C2
    echo # Reference: !REFPATH!
    echo # Moving: !MOVPATH!
    echo # M3C2 Params: !M3C2_PARAMS!
    echo # Output: !OUTPATH!
    echo.
    echo # Global settings
    echo -VERBOSITY !VERBOSITY!
    echo -AUTO_SAVE OFF
    if defined NO_TIMESTAMP echo -NO_TIMESTAMP
    echo.
    echo # Load point clouds ^(moving first, then reference^)
    echo -O "!MOVPATH!"
    echo -O "!REFPATH!"
    
    REM Optional: Statistical Outlier Removal
    if defined ENABLE_SOR (
        echo.
        echo # Statistical Outlier Removal
        echo -SOR 8 2
    )
    
    echo.
    echo # ICP Registration
    echo -ICP -ITER !ITER! -OVERLAP !OVERLAP! -RANDOM_SAMPLING_LIMIT !SAMPLE! -MIN_ERROR_DIFF 1e-6
    
    echo.
    echo # M3C2 Distance Computation
    echo -M3C2 "!M3C2_PARAMS!"
    
    echo.
    echo # Export settings
    echo -C_EXPORT_FMT !FMT!
    if defined EXTOPT echo !EXTOPT!
    
    echo.
    echo # Save aligned cloud with M3C2 distances
    echo -SELECT_ENTITIES -FIRST 1 -CLOUD
    echo -SAVE_CLOUDS FILE "!OUTPATH!"
    
    echo.
    echo # Cleanup
    echo -CLEAR
) > "!CMDFILE!"

REM Display the command file for debugging
echo.
echo [INFO] Command file contents:
echo ----------------------------------------
type "!CMDFILE!"
echo ----------------------------------------
echo.

:execute
REM Find CloudCompare executable
set "CCBIN=CloudCompare"
if defined CLOUDCOMPARE_BIN set "CCBIN=!CLOUDCOMPARE_BIN!"

REM Try to find CloudCompare if not explicitly set
where CloudCompare >nul 2>nul
if errorlevel 1 (
    REM Check common installation paths
    if exist "C:\Program Files\CloudCompare\CloudCompare.exe" (
        set "CCBIN=C:\Program Files\CloudCompare\CloudCompare.exe"
    ) else if exist "C:\Program Files (x86)\CloudCompare\CloudCompare.exe" (
        set "CCBIN=C:\Program Files (x86)\CloudCompare\CloudCompare.exe"
    )
)

echo [INFO] Using CloudCompare: !CCBIN!
echo [INFO] Executing pipeline...
echo.

REM Execute CloudCompare
"!CCBIN!" -SILENT -COMMAND_FILE "!CMDFILE!"
set "RC=!ERRORLEVEL!"

REM Cleanup temporary file
del /q "!CMDFILE!" >nul 2>nul

REM Check result
if not "!RC!"=="0" (
    echo.
    echo [ERROR] CloudCompare exited with code !RC!
    exit /b !RC!
)

if not exist "!OUTPATH!" (
    echo.
    echo [ERROR] Expected output file not found: !OUTPATH!
    echo         CloudCompare may have added a timestamp suffix.
    echo         Check the output directory: !OUTDIR!
    exit /b 2
)

REM Success
echo.
echo ============================================================
echo [SUCCESS] Pipeline completed successfully!
echo ============================================================
echo.
echo Output file: !OUTPATH!
for %%F in ("!OUTPATH!") do echo File size: %%~zF bytes
echo.

exit /b 0

:usage
echo.
echo ============================================================
echo CloudCompare CLI Pipeline: Load -^> ICP -^> M3C2
echo ============================================================
echo.
echo Usage:
echo   %~nx0 ^<REF^> ^<MOV^> ^<OUT^> /M3C2:params.txt [OPTIONS]
echo.
echo Required Arguments:
echo   ^<REF^>          Reference point cloud (e.g., 2015.laz)
echo   ^<MOV^>          Moving point cloud (e.g., 2020.laz)
echo   ^<OUT^>          Output file path (e.g., aligned.laz)
echo   /M3C2:file     M3C2 parameter file (REQUIRED)
echo.
echo Options:
echo   /ITER:N        ICP max iterations (default: 60)
echo   /OVERLAP:P     ICP expected overlap %% 10-100 (default: 80)
echo   /SAMPLE:K      ICP random sampling limit (default: 60000)
echo   /VERB:V        Verbosity 0=verbose to 4=errors (default: 2)
echo   /SOR           Enable Statistical Outlier Removal
echo   /NOTS          Disable timestamp in output filename
echo   /?             Show this help message
echo.
echo Creating M3C2 Parameter File:
echo   1. Open CloudCompare GUI
echo   2. Load two point clouds
echo   3. Go to Tools -^> Distances -^> M3C2
echo   4. Configure parameters
echo   5. Click "Save parameters" to create the file
echo.
echo Examples:
echo.
echo   Basic usage:
echo     %~nx0 ref_2015.laz mov_2020.laz aligned_2020.laz /M3C2:m3c2_params.txt
echo.
echo   With ICP tuning:
echo     %~nx0 ref.laz mov.laz out.laz /M3C2:params.txt /ITER:100 /OVERLAP:70
echo.
echo   Full example:
echo     %~nx0 data\2015\cloud.laz data\2020\cloud.laz output\aligned.laz ^
echo           /M3C2:m3c2_params.txt /ITER:60 /OVERLAP:80 /SOR
echo.
echo Environment Variables:
echo   CLOUDCOMPARE_BIN   Path to CloudCompare.exe (if not on PATH)
echo.
echo Notes:
echo   - Output format is determined by file extension (.laz, .las, .ply, .e57)
echo   - LAZ output requires CloudCompare with LASzip support
echo   - Cloud order matters: first cloud is aligned to second
echo   - M3C2 provides robust change detection with uncertainty estimation
echo.

exit /b 1
