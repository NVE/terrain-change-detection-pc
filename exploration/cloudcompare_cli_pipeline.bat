@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem CloudCompare CLI pipeline (Windows .bat)
rem Usage:
rem   scripts\cloudcompare_cli_pipeline.bat ^
rem     <REF_T1.las|laz|ply|e57> ^
rem     <MOV_T2.las|laz|ply|e57> ^
rem     <OUT_ALIGNED.las|laz|ply|e57> ^
rem     [/ITER:60] [/OVERLAP:80] [/SAMPLE:60000] [/MAXDIST:5] [/VERB:2]
rem
rem Notes:
rem - Ensure CloudCompare is on PATH (or set CLOUDCOMPARE_BIN to full path of CloudCompare.exe)
rem - LAZ export requires LASzip support in your CloudCompare build

if "%~1"=="" goto :usage
if "%~2"=="" goto :usage
if "%~3"=="" goto :usage

set "REFPATH=%~f1"
set "MOVPATH=%~f2"
set "OUTPATH=%~f3"

rem Defaults
set "ITER=60"
set "OVERLAP=80"
set "SAMPLE=60000"
set "MAXDIST="
set "VERBOSITY=2"

rem Parse optional switches after the 3 required args
shift & shift & shift
:parse
if "%~1"=="" goto :build
set "ARG=%~1"
if /I "!ARG:~0,7!"=="/ITER:"    set "ITER=!ARG:~7!"     & shift & goto :parse
if /I "!ARG:~0,9!"=="/OVERLAP:" set "OVERLAP=!ARG:~9!"  & shift & goto :parse
if /I "!ARG:~0,8!"=="/SAMPLE:"  set "SAMPLE=!ARG:~8!"   & shift & goto :parse
if /I "!ARG:~0,9!"=="/MAXDIST:" set "MAXDIST=!ARG:~9!"  & shift & goto :parse
if /I "!ARG:~0,6!"=="/VERB:"    set "VERBOSITY=!ARG:~6!" & shift & goto :parse
if /I "!ARG!"=="/?" goto :usage
echo [WARN] Unknown option: !ARG!
shift
goto :parse

:build
if not exist "!REFPATH!" (
  echo [ERR] Reference file not found: !REFPATH!
  exit /b 1
)
if not exist "!MOVPATH!" (
  echo [ERR] Moving file not found: !MOVPATH!
  exit /b 1
)

rem Ensure output directory exists
for %%D in ("!OUTPATH!") do set "OUTDIR=%%~dpD"
if not exist "!OUTDIR!" mkdir "!OUTDIR!" >nul 2>nul

rem Determine export format from output extension
for %%E in ("!OUTPATH!") do set "EXT=%%~xE"
set "EXTLOW=!EXT:.=!."
set "EXTLOW=!EXTLOW:~1!"  rem remove leading dot
set "FMT=PLY"
set "EXTOPT="
if /I "!EXTLOW!"=="laz" ( set "FMT=LAS" & set "EXTOPT=-EXT laz" )
if /I "!EXTLOW!"=="las" ( set "FMT=LAS" & set "EXTOPT=-EXT las" )
if /I "!EXTLOW!"=="ply" ( set "FMT=PLY" & set "EXTOPT=" )
if /I "!EXTLOW!"=="e57" ( set "FMT=E57" & set "EXTOPT=" )

rem Create temporary command file
set "CMDFILE=%TEMP%\cc_cli_%RANDOM%%RANDOM%.txt"
(
  echo -VERBOSITY !VERBOSITY!
  echo -AUTO_SAVE OFF
  echo -O "!MOVPATH!"
  echo -O "!REFPATH!"
  echo -ICP -ITER !ITER! -OVERLAP !OVERLAP! -RANDOM_SAMPLING_LIMIT !SAMPLE! -MIN_ERROR_DIFF 1e-6
  if defined MAXDIST (echo -C2C_DIST -MAX_DIST !MAXDIST!) else (echo -C2C_DIST)
  echo -C_EXPORT_FMT !FMT!
  if defined EXTOPT echo !EXTOPT!
  echo -SELECT_ENTITIES -FIRST 1 -CLOUD
  echo -SAVE_CLOUDS FILE "!OUTPATH!"
  echo -CLEAR
) > "!CMDFILE!"

echo [cloudcompare-cli] Command file: "!CMDFILE!"
for /f "usebackq delims=" %%L in ("!CMDFILE!") do echo   %%L

rem Resolve CloudCompare executable
set "CCBIN=CloudCompare"
if defined CLOUDCOMPARE_BIN set "CCBIN=%CLOUDCOMPARE_BIN%"

echo [cloudcompare-cli] Running: "%CCBIN%" -SILENT -COMMAND_FILE "!CMDFILE!"
"%CCBIN%" -SILENT -COMMAND_FILE "!CMDFILE!"
set "RC=%ERRORLEVEL%"

del /q "!CMDFILE!" >nul 2>nul

if not "%RC%"=="0" (
  echo [ERR] CloudCompare exited with code %RC%.
  exit /b %RC%
)

if not exist "!OUTPATH!" (
  echo [ERR] Expected output file not found: !OUTPATH!
  exit /b 2
)

echo [cloudcompare-cli] Done. Output: !OUTPATH!
exit /b 0

:usage
echo.
echo Usage:
echo   %~nx0 ^<REF^> ^<MOV^> ^<OUT^> [/ITER:N] [/OVERLAP:P] [/SAMPLE:K] [/MAXDIST:D] [/VERB:V]
echo.
echo Examples:
echo   %~nx0 data\synthetic\synthetic_area\2015\data\synthetic_tile_01.laz ^
echo         data\synthetic\synthetic_area\2020\data\synthetic_tile_01.laz ^
echo         data\synthetic\synthetic_area\outputs\2020_aligned_with_c2c.laz ^
echo         /ITER:60 /OVERLAP:80 /SAMPLE:60000 /MAXDIST:5
echo.
echo Notes: set CLOUDCOMPARE_BIN if CloudCompare.exe is not on PATH.
exit /b 1

