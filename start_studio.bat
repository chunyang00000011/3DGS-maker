@echo off
chcp 65001 >nul
setlocal EnableExtensions
title Real2Sim-GS Studio

set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "PYTHONUTF8=1"

if defined REAL2SIM_PYTHON if exist "%REAL2SIM_PYTHON%" set "PYTHON_EXE=%REAL2SIM_PYTHON%"
if not defined PYTHON_EXE (
    where py >nul 2>nul
    if not errorlevel 1 (
        for /f "delims=" %%P in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
    )
)
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYTHON_EXE if exist "C:\Users\21241\AppData\Local\Programs\Python\Python311\python.exe" set "PYTHON_EXE=C:\Users\21241\AppData\Local\Programs\Python\Python311\python.exe"
if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
)

if not exist "%PROJECT_DIR%\pyproject.toml" (
    echo [ERROR] 未找到项目目录: "%PROJECT_DIR%"
    echo 请检查项目文件夹是否被移动或重命名。
    pause
    exit /b 1
)

if not defined PYTHON_EXE (
    echo [ERROR] 未找到可用 Python 3.11+ 解释器（可设置环境变量 REAL2SIM_PYTHON 指定路径）。
    pause
    exit /b 1
)
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python 解释器路径无效: "%PYTHON_EXE%"
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"

echo ==============================================================
echo   Real2Sim-GS Studio  ^|  3DGS Real-to-Sim 交互式桌面工作台
echo ==============================================================
echo   工作目录 : %PROJECT_DIR%
echo   运行环境 : Python 3.11 + RTX CUDA + pycolmap + gsplat
echo   使用说明 : 请将此黑色后台窗口最小化保留（勿直接关闭黑框），关闭桌面主窗口后将自动退出
echo ==============================================================
echo.

if "%~1"=="" (
    "%PYTHON_EXE%" -m real2sim_gs.api.cli ui --desktop
) else (
    "%PYTHON_EXE%" -m real2sim_gs.api.cli ui %*
)

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Real2Sim-GS Studio 异常退出 (ExitCode=%EXIT_CODE%)，请查看上方日志信息。
    pause
    exit /b %EXIT_CODE%
)
