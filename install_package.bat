@echo off

:: 禁用快速编辑模式
reg add "HKCU\Console" /v QuickEdit /t REG_DWORD /d 0 /f

setlocal enabledelayedexpansion

:: Get the directory of the script
set "SCRIPT_DIR=%~dp0"

:: Change the working directory to the script directory
cd /d "%SCRIPT_DIR%"

rem set PYTHON_EXECUTABLE=C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe
set PYTHON_EXECUTABLE=python

<nul (
    rem 为避免和目标机器原有的python环境冲突，用conda做python环境隔离
    conda activate Panding
    python -m pip install --no-index --find-links=".\packages" -r requirements.txt
    rem  %PYTHON_EXECUTABLE% -m pip install --no-index --find-links=".\packages" -r requirements.txt
)

endlocal

:: 防止执行后自动退出
cmd /k