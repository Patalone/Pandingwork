@echo off

rem 禁用快速编辑模式
reg add "HKCU\Console" /v QuickEdit /t REG_DWORD /d 0 /f

setlocal enabledelayedexpansion

rem Get the directory of the script
set "SCRIPT_DIR=%~dp0"

rem Change the working directory to the script directory
cd /d "%SCRIPT_DIR%"

rem 读取 config.yml 文件，找到包含 serve_port 的行
for /f "tokens=1,2 delims=:" %%a in ('findstr /r /c:"serve_port" config.yml') do (
    set servePort=%%b
)

rem 去除前后的空格
for /f "tokens=* delims= " %%a in ("%servePort%") do set servePort=%%a

rem 打印 servePort 变量的值
echo servePort: %servePort%

rem set PYTHON_EXECUTABLE=C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe
set PYTHON_EXECUTABLE=python

<nul (
     rem start /B python app.py
     conda active Panding
     start /B %PYTHON_EXECUTABLE% app.py
)

endlocal

rem 防止执行后自动退出
cmd /k