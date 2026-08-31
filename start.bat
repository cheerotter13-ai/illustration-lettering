@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
  echo 请先安装 Python 3.10+，并勾选 Add python.exe to PATH。
  echo https://www.python.org/downloads/
  pause
  exit /b 1
)
echo 正在安装依赖（只需几秒到一两分钟）...
python -m pip install -q -r requirements-b.txt
if errorlevel 1 (
  echo pip 安装失败。
  pause
  exit /b 1
)
echo 打开本机网页 http://127.0.0.1:8765/
python web_app.py
pause
