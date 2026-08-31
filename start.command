#!/bin/bash
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "请先安装 Python 3.10+（macOS: brew install python）"
  read -r _
  exit 1
fi
echo "正在安装依赖…"
python3 -m pip install -q -r requirements-b.txt
echo "打开本机网页 http://127.0.0.1:8765/"
python3 web_app.py
