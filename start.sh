#!/bin/sh
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements-b.txt
echo "打开本机网页 http://127.0.0.1:8765/"
exec python3 web_app.py
