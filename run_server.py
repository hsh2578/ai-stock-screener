"""
AI 종목발굴 웹서버 실행

1. 스크리너 실행 (선택)
2. HTTP 서버 시작
"""

import subprocess
import sys
import os
from pathlib import Path
import webbrowser
import http.server
import socketserver

PORT = 8080
BASE_DIR = Path(__file__).parent

def run_screener():
    """스크리너 실행"""
    print("스크리너 실행 중...")
    result = subprocess.run([sys.executable, str(BASE_DIR / "ai_screener.py")],
                          capture_output=False, cwd=str(BASE_DIR))
    return result.returncode == 0

def start_server():
    """HTTP 서버 시작"""
    os.chdir(BASE_DIR)

    handler = http.server.SimpleHTTPRequestHandler
    handler.extensions_map.update({
        '.js': 'application/javascript',
        '.json': 'application/json',
    })

    with socketserver.TCPServer(("", PORT), handler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"\n서버 시작: {url}")
        print("브라우저에서 확인하세요.")
        print("종료: Ctrl+C\n")

        webbrowser.open(url)
        httpd.serve_forever()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-scan', action='store_true', help='스크리너 실행 건너뛰기')
    args = parser.parse_args()

    if not args.skip_scan:
        # 데이터 파일 확인
        data_file = BASE_DIR / "data" / "ai_screener_results.json"
        if not data_file.exists():
            print("데이터 파일이 없습니다. 스크리너를 실행합니다.")
            run_screener()
        else:
            response = input("스크리너를 다시 실행하시겠습니까? (y/N): ").strip().lower()
            if response == 'y':
                run_screener()

    start_server()
