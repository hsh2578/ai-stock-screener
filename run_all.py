# -*- coding: utf-8 -*-
"""
AI 종목 발굴 - 전체 스크리너 실행

두 개의 스크리너를 순차적으로 실행합니다:
1. 52주 신고가 돌파 (ai_screener.py)
2. 박스권 돌파 (box_breakout_screener.py)
"""

import sys
from datetime import datetime

def main():
    print("=" * 70)
    print("AI 종목 발굴 시스템 - 전체 스크리너 실행")
    print("=" * 70)
    print(f"실행 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. 52주 신고가 스크리너
    print("[1/2] 52주 신고가 돌파 스크리너 실행...")
    print("-" * 70)
    try:
        from ai_screener import main as run_52w
        results_52w = run_52w()
        print(f"\n52주 신고가 완료: {len(results_52w)}개 종목 발굴")
    except Exception as e:
        print(f"52주 신고가 스크리너 오류: {e}")
        results_52w = []

    print()

    # 2. 박스권 돌파 스크리너
    print("[2/2] 박스권 돌파 스크리너 실행...")
    print("-" * 70)
    try:
        from box_breakout_screener import main as run_box
        results_box = run_box()
        print(f"\n박스권 돌파 완료: {len(results_box)}개 종목 발굴")
    except Exception as e:
        print(f"박스권 돌파 스크리너 오류: {e}")
        results_box = []

    # 요약
    print()
    print("=" * 70)
    print("전체 실행 완료!")
    print("=" * 70)
    print(f"• 52주 신고가: {len(results_52w)}개")
    print(f"• 박스권 돌파: {len(results_box)}개")
    print()
    print("웹페이지 확인: python -m http.server 8000")
    print("브라우저에서: http://localhost:8000")


if __name__ == "__main__":
    main()
