# AI 종목 발굴 시스템

52주 신고가 돌파 종목을 자동으로 발굴하고, 머신러닝 모델로 수익률을 예측하는 시스템입니다.

## 주요 기능

- **52주 신고가 돌파 스크리닝**: KRX 전체 종목에서 52주 신고가를 돌파한 종목 자동 탐지
- **ML 기반 수익률 예측**: LightGBM 모델로 20거래일 내 예상 최고 수익률 예측
- **성공 확률 계산**: 돌파 종가 대비 10% 이상 상승할 확률 예측
- **BUY/HOLD/PASS 추천**: 성공확률과 예상수익률 기반 자동 추천

## 모델 정보

| 모델 | 목적 | 성능 |
|------|------|------|
| LightGBM Regressor | 예상 최고수익률 예측 | R² = 2.9% |
| LightGBM Classifier | 10% 상승 확률 예측 | Precision = 50.1% |

- 학습 데이터: 2022~2025년 약 4,000건의 52주 신고가 돌파 사례
- 14개 피처: 돌파강도, 거래량비, 변동성, 시장수익률, RS 등

## 사용 방법

### 웹 인터페이스
```bash
python run_server.py
# http://localhost:8000 접속
```

### 스크리너 실행
```bash
python ai_screener.py
# data/ai_screener_results.json 생성
```

## 기술 스택

- **백엔드**: Python, pykrx, LightGBM
- **프론트엔드**: Vanilla JS, CSS Grid/Flexbox
- **데이터**: KRX 시장 데이터, 네이버 금융

## 라이브 데모

[https://hsh2578.github.io/ai-stock-screener/](https://hsh2578.github.io/ai-stock-screener/)

## 주의사항

- 본 시스템은 투자 참고용이며, 투자 결정의 책임은 본인에게 있습니다.
- 모델 성능은 과거 데이터 기반이며, 미래 수익을 보장하지 않습니다.
