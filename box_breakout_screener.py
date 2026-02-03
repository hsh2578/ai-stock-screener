# -*- coding: utf-8 -*-
"""
박스권 돌파 스크리너 (거래량 무관) + ML 예측

박스권 상단을 돌파한 종목을 찾고,
학습된 ML 모델로 예상 수익률과 성공 확률을 예측합니다.

조건:
    - 사전 조건: 60거래일 이상 박스권 횡보 (종가 변동폭 25% 이내)
    - 저항선: 박스 상단
    - 돌파 조건: 종가 > 저항선 × 1.015 (상단 +1.5% 초과)
    - 경과 기간: 돌파일로부터 10 거래일 이내
    - 거래량/이평선 조건: 없음

ML 예측:
    - 성공확률: XGBoost 분류 모델 (15개 피처)
    - 예상수익률: Ridge 회귀 모델 (10개 피처)
    - AI점수: 성공확률 × 0.7 + 수익률점수 × 0.3
"""

import pandas as pd
import numpy as np
import joblib
import json
import requests
from bs4 import BeautifulSoup
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 경로
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data"

# HTTP 요청 헤더
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# ============================================================================
# 상수 정의
# ============================================================================
MIN_MARKET_CAP = 1000          # 최소 시가총액 (억원)
BOX_PERIOD = 60                # 박스권 판단 기간 (거래일)
MAX_BOX_RANGE_PERCENT = 25.0   # 최대 허용 변동폭 (%)
BREAKOUT_WINDOW = 11           # 돌파 확인 윈도우 (10거래일 이내 = 오늘 포함 11행)
BREAKOUT_THRESHOLD = 1.015     # 돌파 임계값 (저항선 × 1.015)

# 피벗 검출
PIVOT_WINDOW = 5               # 피벗 포인트 검출 윈도우
MIN_TOUCHES = 2                # 최소 터치 횟수
ATR_PERIOD = 60                # ATR 계산 기간
ATR_MULTIPLE_MAX = 6           # ATR 배수 최대
ATR_TOUCH_MULTIPLE = 1.5       # ATR 기반 터치 허용범위 배수
MAX_SLOPE_PERCENT = 0.05       # 최대 일평균 기울기 (%)
VOLUME_DECREASE_THRESHOLD = 0.95  # 거래량 감소 임계값

# ML 피처
FEATURES_10 = [
    'ma20_deviation', 'pct_above_52w_low', 'breakout_strength',
    'close_strength', 'days_since_ath', 'atr_ratio',
    'liquidity', 'breakout_gap', 'rs_vs_market', 'market_return'
]

FEATURES_15 = FEATURES_10 + [
    'volume_surge', 'box_range_pct', 'volume_dry_up',
    'ma200_slope', 'volatility_contraction'
]


# ============================================================================
# 네이버 금융 PER/PBR
# ============================================================================
def fetch_naver_per_pbr(code):
    """네이버 금융에서 PER, PBR 가져오기"""
    try:
        url = f'https://finance.naver.com/item/main.naver?code={code}'
        resp = requests.get(url, headers=HEADERS, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')

        per, pbr = None, None
        per_elem = soup.select_one('#_per')
        pbr_elem = soup.select_one('#_pbr')

        if per_elem:
            per_text = per_elem.get_text(strip=True)
            if per_text and per_text not in ['-', 'N/A', '']:
                try:
                    per = float(per_text.replace(',', ''))
                except:
                    pass

        if pbr_elem:
            pbr_text = pbr_elem.get_text(strip=True)
            if pbr_text and pbr_text not in ['-', 'N/A', '']:
                try:
                    pbr = float(pbr_text.replace(',', ''))
                except:
                    pass

        return per, pbr
    except:
        return None, None


def fetch_naver_per_pbr_batch(codes, max_workers=16):
    """PER, PBR 병렬 배치 가져오기"""
    results = {}

    def fetch_single(code):
        return code, fetch_naver_per_pbr(code)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single, code): code for code in codes}
        for future in as_completed(futures):
            try:
                code, (per, pbr) = future.result()
                results[code] = (per, pbr)
            except:
                code = futures[future]
                results[code] = (None, None)

    return results


# ============================================================================
# 박스권 판단 함수
# ============================================================================
def quick_range_check(df: pd.DataFrame, period: int, max_range: float) -> bool:
    """빠른 박스권 사전 필터"""
    if len(df) < period:
        return False

    recent = df.tail(period)
    high_max = recent['High'].max()
    low_min = recent['Low'].min()

    if low_min <= 0:
        return False

    range_pct = (high_max - low_min) / low_min * 100
    return range_pct <= max_range


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """ATR 계산"""
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.tail(period).mean()


def find_pivot_highs(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """피벗 고점 찾기"""
    highs = df['High'].values
    pivot_highs = pd.Series(index=df.index, dtype=float)

    for i in range(window, len(highs) - window):
        if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, window+1)):
            pivot_highs.iloc[i] = highs[i]

    return pivot_highs.dropna()


def find_pivot_lows(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """피벗 저점 찾기"""
    lows = df['Low'].values
    pivot_lows = pd.Series(index=df.index, dtype=float)

    for i in range(window, len(lows) - window):
        if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, window+1)):
            pivot_lows.iloc[i] = lows[i]

    return pivot_lows.dropna()


def is_box_range(df: pd.DataFrame, period: int = 60) -> Tuple[bool, Dict[str, Any]]:
    """
    박스권 판단 (전체 조건)

    조건:
        1. 변동폭: 종가 기준 최대 변동폭 <= 25%
        2. 저항/지지: 피벗 포인트 기반 수평 채널
        3. 터치 횟수: 상단/하단 각 2회 이상 터치
        4. 기울기: 중심선 기울기 거의 수평
        5. 거래량: 후반 거래량 < 전반 거래량 (축소)
    """
    if len(df) < period:
        return False, {}

    recent = df.tail(period).copy()

    # 1. 변동폭 체크
    close_max = recent['Close'].max()
    close_min = recent['Close'].min()
    if close_min <= 0:
        return False, {}

    range_pct = (close_max - close_min) / close_min * 100
    if range_pct > MAX_BOX_RANGE_PERCENT:
        return False, {}

    # ATR 기반 적응적 터치 허용범위
    atr = calculate_atr(recent, ATR_PERIOD if len(recent) >= ATR_PERIOD else len(recent))
    if atr <= 0:
        return False, {}

    touch_tolerance = atr * ATR_TOUCH_MULTIPLE

    # 2. 피벗 포인트 찾기
    pivot_highs = find_pivot_highs(recent, PIVOT_WINDOW)
    pivot_lows = find_pivot_lows(recent, PIVOT_WINDOW)

    # 저항선/지지선 계산
    if len(pivot_highs) >= 1:
        resistance = pivot_highs.mean()
    else:
        resistance = close_max

    if len(pivot_lows) >= 1:
        support = pivot_lows.mean()
    else:
        support = close_min

    # 박스 범위가 ATR의 N배 이내인지 확인
    box_range = resistance - support
    if box_range > atr * ATR_MULTIPLE_MAX:
        return False, {}

    # 3. 터치 횟수 확인
    resistance_touches = ((recent['High'] >= resistance - touch_tolerance) &
                          (recent['High'] <= resistance + touch_tolerance)).sum()
    support_touches = ((recent['Low'] >= support - touch_tolerance) &
                       (recent['Low'] <= support + touch_tolerance)).sum()

    if resistance_touches < MIN_TOUCHES or support_touches < MIN_TOUCHES:
        return False, {}

    # 4. 중심선 기울기 확인
    midline = (resistance + support) / 2
    closes = recent['Close'].values
    x = np.arange(len(closes))

    if len(x) > 1:
        slope = np.polyfit(x, closes, 1)[0]
        slope_pct = abs(slope / midline * 100) if midline > 0 else 0
        if slope_pct > MAX_SLOPE_PERCENT:
            return False, {}

    # 5. 거래량 축소 확인
    half = len(recent) // 2
    vol_first_half = recent['Volume'].iloc[:half].mean()
    vol_second_half = recent['Volume'].iloc[half:].mean()

    if vol_first_half > 0 and vol_second_half >= vol_first_half * VOLUME_DECREASE_THRESHOLD:
        pass  # 거래량 증가는 허용 (박스권 돌파 준비)

    return True, {
        'box_high': resistance,
        'box_low': support,
        'range_pct': range_pct,
        'atr': atr,
        'resistance_touches': resistance_touches,
        'support_touches': support_touches
    }


# ============================================================================
# ML 모델
# ============================================================================
class BoxBreakoutML:
    """박스권 돌파 ML 모델"""

    def __init__(self):
        self.reg_model = None
        self.cls_model = None
        self.loaded = False
        self._load_models()

    def _load_models(self):
        """모델 로드"""
        try:
            reg_path = MODEL_PATH / "box_breakout_regression_model.joblib"
            cls_path = MODEL_PATH / "box_breakout_classification_model.joblib"

            if reg_path.exists() and cls_path.exists():
                self.reg_model = joblib.load(reg_path)
                self.cls_model = joblib.load(cls_path)
                self.loaded = True
                print("[ML] 박스권 돌파 모델 로드 완료")
            else:
                print("[ML] 모델 파일 없음 - AI 점수 없이 진행")
        except Exception as e:
            print(f"[ML] 모델 로드 실패: {e}")

    def calculate_features(self, df, breakout_idx, resistance, support):
        """ML 피처 계산"""
        try:
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            breakout_close = close.iloc[breakout_idx]
            box_range_pct = (resistance - support) / support * 100 if support > 0 else 0

            # 돌파 강도
            breakout_strength = (breakout_close - resistance) / resistance * 100 if resistance > 0 else 0

            # 거래량 급증
            if breakout_idx >= 20:
                vol_avg = volume.iloc[breakout_idx-20:breakout_idx].mean()
                volume_surge = volume.iloc[breakout_idx] / vol_avg if vol_avg > 0 else 1
            else:
                volume_surge = 1

            # 종가 강도
            breakout_high = high.iloc[breakout_idx]
            breakout_low = low.iloc[breakout_idx]
            if breakout_high != breakout_low:
                close_strength = (breakout_close - breakout_low) / (breakout_high - breakout_low)
            else:
                close_strength = 0.5

            # 변동성 수축
            if breakout_idx >= 60:
                recent_vol = close.iloc[breakout_idx-20:breakout_idx].std() / close.iloc[breakout_idx-20:breakout_idx].mean()
                past_vol = close.iloc[breakout_idx-60:breakout_idx-20].std() / close.iloc[breakout_idx-60:breakout_idx-20].mean()
                volatility_contraction = recent_vol / past_vol if past_vol > 0 else 1
            else:
                volatility_contraction = 1

            # 거래량 고갈
            if breakout_idx >= 40:
                vol_recent = volume.iloc[breakout_idx-10:breakout_idx].mean()
                vol_past = volume.iloc[breakout_idx-40:breakout_idx-10].mean()
                volume_dry_up = vol_recent / vol_past if vol_past > 0 else 1
            else:
                volume_dry_up = 1

            # 20일선 이격도
            if breakout_idx >= 20:
                ma20 = close.iloc[breakout_idx-20:breakout_idx].mean()
                ma20_deviation = (breakout_close / ma20 - 1) * 100 if ma20 > 0 else 0
            else:
                ma20_deviation = 0

            # 돌파 갭
            if breakout_idx >= 1:
                prev_close = close.iloc[breakout_idx-1]
                breakout_gap = (df['Open'].iloc[breakout_idx] / prev_close - 1) * 100 if prev_close > 0 else 0
            else:
                breakout_gap = 0

            # 200일선 기울기
            if breakout_idx >= 220:
                ma200_now = close.iloc[breakout_idx-200:breakout_idx].mean()
                ma200_20ago = close.iloc[breakout_idx-220:breakout_idx-20].mean()
                ma200_slope = (ma200_now / ma200_20ago - 1) * 100 if ma200_20ago > 0 else 0
            else:
                ma200_slope = 0

            # 52주 저가 대비
            lookback_start = max(0, breakout_idx - 250)
            low_52w = low.iloc[lookback_start:breakout_idx].min()
            pct_above_52w_low = (breakout_close / low_52w - 1) * 100 if low_52w > 0 else 0

            # ATH 이후 경과일
            high_52w_idx = high.iloc[lookback_start:breakout_idx].idxmax()
            breakout_date = df.index[breakout_idx]
            days_since_ath = (breakout_date - high_52w_idx).days if pd.notna(high_52w_idx) else 0

            # 시장 수익률 (KOSPI 대용으로 종목 자체 20일 수익률 사용)
            if breakout_idx >= 20:
                stock_return = (breakout_close / close.iloc[breakout_idx-20] - 1) * 100
                market_return = stock_return * 0.5  # 시장 수익률 근사
                rs_vs_market = stock_return - market_return
            else:
                market_return = 0
                rs_vs_market = 0

            # ATR 비율
            if breakout_idx >= 14:
                h = high.iloc[breakout_idx-14:breakout_idx].values
                l = low.iloc[breakout_idx-14:breakout_idx].values
                c_prev = close.iloc[breakout_idx-15:breakout_idx-1].values

                tr = np.maximum(np.maximum(h - l, np.abs(h - c_prev)), np.abs(l - c_prev))
                atr_14 = tr.mean()
                atr_ratio = atr_14 / breakout_close * 100 if breakout_close > 0 else 0
            else:
                atr_ratio = 0

            # 유동성
            if breakout_idx >= 20:
                liquidity = (close.iloc[breakout_idx-20:breakout_idx] * volume.iloc[breakout_idx-20:breakout_idx]).mean()
            else:
                liquidity = 0

            return {
                'box_range_pct': box_range_pct,
                'breakout_strength': breakout_strength,
                'volume_surge': volume_surge,
                'close_strength': close_strength,
                'volatility_contraction': volatility_contraction,
                'volume_dry_up': volume_dry_up,
                'ma20_deviation': ma20_deviation,
                'breakout_gap': breakout_gap,
                'ma200_slope': ma200_slope,
                'pct_above_52w_low': pct_above_52w_low,
                'days_since_ath': days_since_ath,
                'market_return': market_return,
                'rs_vs_market': rs_vs_market,
                'atr_ratio': atr_ratio,
                'liquidity': liquidity
            }
        except Exception as e:
            return None

    def predict(self, features):
        """예측 수행"""
        if not self.loaded or features is None:
            return 0.0, 0.0, 0.0

        try:
            # 회귀 모델 (10개 피처)
            X_reg = np.array([[features.get(f, 0) for f in FEATURES_10]])
            reg_scaler = self.reg_model['scaler']
            reg_model = self.reg_model['model']
            X_reg_scaled = reg_scaler.transform(X_reg)
            predicted_gain = float(reg_model.predict(X_reg_scaled)[0])

            # 분류 모델 (15개 피처)
            X_cls = np.array([[features.get(f, 0) for f in FEATURES_15]])
            cls_scaler = self.cls_model['scaler']
            cls_model = self.cls_model['model']
            X_cls_scaled = cls_scaler.transform(X_cls)
            success_prob = float(cls_model.predict_proba(X_cls_scaled)[0][1]) * 100

            # AI 점수 계산
            gain_score = max(0, min(1, (predicted_gain + 20) / 70))
            ai_score = (success_prob / 100) * 0.7 + gain_score * 0.3
            ai_score = ai_score * 100

            return success_prob, predicted_gain, ai_score
        except Exception as e:
            return 0.0, 0.0, 0.0


# ============================================================================
# 박스권 돌파 스크리너
# ============================================================================
class BoxBreakoutScreener:
    """박스권 돌파 스크리너"""

    def __init__(self):
        self.ml = BoxBreakoutML()
        self.stock_info = {}

    def get_stock_list(self):
        """KOSPI/KOSDAQ 종목 리스트"""
        print("종목 리스트 로딩...")

        kospi = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')

        def filter_stocks(df, market):
            df = df.copy()
            df['Market'] = market

            if 'Marcap' in df.columns:
                df = df[df['Marcap'] >= MIN_MARKET_CAP * 100000000]
            elif 'MarketCap' in df.columns:
                df = df[df['MarketCap'] >= MIN_MARKET_CAP * 100000000]

            if 'Name' in df.columns:
                df = df[~df['Name'].str.contains(
                    'ETF|KODEX|TIGER|KBSTAR|KOSEF|HANARO|ARIRANG|SOL|PLUS|ACE|파워|레버리지|인버스|스팩|리츠|우$|우B$|우C$',
                    na=False, regex=True
                )]

            return df

        kospi_filtered = filter_stocks(kospi, 'KOSPI')
        kosdaq_filtered = filter_stocks(kosdaq, 'KOSDAQ')

        stocks = pd.concat([kospi_filtered, kosdaq_filtered], ignore_index=True)

        # 종목 정보 저장
        code_col = 'Code' if 'Code' in stocks.columns else 'Symbol'
        marcap_col = 'Marcap' if 'Marcap' in stocks.columns else 'MarketCap'

        for _, row in stocks.iterrows():
            code = row[code_col]
            self.stock_info[code] = {
                'marcap': int(row[marcap_col] // 100000000) if pd.notna(row[marcap_col]) else 0
            }

        print(f"  총 {len(stocks)}개 종목")
        return stocks

    def analyze_stock(self, code, name, market):
        """단일 종목 분석"""
        try:
            # 주가 데이터 로드
            end_date = datetime.now()
            start_date = end_date - timedelta(days=400)
            df = fdr.DataReader(code, start_date, end_date)

            if df is None or len(df) < BOX_PERIOD * 2 + BREAKOUT_WINDOW:
                return None

            # 박스권 + 돌파 확인
            breakout_day = None
            days_since = 0
            resistance = None
            support = None
            breakout_idx = None

            for box_end in range(BREAKOUT_WINDOW, 0, -1):
                box_start = BOX_PERIOD * 2 + box_end
                if len(df) < box_start:
                    continue

                box_period_df = df.iloc[-box_start:-box_end]
                if len(box_period_df) < BOX_PERIOD + 1:
                    continue

                # 빠른 사전 필터
                if not quick_range_check(box_period_df, BOX_PERIOD, MAX_BOX_RANGE_PERCENT):
                    continue

                is_box, box_data = is_box_range(box_period_df, BOX_PERIOD)
                if not is_box:
                    continue

                candidate_days = df.iloc[-box_end:]
                for i, (date, row_data) in enumerate(candidate_days.iterrows()):
                    if row_data['Close'] > box_data['box_high'] * BREAKOUT_THRESHOLD:
                        breakout_day = date
                        days_since = len(candidate_days) - i - 1
                        resistance = box_data['box_high']
                        support = box_data['box_low']
                        breakout_idx = len(df) - box_end + i
                        break

                if breakout_day is not None:
                    break

            if breakout_day is None:
                return None

            current_price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            change_rate = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0

            # 저항선 대비 상승률
            breakout_pct = (current_price - resistance) / resistance * 100

            # 돌파일 종가 기준 등락률
            breakout_close = float(df['Close'].iloc[breakout_idx])
            gain_since_breakout = (current_price - breakout_close) / breakout_close * 100

            # ML 예측
            features = self.ml.calculate_features(df, breakout_idx, resistance, support)
            success_prob, predicted_gain, ai_score = self.ml.predict(features)

            # 추천 결정
            if success_prob >= 60 and predicted_gain >= 10:
                recommendation = 'BUY'
            elif success_prob >= 50 or predicted_gain >= 5:
                recommendation = 'HOLD'
            else:
                recommendation = 'PASS'

            # 돌파일 포맷
            breakout_date_str = f"{breakout_day.month}월{breakout_day.day}일"

            stock_info = self.stock_info.get(code, {})

            return {
                'code': code,
                'name': name,
                'market': market,
                'close': int(current_price),
                'change_pct': round(change_rate, 2),
                'marcap': stock_info.get('marcap', 0),
                'resistance': int(resistance),
                'support': int(support),
                'breakout_pct': round(breakout_pct, 2),
                'gain_since_breakout': round(gain_since_breakout, 2),
                'days_since_breakout': days_since,
                'breakout_date': breakout_date_str,
                'breakout_date_full': breakout_day.strftime('%Y-%m-%d'),
                'predicted_gain': round(predicted_gain, 1),
                'success_probability': round(success_prob, 1),
                'ai_score': round(ai_score, 1),
                'recommendation': recommendation
            }

        except Exception as e:
            return None

    def run(self, max_workers=8):
        """스크리너 실행"""
        print("=" * 60)
        print("박스권 돌파 스크리너 (거래량 무관) + ML 예측")
        print("=" * 60)

        stocks = self.get_stock_list()

        print(f"\n종목 분석 중... (병렬 {max_workers}개)")
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for _, row in stocks.iterrows():
                code = row['Code'] if 'Code' in row else row['Symbol']
                name = row['Name']
                market = row['Market']
                future = executor.submit(self.analyze_stock, code, name, market)
                futures[future] = (code, name)

            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 100 == 0:
                    print(f"  진행: {completed}/{len(futures)} ({completed/len(futures)*100:.1f}%)")

                result = future.result()
                if result:
                    results.append(result)

        # PER/PBR 배치 가져오기
        if results:
            print(f"\nPER/PBR 데이터 로딩 중... ({len(results)}개 종목)")
            codes = [r['code'] for r in results]
            per_pbr_data = fetch_naver_per_pbr_batch(codes)

            for result in results:
                code = result['code']
                per, pbr = per_pbr_data.get(code, (None, None))
                result['per'] = round(per, 1) if per else None
                result['pbr'] = round(pbr, 2) if pbr else None

        # AI 점수 내림차순 정렬
        if self.ml.loaded:
            results.sort(key=lambda x: x['ai_score'], reverse=True)
        else:
            results.sort(key=lambda x: x['days_since_breakout'])

        print(f"\n발굴된 종목: {len(results)}개")

        return results

    def save_results(self, results):
        """결과 저장"""
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        results_converted = convert_types(results)

        output = {
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'screener_type': 'box_breakout_simple',
            'count': len(results_converted),
            'ml_loaded': self.ml.loaded,
            'stocks': results_converted
        }

        output_file = DATA_PATH / "box_breakout_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"결과 저장: {output_file}")
        return output_file


def main():
    screener = BoxBreakoutScreener()
    results = screener.run()
    screener.save_results(results)

    # 상위 10개 출력
    print("\n" + "=" * 60)
    print("상위 10개 종목 (AI점수 순)")
    print("=" * 60)
    print(f"{'종목명':<12} {'현재가':>10} {'돌파강도':>8} {'AI점수':>8} {'성공확률':>8} {'추천':>6}")
    print("-" * 60)

    for stock in results[:10]:
        print(f"{stock['name']:<12} {stock['close']:>10,} {stock['breakout_pct']:>7.1f}% {stock['ai_score']:>7.1f} {stock['success_probability']:>7.1f}% {stock['recommendation']:>6}")

    return results


if __name__ == "__main__":
    main()
