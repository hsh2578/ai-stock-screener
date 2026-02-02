"""
AI 종목 발굴 - 52주 신고가 돌파 + ML 예측

52주 신고가를 돌파한 종목을 찾고,
학습된 ML 모델로 예상 수익률과 성공 확률을 예측합니다.
"""

import pandas as pd
import numpy as np
import pickle
import json
import requests
from bs4 import BeautifulSoup
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# HTTP 요청 헤더
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# 경로
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data"

# 설정
MIN_MARKET_CAP = 1000  # 최소 시가총액 (억원)
LOOKBACK_DAYS = 250    # 52주 = 약 250 거래일 (기존 사이트와 동일)
ATH_WINDOW = 8         # 최근 8 거래일 내 신고가
VOLUME_RATIO_MIN = 1.5 # 돌파일 거래량 >= 20일 평균 * 1.5
MA150_PERIOD = 150     # 150일 이동평균
MA150_SLOPE_DAYS = 20  # 150MA 우상향 판단 기간

# ML 피처
FEATURES = [
    'breakout_pct', 'volume_surge', 'close_strength',
    'base_length', 'volatility_contraction', 'volume_dry_up',
    'ma200_slope', 'pct_above_52w_low',
    'rs_vs_market', 'market_return',
    'ma20_deviation', 'liquidity',
    'days_since_ath', 'atr_ratio'
]


def fetch_naver_per_pbr(code):
    """네이버 금융에서 PER, PBR 가져오기 (단일 종목용)"""
    try:
        url = f'https://finance.naver.com/item/main.naver?code={code}'
        resp = requests.get(url, headers=HEADERS, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')

        per = None
        pbr = None

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
    """
    네이버 금융에서 PER, PBR 병렬 배치 가져오기

    성능 최적화: 종목별 개별 요청(N+1 문제)을 병렬 처리로 개선
    - 기존: 100개 종목 × 순차 요청 = ~500초 (종목당 5초 타임아웃)
    - 개선: 100개 종목 / 16 워커 = ~32초 (약 15배 향상)
    """
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


class AIScreener:
    """AI 기반 종목 발굴기"""

    def __init__(self):
        self.regressor = None
        self.classifier = None
        self.scaler = None
        self.meta = None
        self.kospi_data = None
        self.stock_info = {}  # 종목 기본정보 (시가총액)
        self._load_models()

    def _load_models(self):
        """ML 모델 로드"""
        try:
            with open(MODEL_PATH / "regressor.pkl", 'rb') as f:
                self.regressor = pickle.load(f)
            with open(MODEL_PATH / "classifier.pkl", 'rb') as f:
                self.classifier = pickle.load(f)
            with open(MODEL_PATH / "scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            with open(MODEL_PATH / "model_meta.pkl", 'rb') as f:
                self.meta = pickle.load(f)
            print("ML 모델 로드 완료")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise

    def get_stock_list(self):
        """KOSPI/KOSDAQ 종목 리스트 가져오기"""
        print("종목 리스트 로딩...")

        kospi = fdr.StockListing('KOSPI')
        kosdaq = fdr.StockListing('KOSDAQ')

        # 필터링: 시가총액, 일반 주식만
        def filter_stocks(df, market):
            df = df.copy()
            df['Market'] = market

            # 시가총액 필터 (억원)
            if 'Marcap' in df.columns:
                df = df[df['Marcap'] >= MIN_MARKET_CAP * 100000000]
            elif 'MarketCap' in df.columns:
                df = df[df['MarketCap'] >= MIN_MARKET_CAP * 100000000]

            # ETF, 우선주 등 제외
            if 'Name' in df.columns:
                df = df[~df['Name'].str.contains('ETF|KODEX|TIGER|KBSTAR|KOSEF|HANARO|ARIRANG|SOL|PLUS|ACE|파워|레버리지|인버스|스팩|리츠|우$|우B$|우C$', na=False, regex=True)]

            return df

        kospi_filtered = filter_stocks(kospi, 'KOSPI')
        kosdaq_filtered = filter_stocks(kosdaq, 'KOSDAQ')

        stocks = pd.concat([kospi_filtered, kosdaq_filtered], ignore_index=True)

        # 종목 기본정보 저장 (시가총액, PER, PBR) - 벡터화 처리
        # iterrows() 대신 to_dict('records')로 한 번에 변환 (약 10배 빠름)
        code_col = 'Code' if 'Code' in stocks.columns else 'Symbol'
        marcap_col = 'Marcap' if 'Marcap' in stocks.columns else 'MarketCap'

        # 벡터화된 시가총액 계산 (억원 단위)
        marcap_values = stocks[marcap_col].fillna(0).values // 100000000

        # 한 번의 반복으로 딕셔너리 생성 (zip 사용)
        codes = stocks[code_col].values
        pers = stocks['PER'].fillna(0).values if 'PER' in stocks.columns else np.zeros(len(stocks))
        pbrs = stocks['PBR'].fillna(0).values if 'PBR' in stocks.columns else np.zeros(len(stocks))

        self.stock_info = {
            code: {'marcap': int(marcap), 'per': per, 'pbr': pbr}
            for code, marcap, per, pbr in zip(codes, marcap_values, pers, pbrs)
        }

        print(f"  총 {len(stocks)}개 종목")

        return stocks

    def get_market_data(self):
        """KOSPI 지수 데이터 (시장 수익률 계산용)"""
        if self.kospi_data is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=400)
            self.kospi_data = fdr.DataReader('KS11', start_date, end_date)
        return self.kospi_data

    def analyze_stock(self, code, name, market):
        """단일 종목 분석 - 기존 사이트 조건과 동일"""
        try:
            # 주가 데이터 로드
            end_date = datetime.now()
            start_date = end_date - timedelta(days=500)  # 충분한 데이터 확보
            df = fdr.DataReader(code, start_date, end_date)

            # 최소 데이터 길이: 52주(250) + 돌파확인(8) + 여유(2) = 260
            required_days = LOOKBACK_DAYS + ATH_WINDOW + 2
            if df is None or len(df) < required_days:
                return None

            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            total_len = len(df)

            current_close = close.iloc[-1]
            prev_close = close.iloc[-2] if len(close) >= 2 else current_close

            # --- 추세 필터: 150일선 위 + 우상향 ---
            if total_len < MA150_PERIOD:
                return None

            ma150_now = close.iloc[-MA150_PERIOD:].mean()

            # 150MA 우상향 체크 (데이터가 충분한 경우만)
            if total_len >= MA150_PERIOD + MA150_SLOPE_DAYS:
                ma150_20ago = close.iloc[-(MA150_PERIOD + MA150_SLOPE_DAYS):-MA150_SLOPE_DAYS].mean()
            else:
                # 데이터 부족시 현재 MA의 절반 기간으로 비교
                half_period = MA150_PERIOD // 2
                ma150_20ago = close.iloc[-MA150_PERIOD:-half_period].mean() if total_len > MA150_PERIOD else ma150_now * 0.99

            # 현재가 > 150일선
            if current_close <= ma150_now:
                return None

            # 150일선 우상향
            if ma150_now <= ma150_20ago:
                return None

            # --- 52주 신고가 계산 (High 기준) ---
            # 돌파 확인 시작점 (8거래일 전)
            base_idx = total_len - 1 - ATH_WINDOW - 1
            if base_idx < LOOKBACK_DAYS:
                return None

            high_52w = high.iloc[base_idx - LOOKBACK_DAYS:base_idx].max()
            low_52w = low.iloc[base_idx - LOOKBACK_DAYS:base_idx].min()

            if pd.isna(high_52w) or high_52w <= 0:
                return None

            # --- 돌파일 찾기: 종가 > 52주 고가 ---
            breakout_idx = None
            breakout_close = None
            breakout_volume = None

            for i in range(base_idx, total_len):
                if close.iloc[i] > high_52w:
                    breakout_idx = i
                    breakout_close = close.iloc[i]
                    breakout_volume = volume.iloc[i]
                    break

            if breakout_idx is None:
                return None

            # --- 거래량 확인: 돌파일 거래량 >= 20일 평균 × 1.5 ---
            if breakout_idx < 20:
                return None

            avg_vol_20 = volume.iloc[breakout_idx - 20:breakout_idx].mean()
            if avg_vol_20 <= 0 or breakout_volume < avg_vol_20 * VOLUME_RATIO_MIN:
                return None

            # --- 현재가가 52주 고가 위에 있어야 함 ---
            if current_close <= high_52w:
                return None

            # 돌파 후 경과일
            days_since_breakout = total_len - 1 - breakout_idx

            # 돌파 종가 대비 현재가
            vs_breakout_close = (current_close / breakout_close - 1) * 100 if breakout_close else 0

            # 등락률 (전일 대비)
            change_pct = (current_close / prev_close - 1) * 100

            # 거래량 비율 (돌파일 기준 - 기존 사이트와 동일)
            volume_ratio = round(breakout_volume / avg_vol_20, 1) if avg_vol_20 > 0 else 0

            # 52주 고가 대비 상승률
            above_high_percent = (current_close / high_52w - 1) * 100

            # 피처 계산 (ML용) - 돌파일 기준으로 계산
            features = self._calculate_features(df, code, breakout_idx)
            if features is None:
                return None

            # ML 예측
            X = np.array([[features.get(f, 0) for f in FEATURES]])
            X_scaled = self.scaler.transform(X)

            predicted_gain = float(self.regressor.predict(X_scaled)[0])
            success_proba = float(self.classifier.predict_proba(X_scaled)[0][1])

            # 추천 결정
            if success_proba >= 0.6 and predicted_gain >= 10:
                recommendation = 'BUY'
            elif success_proba >= 0.5 or predicted_gain >= 5:
                recommendation = 'HOLD'
            else:
                recommendation = 'PASS'

            # 종목 기본정보
            stock_info = self.stock_info.get(code, {})

            # PER/PBR은 나중에 배치로 가져옴 (N+1 문제 방지)
            per, pbr = None, None

            # 돌파일 날짜
            breakout_date = df.index[breakout_idx]
            # "2월2일" 형식으로 변환
            breakout_date_str = f"{breakout_date.month}월{breakout_date.day}일"

            return {
                'code': code,
                'name': name,
                'market': market,
                'close': int(current_close),
                'change_pct': round(change_pct, 2),
                'marcap': stock_info.get('marcap', 0),
                'per': round(per, 1) if per else None,
                'pbr': round(pbr, 2) if pbr else None,
                'high_52w': int(high_52w),
                'low_52w': int(low_52w),
                'breakout_pct': round(above_high_percent, 2),  # 52주 고가 대비 상승률
                'vs_breakout_close': round(vs_breakout_close, 2),
                'volume_surge': volume_ratio,  # 돌파일 거래량 비율 (기존 사이트와 동일)
                'days_since_breakout': days_since_breakout,
                'breakout_date': breakout_date_str,  # "2월2일" 형식
                'breakout_date_full': breakout_date.strftime('%Y-%m-%d'),  # 전체 날짜 (정렬용)
                'predicted_gain': round(predicted_gain, 1),
                'success_probability': round(success_proba * 100, 1),
                'recommendation': recommendation,
                'features': {k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()}
            }

        except Exception as e:
            return None

    def _calculate_features(self, df, code, breakout_idx):
        """
        ML 피처 계산 - 돌파일 기준으로 계산

        Args:
            df: 주가 데이터
            code: 종목코드
            breakout_idx: 돌파일 인덱스 (iloc 기준)
        """
        try:
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']

            # 돌파일 기준 데이터 (모델 학습 시점과 동일)
            breakout_close = close.iloc[breakout_idx]

            # 돌파일까지의 데이터만 사용
            close_to_breakout = close.iloc[:breakout_idx + 1]
            high_to_breakout = high.iloc[:breakout_idx + 1]
            low_to_breakout = low.iloc[:breakout_idx + 1]
            volume_to_breakout = volume.iloc[:breakout_idx + 1]

            # 52주 고저 (돌파일 기준)
            lookback_start = max(0, breakout_idx - LOOKBACK_DAYS)
            high_52w = high.iloc[lookback_start:breakout_idx].max()  # 돌파 전까지의 52주 고가
            low_52w = low.iloc[lookback_start:breakout_idx].min()

            # 돌파 강도 (돌파일 종가 기준)
            breakout_pct = (breakout_close / high_52w - 1) * 100 if high_52w > 0 else 0

            # 거래량 급증 (돌파일 기준)
            if breakout_idx >= 20:
                vol_20d_avg = volume.iloc[breakout_idx - 20:breakout_idx].mean()
                vol_breakout = volume.iloc[breakout_idx]
                volume_surge = vol_breakout / vol_20d_avg if vol_20d_avg > 0 else 1
            else:
                volume_surge = 1

            # 종가 강도 (돌파일 캔들 내 위치)
            breakout_high = high.iloc[breakout_idx]
            breakout_low = low.iloc[breakout_idx]
            if breakout_high != breakout_low:
                close_strength = (breakout_close - breakout_low) / (breakout_high - breakout_low)
            else:
                close_strength = 0.5

            # 베이스 기간 (돌파 전 60일 횡보 기간)
            if breakout_idx >= 60:
                pct_change = close.iloc[breakout_idx - 60:breakout_idx].pct_change().abs()
                base_length = (pct_change < 0.03).sum()
            else:
                base_length = 0

            # 변동성 수축 (돌파일 기준)
            if breakout_idx >= 60:
                recent_close = close.iloc[breakout_idx - 20:breakout_idx]
                past_close = close.iloc[breakout_idx - 60:breakout_idx - 20]
                recent_vol = recent_close.std() / recent_close.mean() if recent_close.mean() > 0 else 0
                past_vol = past_close.std() / past_close.mean() if past_close.mean() > 0 else 0
                volatility_contraction = recent_vol / past_vol if past_vol > 0 else 1
            else:
                volatility_contraction = 1

            # 거래량 고갈 (돌파 전)
            if breakout_idx >= 40:
                vol_recent = volume.iloc[breakout_idx - 10:breakout_idx].mean()
                vol_past = volume.iloc[breakout_idx - 40:breakout_idx - 10].mean()
                volume_dry_up = vol_recent / vol_past if vol_past > 0 else 1
            else:
                volume_dry_up = 1

            # 200일선 기울기 (돌파일 기준)
            if breakout_idx >= 220:
                ma200_now = close.iloc[breakout_idx - 200:breakout_idx].mean()
                ma200_20d_ago = close.iloc[breakout_idx - 220:breakout_idx - 20].mean()
                ma200_slope = (ma200_now / ma200_20d_ago - 1) * 100 if ma200_20d_ago > 0 else 0
            else:
                ma200_slope = 0

            # 52주 저점 대비 상승률 (돌파일 기준)
            pct_above_52w_low = (breakout_close / low_52w - 1) * 100 if low_52w > 0 else 0

            # 시장 대비 상대강도 (돌파일 기준)
            kospi = self.get_market_data()
            breakout_date = df.index[breakout_idx]

            # KOSPI 데이터에서 돌파일에 해당하는 인덱스 찾기
            try:
                kospi_breakout_idx = kospi.index.get_indexer([breakout_date], method='nearest')[0]
                if kospi_breakout_idx >= 20:
                    market_return = (kospi['Close'].iloc[kospi_breakout_idx] / kospi['Close'].iloc[kospi_breakout_idx - 20] - 1) * 100
                    if breakout_idx >= 20:
                        stock_return = (breakout_close / close.iloc[breakout_idx - 20] - 1) * 100
                        rs_vs_market = stock_return - market_return
                    else:
                        rs_vs_market = 0
                else:
                    market_return = 0
                    rs_vs_market = 0
            except:
                market_return = 0
                rs_vs_market = 0

            # 20일선 대비 이격도 (돌파일 기준)
            if breakout_idx >= 20:
                ma20 = close.iloc[breakout_idx - 20:breakout_idx].mean()
                ma20_deviation = (breakout_close / ma20 - 1) * 100 if ma20 > 0 else 0
            else:
                ma20_deviation = 0

            # 유동성 (돌파일 전 20일 평균 거래대금)
            if breakout_idx >= 20:
                liquidity = (close.iloc[breakout_idx - 20:breakout_idx] * volume.iloc[breakout_idx - 20:breakout_idx]).mean()
            else:
                liquidity = 0

            # ATH 이후 경과일 (돌파일 기준)
            ath_start = max(0, breakout_idx - LOOKBACK_DAYS)
            ath_idx = high.iloc[ath_start:breakout_idx].idxmax()
            days_since_ath = (breakout_date - ath_idx).days

            # ATR 비율 (돌파일 기준) - numpy로 최적화
            # pd.concat 반복 호출 대신 numpy 배열 연산 사용 (약 5배 빠름)
            if breakout_idx >= 14:
                h = high.iloc[breakout_idx - 14:breakout_idx].values
                l = low.iloc[breakout_idx - 14:breakout_idx].values
                c_prev = close.iloc[breakout_idx - 15:breakout_idx - 1].values

                # True Range 계산: max(H-L, |H-C_prev|, |L-C_prev|)
                tr1 = h - l
                tr2 = np.abs(h - c_prev)
                tr3 = np.abs(l - c_prev)
                tr = np.maximum(np.maximum(tr1, tr2), tr3)

                atr_14 = tr.mean()
                atr_ratio = atr_14 / breakout_close * 100 if breakout_close > 0 else 0
            else:
                atr_ratio = 0

            return {
                'breakout_pct': breakout_pct,
                'volume_surge': volume_surge,
                'close_strength': close_strength,
                'base_length': base_length,
                'volatility_contraction': volatility_contraction,
                'volume_dry_up': volume_dry_up,
                'ma200_slope': ma200_slope,
                'pct_above_52w_low': pct_above_52w_low,
                'rs_vs_market': rs_vs_market,
                'market_return': market_return,
                'ma20_deviation': ma20_deviation,
                'liquidity': liquidity,
                'days_since_ath': days_since_ath,
                'atr_ratio': atr_ratio
            }

        except Exception as e:
            return None

    def run(self, max_workers=8):
        """스크리너 실행"""
        print("=" * 60)
        print("AI 종목 발굴 - 52주 신고가 돌파 + ML 예측")
        print("=" * 60)

        # 종목 리스트
        stocks = self.get_stock_list()

        # 병렬 분석
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

        # PER/PBR 병렬 배치 가져오기 (N+1 문제 해결)
        if results:
            print(f"\nPER/PBR 데이터 로딩 중... ({len(results)}개 종목)")
            codes = [r['code'] for r in results]
            per_pbr_data = fetch_naver_per_pbr_batch(codes, max_workers=16)

            for result in results:
                code = result['code']
                per, pbr = per_pbr_data.get(code, (None, None))
                result['per'] = round(per, 1) if per else None
                result['pbr'] = round(pbr, 2) if pbr else None

        # 정렬 (돌파 후 최신순 - days_since_breakout 오름차순)
        results.sort(key=lambda x: (x['days_since_breakout'], -x['success_probability']))

        print(f"\n발굴된 종목: {len(results)}개")

        return results

    def save_results(self, results):
        """결과 저장"""
        # numpy 타입을 Python 기본 타입으로 변환
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
            'model_info': {
                'trained_at': self.meta['trained_at'],
                'samples': int(self.meta['samples']),
                'regressor_r2': round(float(self.meta['regressor_performance']['r2']), 4),
                'classifier_precision': round(float(self.meta['classifier_performance']['precision']), 4)
            },
            'count': len(results_converted),
            'stocks': results_converted
        }

        output_file = DATA_PATH / "ai_screener_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"결과 저장: {output_file}")
        return output_file


def main():
    screener = AIScreener()
    results = screener.run()
    screener.save_results(results)

    # 상위 10개 출력
    print("\n" + "=" * 60)
    print("상위 10개 종목")
    print("=" * 60)
    print(f"{'종목명':<12} {'현재가':>10} {'돌파강도':>8} {'예상수익':>8} {'성공확률':>8} {'추천':>6}")
    print("-" * 60)

    for stock in results[:10]:
        print(f"{stock['name']:<12} {stock['close']:>10,} {stock['breakout_pct']:>7.1f}% {stock['predicted_gain']:>7.1f}% {stock['success_probability']:>7.1f}% {stock['recommendation']:>6}")

    return results


if __name__ == "__main__":
    main()
