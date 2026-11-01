"""Trading bot configuration – Quantitative Machine Learning & Risk Management Hub."""

from __future__ import annotations
import os

# ── Mode ──────────────────────────────────────────────────────────────
# "OPTIMIZE" | "TRAIN_OPT" | "TRAIN" | "PAPER" | "LIVE"
MODE = "OPTIMIZE"

# ── Symbols & Exchange ───────────────────────────────────────────────
SYMBOLS = [
    'BTC/USDT',   # Bitcoin
    'ETH/USDT',   # Ethereum
    'SOL/USDT',   # Solana
    'BNB/USDT',   # Binance Coin
    'XRP/USDT',   # Ripple
    'ADA/USDT',   # Cardano
    'AVAX/USDT',  # Avalanche
    'LINK/USDT',  # Chainlink
    'NEAR/USDT',  # Near Protocol
    'SUI/USDT',   # Sui
    'TON/USDT',   # Toncoin
    'UNI/USDT',   # Uniswap
    'POL/USDT',   # Polygon
    'ATOM/USDT',  # Cosmos
    'XLM/USDT',   # Stellar
    'LTC/USDT',   # Litecoin
    'DOGE/USDT',  # Dogecoin
    'SHIB/USDT',  # Shiba Inu
    'ARB/USDT',   # Arbitrum
    'APT/USDT'    # Aptos
]

SYMBOL = SYMBOLS[0]
TIMEFRAME = "4h"          # 1m, 5m, 15m, 1h, 4h, 1d …
DATA_SOURCE = "ccxt"      # "ccxt" | "yfinance"
EXCHANGE_ID = "binance"   # ccxt exchange id

# ── API Keys (Live mod için gereklidir) ───────────────────────────────
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# ── Machine Learning (Sniper Confidence & Paths) ──────────────────────
PROBABILITY_THRESHOLD = 0.50  # Keskin Nişancı Eşiği: Tahmin olasılığı >= %50 değilse pas geç (0)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
MODEL_OPT_PATH = os.path.join(MODEL_DIR, "rf_model_opt.joblib")

# ── Triple-Barrier Method (Marcos López de Prado) ─────────────────────
USE_TRIPLE_BARRIER = True
TB_VOLATILITY_WINDOW = 20     # Geçmiş 20 mumluk getiri volatilitesi
TB_PT_FACTOR = 2.0            # Kâr Al Bariyeri: 2.0 x Volatilite
TB_SL_FACTOR = 1.5            # Zarar Kes Bariyeri: 1.5 x Volatilite
TB_VERTICAL_BARRIER = 8       # Zaman Bariyeri: 8 mum (4h için 32 saat)

# Yedek Sabit Eşikler (Triple Barrier inaktifken)
LOOKAHEAD_PERIODS = 4
TARGET_PROFIT_PCT = 2.0
STOP_LOSS_PCT = 1.5

# ── Optuna Bayesian Optimization Hiperparametre Arama ─────────────────
OPTUNA_TRIALS = 35            # Bayesian arama deneme sayısı
OPTUNA_CV_SPLITS = 5          # TimeSeriesSplit fold sayısı
OPTUNA_METRIC = "f1_macro"    # Optimizasyon metriği ("f1_macro" | "precision_buy")

# Varsayılan XGBoost Parametreleri (Optuna öncesi / TRAIN_OPT için)
OPT_MODEL_TYPE = "XGBOOST"
XGB_N_ESTIMATORS = 350
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.03
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_GAMMA = 0.5
XGB_MIN_CHILD_WEIGHT = 3

# Standart Random Forest Hiperparametreleri
RF_N_ESTIMATORS = 250
RF_MAX_DEPTH = 6
RF_MIN_SAMPLES_SPLIT = 30
RF_MIN_SAMPLES_LEAF = 15
RF_CLASS_WEIGHT = "balanced"
RF_RANDOM_STATE = 42

# Veri Bölme (Zaman Serisi Ayrımı)
TRAIN_TEST_SPLIT_RATIO = 0.8  # %80 Train, %20 Out-of-Sample Test

# ── Sanal Cüzdan & Güven Odaklı Dinamik Pozisyon (Kelly Sizing) ────────
PORTFOLIO_STATE_PATH = "portfolio_state.json"
PORTFOLIO_START_BALANCE = 10_000.0            # Başlangıç kasası ($10,000 USDT)

USE_DYNAMIC_SIZING = True                     # Güven bazlı dinamik pozisyonlama
DYNAMIC_SIZE_TIER_1 = 5.0                     # %50 - %70 güven: Kasanın %5'i
DYNAMIC_SIZE_TIER_2 = 10.0                    # %70 - %80 güven: Kasanın %10'u (Güçlü Sinyal)
DYNAMIC_SIZE_TIER_3 = 15.0                    # %80+ güven: Kasanın %15'i (Kesin Fırsat)
POSITION_SIZE_PCT = 10.0                      # Dinamik boyutlama kapalıyken varsayılan %

MAX_OPEN_POSITIONS = 3                        # Aynı anda maksimum açık Spot pozisyonu
MAX_PORTFOLIO_RISK_PCT = 25.0                 # Toplam portföy riski tavanı (%)

# ── [v3.0 HEDGE FUND QUANT EXPANSION] ────────────────────────────────
# 1. Alternatif Veri (Binance Futures Vadeli Veri Entegrasyonu)
USE_FUTURES_DATA = True                       # Açık Pozisyonlar (OI) ve Fonlama Oranı (FR)
FUTURES_TIMEOUT_MS = 15000

# 2. Piyasa Rejimi Tespiti (HMM / GMM Market Regime)
USE_MARKET_REGIME = True                      # HMM/GMM dinamik piyasa rejimi tespiti
REGIME_NORMAL_THRESHOLD = 0.50                # Rejim 0: Düşük Volatilite (Ölü Piyasa) -> %50
REGIME_CHOP_THRESHOLD = 0.65                  # Rejim 1: Yüksek Volatilite (Testere) -> %65 (Defansif)
REGIME_TREND_THRESHOLD = 0.48                 # Rejim 2: Güçlü Trend (Boğa/Ayı) -> %48 (Agresif)

# 3. Portföy Korelasyon Kalkanı (Correlation Shield)
USE_CORRELATION_SHIELD = True                 # Risk yığılmasını önleyen Pearson kalkanı
MAX_CORRELATION_THRESHOLD = 0.75              # Açık pozisyonlarla maksimum izin verilen korelasyon
CORRELATION_WINDOW = 30                       # Korelasyon hesaplama penceresi (30 mum)

# 4. Sürekli Öğrenme (Auto-Retraining & Walk-Forward Validation)
USE_AUTO_RETRAIN = True                       # Canlı çalışırken periyodik otomatik yeniden eğitim
RETRAIN_INTERVAL_HOURS = 168                  # 7 gün (Haftalık)
RETRAIN_DAY_OF_WEEK = 6                       # 6 = Pazar gecesi
RETRAIN_HOUR_UTC = 0                          # 00:00 UTC
RETRAIN_ROLLING_WEEKS = 52                    # 1 yıllık kayan pencere (Rolling Window)

# ── Take Profit / Stop Loss & Risk Yönetimi ──────────────────────────
USE_ATR_STOPS = True
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5       # SL = Entry - (ATR * 1.5)
ATR_TP_MULTIPLIER = 1.8       # TP = Entry + (ATR * 1.8)

# Breakeven (Kâr Koruma) Mekanizması
USE_BREAKEVEN_STOP = True
BREAKEVEN_TRIGGER_ATR = 1.5   # +1.5x ATR kâra ulaşınca SL = Entry Price
BREAKEVEN_TRIGGER_PCT = 1.5

# ── Maliyetler & Kayma Payı (Slippage) ──────────────────────────────
COMMISSION_PCT = 0.075        # Borsa işlem komisyonu (%)
SLIPPAGE_PCT = 0.05           # Baz kayma payı (%)
DYNAMIC_SLIPPAGE = True       # Volatiliteye duyarlı kayma payı
DYNAMIC_SLIPPAGE_MIN = 0.05
DYNAMIC_SLIPPAGE_MAX = 0.15

# ── Veri Depolama & Loglama ──────────────────────────────────────────
DB_PATH = "market_data.db"
DATA_DIR = "data"
LOG_FILE = "bot_activity.log"
LOG_LEVEL = "INFO"

# Canlı / Paper Polling Döngüsü
POLL_INTERVAL_SECONDS = 10     # Döngü sorgu aralığı (saniye)

# ── Geçmiş Veri Başlangıç Tarihleri ─────────────────────────────────
START_DATE = "2017-08-17"
EXCHANGE_START_MS: dict[str, int] = {
    "BTC/USDT": 1502928000000,
    "ETH/USDT": 1502928000000,
    "SOL/USDT": 1586476800000,
    "BNB/USDT": 1509408000000,
    "XRP/USDT": 1514764800000,
    "ADA/USDT": 1514764800000,
    "AVAX/USDT": 1598486400000,
    "LINK/USDT": 1526428800000,
    "NEAR/USDT": 1600387200000,
    "SUI/USDT": 1682553600000,
    "TON/USDT": 1715040000000,
    "UNI/USDT": 1600214400000,
    "POL/USDT": 1567123200000,
    "ATOM/USDT": 1555459200000,
    "XLM/USDT": 1514764800000,
    "LTC/USDT": 1508889600000,
    "DOGE/USDT": 1573516800000,
    "SHIB/USDT": 1619740800000,
    "ARB/USDT": 1679616000000,
    "APT/USDT": 1665792000000,
    "DEFAULT":  1502928000000,
}
