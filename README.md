import streamlit as st
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json

# تنظیمات صفحه
st.set_page_config(
    page_title="🚀 سیستم فارکس کامل",
    page_icon="💰",
    layout="wide"
)

# Initialize session states
if 'mt5_connected' not in st.session_state:
    st.session_state.mt5_connected = False
if 'market_data' not in st.session_state:
    st.session_state.market_data = {}
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD']
if 'smart_money_data' not in st.session_state:
    st.session_state.smart_money_data = {}
if 'trading_signals' not in st.session_state:
    st.session_state.trading_signals = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    
    .signal-bullish {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    
    .signal-bearish {
        background: linear-gradient(90deg, #cb2d3e 0%, #ef473a 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    
    .signal-neutral {
        background: linear-gradient(90deg, #757f9a 0%, #d7dde8 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🚀 سیستم فارکس کامل و پیشرفته</h1>
    <p>💰 Smart Money + AI Analysis + Real-time Trading + Advanced Charts</p>
</div>
""", unsafe_allow_html=True)

# Core Functions
def connect_to_mt5_quick():
    """اتصال سریع به MT5"""
    try:
        if not mt5.initialize():
            return False, "خطا در راه‌اندازی MT5"
        
        account_info = mt5.account_info()
        if account_info is None:
            return False, "اطلاعات حساب دریافت نشد"
        
        return True, {
            'login': account_info.login,
            'balance': account_info.balance,
            'equity': account_info.equity,
            'server': account_info.server,
            'company': account_info.company,
            'currency': account_info.currency,
            'leverage': account_info.leverage,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free
        }
    except Exception as e:
        return False, f"خطا در اتصال: {str(e)}"

def get_symbol_categories():
    """دریافت و دسته‌بندی نمادها"""
    try:
        symbols = mt5.symbols_get()
        if not symbols:
            return {}
        
        categories = {
            'forex_major': [],
            'forex_minor': [],
            'gold_metals': [],
            'commodities': [],
            'indices': [],
            'crypto': [],
            'other': []
        }
        
        for symbol in symbols:
            name = symbol.name.upper()
            desc = getattr(symbol, 'description', '').upper()
            
            # Major Forex
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
            if any(pair in name.replace('/', '').replace('.', '') for pair in major_pairs):
                categories['forex_major'].append(symbol.name)
            
            # Gold & Metals
            elif any(metal in name for metal in ['XAUUSD', 'GOLD', 'XAGUSD', 'SILVER', 'PLATINUM', 'PALLADIUM']):
                categories['gold_metals'].append(symbol.name)
            
            # Minor Forex
            elif any(curr in name for curr in ['EUR', 'GBP', 'USD', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']) and len(name) <= 10:
                categories['forex_minor'].append(symbol.name)
            
            # Commodities
            elif any(comm in name or comm in desc for comm in ['OIL', 'CRUDE', 'BRENT', 'WTI', 'COPPER', 'ZINC']):
                categories['commodities'].append(symbol.name)
            
            # Indices
            elif any(index in name or index in desc for index in ['US30', 'NAS100', 'SPX500', 'DAX', 'FTSE', 'NIKKEI']):
                categories['indices'].append(symbol.name)
            
            # Crypto
            elif any(crypto in name or crypto in desc for crypto in ['BTC', 'ETH', 'BITCOIN', 'ETHEREUM']):
                categories['crypto'].append(symbol.name)
            
            else:
                categories['other'].append(symbol.name)
        
        return categories
    
    except Exception as e:
        st.error(f"خطا در دریافت نمادها: {str(e)}")
        return {
            'forex_major': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
            'gold_metals': ['XAUUSD'],
            'forex_minor': [],
            'commodities': [],
            'indices': [],
            'crypto': [],
            'other': []
        }

def get_market_data(symbol, timeframe='H1', count=500):
    """دریافت داده بازار"""
    try:
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df, True
        
        return None, False
    
    except Exception as e:
        st.error(f"خطا در دریافت داده {symbol}: {str(e)}")
        return None, False

def calculate_all_indicators(df):
    """محاسبه همه اندیکاتورها"""
    if df is None or len(df) < 50:
        return df
    
    try:
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Moving Averages
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['SMA_200'] = df['close'].rolling(200).mean()
        df['EMA_20'] = df['close'].ewm(span=20).mean()
        df['EMA_50'] = df['close'].ewm(span=50).mean()
        df['EMA_200'] = df['close'].ewm(span=200).mean()
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['BB_Upper'] = sma_20 + (std_20 * 2)
        df['BB_Lower'] = sma_20 - (std_20 * 2)
        df['BB_Middle'] = sma_20
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['Stoch_K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift())
        low_close_prev = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Volume indicators
        df['Volume_MA'] = df['tick_volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['tick_volume'] / df['Volume_MA']
        df['Volume_Spike'] = df['Volume_Ratio'] > 1.5
        
        # Price action
        df['Body'] = abs(df['close'] - df['open'])
        df['Upper_Shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['Lower_Shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['Range'] = df['high'] - df['low']
        df['Body_Ratio'] = df['Body'] / df['Range']
        
        # Trend identification
        df['Bullish'] = df['close'] > df['open']
        df['Bearish'] = df['close'] < df['open']
        df['Doji'] = df['Body_Ratio'] < 0.1
        df['Hammer'] = (df['Lower_Shadow'] > df['Body'] * 2) & (df['Upper_Shadow'] < df['Body'] * 0.1)
        df['Shooting_Star'] = (df['Upper_Shadow'] > df['Body'] * 2) & (df['Lower_Shadow'] < df['Body'] * 0.1)
        
        return df
    
    except Exception as e:
        st.error(f"خطا در محاسبه اندیکاتورها: {str(e)}")
        return df

def advanced_market_analysis(df, symbol):
    """تحلیل پیشرفته بازار"""
    try:
        if df is None or len(df) < 50:
            return {
                'overall_signal': 'خطا',
                'confidence': 0,
                'trend': 'نامشخص',
                'strength': 0,
                'risk_level': 'نامشخص'
            }
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        current_macd = df['MACD'].iloc[-1]
        current_macd_signal = df['MACD_Signal'].iloc[-1]
        current_stoch_k = df['Stoch_K'].iloc[-1]
        current_bb_position = df['BB_Position'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        
        # Moving averages
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        sma_200 = df['SMA_200'].iloc[-1] if len(df) >= 200 else current_price
        ema_20 = df['EMA_20'].iloc[-1]
        ema_50 = df['EMA_50'].iloc[-1]
        
        # Trend analysis
        trend_signals = []
        trend_strength = 0
        
        # MA Trend
        if current_price > ema_20 > ema_50 > sma_200:
            trend_signals.append('strong_uptrend')
            trend_strength += 4
        elif current_price > ema_20 > ema_50:
            trend_signals.append('uptrend')
            trend_strength += 2
        elif current_price < ema_20 < ema_50 < sma_200:
            trend_signals.append('strong_downtrend')
            trend_strength -= 4
        elif current_price < ema_20 < ema_50:
            trend_signals.append('downtrend')
            trend_strength -= 2
        else:
            trend_signals.append('sideways')
        
        # Price momentum
        price_change_5 = ((current_price / df['close'].iloc[-6]) - 1) * 100
        price_change_20 = ((current_price / df['close'].iloc[-21]) - 1) * 100
        
        if price_change_20 > 3:
            trend_signals.append('strong_bullish_momentum')
            trend_strength += 3
        elif price_change_20 > 1:
            trend_signals.append('bullish_momentum')
            trend_strength += 1
        elif price_change_20 < -3:
            trend_signals.append('strong_bearish_momentum')
            trend_strength -= 3
        elif price_change_20 < -1:
            trend_signals.append('bearish_momentum')
            trend_strength -= 1
        
        # Oscillator analysis
        oscillator_signals = []
        osc_strength = 0
        
        # RSI
        if current_rsi < 20:
            oscillator_signals.append('rsi_extremely_oversold')
            osc_strength += 3
        elif current_rsi < 30:
            oscillator_signals.append('rsi_oversold')
            osc_strength += 2
        elif current_rsi > 80:
            oscillator_signals.append('rsi_extremely_overbought')
            osc_strength -= 3
        elif current_rsi > 70:
            oscillator_signals.append('rsi_overbought')
            osc_strength -= 2
        
        # MACD
        if current_macd > current_macd_signal and current_macd > 0:
            oscillator_signals.append('macd_strong_bullish')
            osc_strength += 2
        elif current_macd > current_macd_signal:
            oscillator_signals.append('macd_bullish')
            osc_strength += 1
        elif current_macd < current_macd_signal and current_macd < 0:
            oscillator_signals.append('macd_strong_bearish')
            osc_strength -= 2
        elif current_macd < current_macd_signal:
            oscillator_signals.append('macd_bearish')
            osc_strength -= 1
        
        # Stochastic
        if current_stoch_k < 20:
            oscillator_signals.append('stoch_oversold')
            osc_strength += 1
        elif current_stoch_k > 80:
            oscillator_signals.append('stoch_overbought')
            osc_strength -= 1
        
        # Volume analysis
        volume_signals = []
        vol_strength = 0
        
        recent_volume_avg = df['Volume_Ratio'].tail(5).mean()
        if recent_volume_avg > 1.5:
            volume_signals.append('high_volume_activity')
            vol_strength += 2
        elif recent_volume_avg > 1.2:
            volume_signals.append('above_average_volume')
            vol_strength += 1
        elif recent_volume_avg < 0.8:
            volume_signals.append('low_volume')
            vol_strength -= 1
        
        # Support/Resistance analysis
        sr_signals = []
        sr_strength = 0
        
        # Bollinger Bands position
        if current_bb_position > 0.9:
            sr_signals.append('near_bb_upper')
            sr_strength -= 1
        elif current_bb_position < 0.1:
            sr_signals.append('near_bb_lower')
            sr_strength += 1
        
        # Recent highs/lows
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        if abs(current_price - recent_high) / current_price < 0.005:
            sr_signals.append('near_recent_high')
            sr_strength -= 1
        elif abs(current_price - recent_low) / current_price < 0.005:
            sr_signals.append('near_recent_low')
            sr_strength += 1
        
        # Volatility analysis
        volatility = current_atr / current_price * 100
        
        if volatility > 2:
            risk_level = 'بالا'
            vol_multiplier = 0.8
        elif volatility > 1:
            risk_level = 'متوسط'
            vol_multiplier = 1.0
        else:
            risk_level = 'پایین'
            vol_multiplier = 1.2
        
        # Final calculation
        total_strength = (trend_strength + osc_strength + vol_strength + sr_strength) * vol_multiplier
        
        # Signal determination
        if total_strength >= 6:
            overall_signal = "خرید فوق‌العاده قوی"
            action = "STRONG_BUY"
            confidence = min(90 + (total_strength - 6) * 2, 98)
        elif total_strength >= 4:
            overall_signal = "خرید قوی"
            action = "BUY"
            confidence = min(80 + (total_strength - 4) * 5, 90)
        elif total_strength >= 2:
            overall_signal = "خرید"
            action = "WEAK_BUY"
            confidence = min(70 + (total_strength - 2) * 5, 80)
        elif total_strength <= -6:
            overall_signal = "فروش فوق‌العاده قوی"
            action = "STRONG_SELL"
            confidence = min(90 + abs(total_strength + 6) * 2, 98)
        elif total_strength <= -4:
            overall_signal = "فروش قوی"
            action = "SELL"
            confidence = min(80 + abs(total_strength + 4) * 5, 90)
        elif total_strength <= -2:
            overall_signal = "فروش"
            action = "WEAK_SELL"
            confidence = min(70 + abs(total_strength + 2) * 5, 80)
        else:
            overall_signal = "خنثی"
            action = "HOLD"
            confidence = 50 + abs(total_strength) * 3
        
        # Trend classification
        if trend_strength >= 4:
            trend_class = "صعودی فوق‌العاده"
        elif trend_strength >= 2:
            trend_class = "صعودی"
        elif trend_strength <= -4:
            trend_class = "نزولی فوق‌العاده"
        elif trend_strength <= -2:
            trend_class = "نزولی"
        else:
            trend_class = "رنج"
        
        return {
            'overall_signal': overall_signal,
            'action': action,
            'confidence': round(confidence, 1),
            'trend': trend_class,
            'strength': round(total_strength, 1),
            'risk_level': risk_level,
            'volatility': round(volatility, 3),
            'price_change_5': round(price_change_5, 2),
            'price_change_20': round(price_change_20, 2),
            'indicators': {
                'RSI': round(current_rsi, 1),
                'MACD': round(current_macd, 5),
                'MACD_Signal': round(current_macd_signal, 5),
                'Stoch_K': round(current_stoch_k, 1),
                'BB_Position': round(current_bb_position * 100, 1),
                'ATR': round(current_atr, 5),
                'Volume_Ratio': round(recent_volume_avg, 2)
            },
            'levels': {
                'current_price': current_price,
                'resistance': recent_high,
                'support': recent_low,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'ema_20': ema_20,
                'ema_50': ema_50
            },
            'signals': {
                'trend': trend_signals,
                'oscillator': oscillator_signals,
                'volume': volume_signals,
                'support_resistance': sr_signals
            },
            'analysis_time': datetime.now()
        }
    
    except Exception as e:
        st.error(f"خطا در تحلیل: {str(e)}")
        return {
            'overall_signal': 'خطا در تحلیل',
            'action': 'HOLD',
            'confidence': 0,
            'trend': 'نامشخص',
            'strength': 0,
            'risk_level': 'نامشخص',
            'error': str(e)
        }

def smart_money_analysis(df, symbol):
    """تحلیل Smart Money"""
    try:
        if df is None or len(df) < 50:
            return {
                'smart_signal': 'خطا',
                'institution_activity': 'نامشخص',
                'order_flow': 'نامشخص'
            }
        
        # Volume analysis for institutional activity
        volume_ma_20 = df['tick_volume'].rolling(20).mean()
        recent_volume = df['tick_volume'].tail(10).mean()
        volume_spike_ratio = recent_volume / volume_ma_20.iloc[-1]
        
        # Large body candles with high volume
        large_body_high_vol = df[(df['Body_Ratio'] > 0.7) & (df['Volume_Ratio'] > 1.5)].tail(10)
        
        # Price rejection analysis
        recent_hammers = df[df['Hammer']].tail(5)
        recent_shooting_stars = df[df['Shooting_Star']].tail(5)
        
        # Order flow analysis
        bullish_volume = df[df['Bullish'] & (df['Volume_Ratio'] > 1.2)]['tick_volume'].tail(10).sum()
        bearish_volume = df[df['Bearish'] & (df['Volume_Ratio'] > 1.2)]['tick_volume'].tail(10).sum()
        
        smart_signals = []
        institution_strength = 0
        
        # High volume spikes
        if volume_spike_ratio > 2:
            smart_signals.append("فعالیت نهادی قوی")
            institution_strength += 3
        elif volume_spike_ratio > 1.5:
            smart_signals.append("فعالیت نهادی متوسط")
            institution_strength += 2
        
        # Large body analysis
        if len(large_body_high_vol) >= 3:
            if large_body_high_vol['Bullish'].sum() > large_body_high_vol['Bearish'].sum():
                smart_signals.append("Smart Money خرید")
                institution_strength += 2
            else:
                smart_signals.append("Smart Money فروش")
                institution_strength -= 2
        
        # Rejection analysis
        if len(recent_hammers) >= 2:
            smart_signals.append("رد فروش توسط خریداران قوی")
            institution_strength += 2
        
        if len(recent_shooting_stars) >= 2:
            smart_signals.append("رد خرید توسط فروشندگان قوی")
            institution_strength -= 2
        
        # Order flow
        if bullish_volume > bearish_volume * 1.5:
            order_flow = "خریداران کنترل دارند"
            institution_strength += 1
        elif bearish_volume > bullish_volume * 1.5:
            order_flow = "فروشندگان کنترل دارند"
            institution_strength -= 1
        else:
            order_flow = "تعادل قدرت"
        
        # Final smart money signal
        if institution_strength >= 4:
            smart_signal = "Smart Money خرید قوی"
        elif institution_strength >= 2:
            smart_signal = "Smart Money خرید"
        elif institution_strength <= -4:
            smart_signal = "Smart Money فروش قوی"
        elif institution_strength <= -2:
            smart_signal = "Smart Money فروش"
        else:
            smart_signal = "فعالیت نهادی خنثی"
        
        # Institution activity level
        if abs(institution_strength) >= 4:
            activity_level = "بسیار بالا"
        elif abs(institution_strength) >= 2:
            activity_level = "بالا"
        elif abs(institution_strength) >= 1:
            activity_level = "متوسط"
        else:
            activity_level = "پایین"
        
        return {
            'smart_signal': smart_signal,
            'institution_activity': activity_level,
            'order_flow': order_flow,
            'volume_spike_ratio': round(volume_spike_ratio, 2),
            'institution_strength': institution_strength,
            'signals': smart_signals,
            'bullish_volume': bullish_volume,
            'bearish_volume': bearish_volume
        }
    
    except Exception as e:
        return {
            'smart_signal': 'خطا در تحلیل Smart Money',
            'institution_activity': 'نامشخص',
            'order_flow': 'نامشخص',
            'error': str(e)
        }

def get_live_price(symbol):
    """دریافت قیمت زنده"""
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                spread = (tick.ask - tick.bid) / symbol_info.point
                
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'spread': spread,
                    'time': datetime.fromtimestamp(tick.time),
                    'point': symbol_info.point,
                    'digits': symbol_info.digits
                }
        return None
    except Exception as e:
        return None

def scan_market(symbols_list, timeframe='H1', min_confidence=70):
    """اسکن بازار"""
    results = []
    
    for symbol in symbols_list:
        try:
            data, success = get_market_data(symbol, timeframe, 200)
            
            if success:
                data = calculate_all_indicators(data)
                analysis = advanced_market_analysis(data, symbol)
                smart_analysis = smart_money_analysis(data, symbol)
                live_price = get_live_price(symbol)
                
                if analysis['confidence'] >= min_confidence:
                    results.append({
                        'symbol': symbol,
                        'analysis': analysis,
                        'smart_money': smart_analysis,
                        'live_price': live_price,
                        'scan_time': datetime.now()
                    })
        
        except Exception as e:
            continue
    
    return results

# Sidebar
with st.sidebar:
    st.header("🔗 اتصال MT5")
    
    if not st.session_state.mt5_connected:
        st.warning("🔴 MT5 قطع")
        
        if st.button("🔗 اتصال سریع", use_container_width=True, type="primary"):
            with st.spinner("در حال اتصال..."):
                success, result = connect_to_mt5_quick()
                if success:
                    st.session_state.mt5_connected = True
                    st.session_state.account_info = result
                    st.success("✅ اتصال برقرار شد!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ {result}")
        
        st.markdown("---")
        st.markdown("### 🔧 اتصال دستی")
        
        with st.form("manual_connection"):
            login = st.text_input("👤 Login:")
            password = st.text_input("🔐 Password:", type="password")
            server = st.text_input("🌐 Server:")
            
            if st.form_submit_button("🔗 اتصال"):
                if login and password and server:
                    try:
                        mt5.shutdown()
                        if mt5.initialize():
                            authorized = mt5.login(int(login), password=password, server=server)
                            if authorized:
                                account_info = mt5.account_info()
                                if account_info:
                                    st.session_state.mt5_connected = True
                                    st.session_state.account_info = {
                                        'login': account_info.login,
                                        'balance': account_info.balance,
                                        'equity': account_info.equity,
                                        'server': account_info.server,
                                        'company': account_info.company
                                    }
                                    st.success("✅ اتصال برقرار شد!")
                                    st.rerun()
                            else:
                                st.error("❌ اطلاعات نادرست!")
                    except:
                        st.error("❌ خطا در اتصال!")
    
    else:
        st.success("🟢 MT5 متصل")
        
        account = st.session_state.account_info
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 اطلاعات حساب</h4>
            <p><strong>👤 Login:</strong> {account['login']}</p>
            <p><strong>🏢 شرکت:</strong> {account['company']}</p>
            <p><strong>💰 موجودی:</strong> ${account['balance']:,.2f}</p>
            <p><strong>📊 Equity:</strong> ${account['equity']:,.2f}</p>
            <p><strong>🎯 اهرم:</strong> 1:{account.get('leverage', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("❌ قطع اتصال", use_container_width=True):
            mt5.shutdown()
            st.session_state.mt5_connected = False
            st.session_state.account_info = {}
            st.rerun()
    
    st.markdown("---")
    
    # Watchlist Management
    st.header("👁️ واچ لیست")
    
    # Add new symbol
    new_symbol = st.text_input("➕ اضافه کردن نماد:")
    if st.button("اضافه") and new_symbol:
        if new_symbol.upper() not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_symbol.upper())
            st.success(f"✅ {new_symbol} اضافه شد!")
            st.rerun()
        else:
            st.warning("⚠️ این نماد قبلاً موجود است!")
    
    # Display watchlist
    st.markdown("### 📋 نمادهای واچ لیست")
    for i, symbol in enumerate(st.session_state.watchlist):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"💱 {symbol}")
        with col2:
            if st.button("🗑️", key=f"remove_{i}", help="حذف"):
                st.session_state.watchlist.remove(symbol)
                st.rerun()
    
    st.markdown("---")
    
    # Quick Stats
    if st.session_state.mt5_connected:
        st.header("📊 آمار سریع")
        
        current_time = datetime.now()
        st.write(f"🕒 {current_time.strftime('%H:%M:%S')}")
        st.write(f"📅 {current_time.strftime('%Y-%m-%d')}")
        
        if 'market_data' in st.session_state and st.session_state.market_data:
            total_symbols = len(st.session_state.market_data)
            st.metric("📊 نمادهای تحلیل شده", total_symbols)

# Main Content
if st.session_state.mt5_connected:
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 داشبورد اصلی",
        "📊 تحلیل پیشرفته", 
        "🔍 اسکنر بازار",
        "📈 نمودارهای زنده",
        "💰 Smart Money",
        "🚨 هشدارها و تنظیمات",
        "🤖 ربات معاملاتی"
    ])
    
    with tab1:
        st.header("🏠 داشبورد اصلی")
        
        # Market overview cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 بروزرسانی واچ لیست", type="primary", use_container_width=True):
                with st.spinner("در حال بروزرسانی..."):
                    scan_results = scan_market(st.session_state.watchlist[:8], 'H1', 60)
                    st.session_state.market_data = {r['symbol']: r for r in scan_results}
                    st.success(f"✅ {len(scan_results)} نماد بروزرسانی شد!")
                    st.rerun()
        
        with col2:
            auto_refresh = st.checkbox("🔄 بروزرسانی خودکار", key="auto_refresh_main")
        
        with col3:
            refresh_interval = st.selectbox("⏰ فاصله زمانی:", [30, 60, 120, 300], index=1)
        
        # Market summary
        if 'market_data' in st.session_state and st.session_state.market_data:
            st.subheader("📊 خلاصه بازار")
            
            summary_data = []
            for symbol, data in st.session_state.market_data.items():
                analysis = data['analysis']
                live_price = data.get('live_price')
                smart_data = data.get('smart_money', {})
                
                # Signal color coding
                signal = analysis['overall_signal']
                if 'قوی' in signal and 'خرید' in signal:
                    signal_color = "🟢"
                    signal_class = "signal-bullish"
                elif 'قوی' in signal and 'فروش' in signal:
                    signal_color = "🔴"
                    signal_class = "signal-bearish"
                elif 'خرید' in signal:
                    signal_color = "🟡"
                    signal_class = "signal-bullish"
                elif 'فروش' in signal:
                    signal_color = "🟠"
                    signal_class = "signal-bearish"
                else:
                    signal_color = "⚪"
                    signal_class = "signal-neutral"
                
                summary_data.append({
                    'نماد': symbol,
                    'قیمت': f"{live_price['bid']:.5f}" if live_price else "N/A",
                    'سیگنال': f"{signal_color} {signal}",
                    'اعتماد': f"{analysis['confidence']}%",
                    'روند': analysis['trend'],
                    'RSI': f"{analysis['indicators']['RSI']:.1f}",
                    'Smart Money': smart_data.get('smart_signal', 'N/A'),
                    'تغییر 20': f"{analysis.get('price_change_20', 0):.2f}%",
                    'ریسک': analysis.get('risk_level', 'N/A')
                })
            
            # Create DataFrame and display
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, height=400)
            
            # Market statistics
            st.subheader("📈 آمار بازار")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                buy_signals = len([d for d in st.session_state.market_data.values() 
                                 if 'خرید' in d['analysis']['overall_signal']])
                st.metric("🟢 سیگنال خرید", buy_signals)
            
            with col2:
                sell_signals = len([d for d in st.session_state.market_data.values() 
                                  if 'فروش' in d['analysis']['overall_signal']])
                st.metric("🔴 سیگنال فروش", sell_signals)
            
            with col3:
                neutral_signals = len([d for d in st.session_state.market_data.values() 
                                     if 'خنثی' in d['analysis']['overall_signal']])
                st.metric("⚪ خنثی", neutral_signals)
            
            with col4:
                avg_confidence = np.mean([d['analysis']['confidence'] 
                                        for d in st.session_state.market_data.values()])
                st.metric("🎯 میانگین اعتماد", f"{avg_confidence:.1f}%")
            
            with col5:
                high_confidence = len([d for d in st.session_state.market_data.values() 
                                     if d['analysis']['confidence'] > 80])
                st.metric("⭐ اعتماد بالا", high_confidence)
            
            # Top opportunities
            st.subheader("🚀 بهترین فرصت‌ها")
            
            # Sort by confidence
            sorted_opportunities = sorted(
                st.session_state.market_data.items(),
                key=lambda x: x[1]['analysis']['confidence'],
                reverse=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 برترین خرید")
                buy_opportunities = [(symbol, data) for symbol, data in sorted_opportunities 
                                   if 'خرید' in data['analysis']['overall_signal']]
                
                for symbol, data in buy_opportunities[:3]:
                    analysis = data['analysis']
                    st.markdown(f"""
                    <div class="signal-bullish">
                        <strong>{symbol}</strong>: {analysis['overall_signal']} 
                        (اعتماد: {analysis['confidence']}%)
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(f"💡 {analysis['trend']} - RSI: {analysis['indicators']['RSI']:.1f}")
                    st.markdown("---")
            
            with col2:
                st.markdown("### 📉 برترین فروش")
                sell_opportunities = [(symbol, data) for symbol, data in sorted_opportunities 
                                    if 'فروش' in data['analysis']['overall_signal']]
                
                for symbol, data in sell_opportunities[:3]:
                    analysis = data['analysis']
                    st.markdown(f"""
                    <div class="signal-bearish">
                        <strong>{symbol}</strong>: {analysis['overall_signal']} 
                        (اعتماد: {analysis['confidence']}%)
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(f"💡 {analysis['trend']} - RSI: {analysis['indicators']['RSI']:.1f}")
                    st.markdown("---")
        
        else:
            st.info("💡 برای مشاهده تحلیل بازار، دکمه 'بروزرسانی واچ لیست' را فشار دهید")
        
        # Auto refresh functionality
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
    
    with tab2:
        st.header("📊 تحلیل پیشرفته")
        
        # Symbol selection
        symbols = get_symbol_categories()
        all_symbols = []
        for category_symbols in symbols.values():
            all_symbols.extend(category_symbols)
        
        if all_symbols:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                selected_symbol = st.selectbox("💱 انتخاب نماد:", all_symbols[:20], 
                                             key="analysis_symbol")
            
            with col2:
                timeframe = st.selectbox("⏰ تایم فریم:", 
                                       ['M15', 'H1', 'H4', 'D1', 'W1'], 
                                       index=1, key="analysis_timeframe")
            
            with col3:
                periods = st.slider("📊 تعداد کندل:", 100, 1000, 300, 
                                  key="analysis_periods")
            
            if st.button("🧠 تحلیل جامع", type="primary", use_container_width=True):
                with st.spinner("در حال تحلیل..."):
                    data, success = get_market_data(selected_symbol, timeframe, periods)
                    
                    if success and data is not None:
                        # Calculate indicators
                        data = calculate_all_indicators(data)
                        
                        # Perform analysis
                        analysis = advanced_market_analysis(data, selected_symbol)
                        smart_analysis = smart_money_analysis(data, selected_symbol)
                        live_price = get_live_price(selected_symbol)
                        
                        # Display results
                        st.subheader("🎯 نتایج تحلیل")
                        
                        # Main metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            confidence_color = ("success" if analysis['confidence'] > 80 
                                              else "warning" if analysis['confidence'] > 60 
                                              else "error")
                            
                            signal = analysis['overall_signal']
                            if 'قوی' in signal and 'خرید' in signal:
                                st.markdown(f"""
                                <div class="signal-bullish">
                                    <h4>🎯 {signal}</h4>
                                    <p>اعتماد: {analysis['confidence']}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif 'قوی' in signal and 'فروش' in signal:
                                st.markdown(f"""
                                <div class="signal-bearish">
                                    <h4>🎯 {signal}</h4>
                                    <p>اعتماد: {analysis['confidence']}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="signal-neutral">
                                    <h4>🎯 {signal}</h4>
                                    <p>اعتماد: {analysis['confidence']}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("📈 روند", analysis['trend'])
                            st.metric("💪 قدرت", analysis['strength'])
                        
                        with col3:
                            st.metric("⚠️ ریسک", analysis['risk_level'])
                            st.metric("📊 نوسان", f"{analysis['volatility']:.3f}%")
                        
                        with col4:
                            if live_price:
                                st.metric("💰 قیمت فعلی", f"{live_price['bid']:.5f}")
                                st.metric("📊 Spread", f"{live_price['spread']:.1f} pips")
                        
                        # Technical indicators
                        st.subheader("📊 اندیکاتورهای تکنیکال")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            rsi = analysis['indicators']['RSI']
                            rsi_color = ("🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡")
                            st.metric("📊 RSI", f"{rsi:.1f} {rsi_color}")
                        
                        with col2:
                            macd = analysis['indicators']['MACD']
                            macd_signal = analysis['indicators']['MACD_Signal']
                            macd_color = ("🟢" if macd > macd_signal else "🔴")
                            st.metric("📈 MACD", f"{macd:.5f} {macd_color}")
                        
                        with col3:
                            stoch = analysis['indicators']['Stoch_K']
                            stoch_color = ("🔴" if stoch > 80 else "🟢" if stoch < 20 else "🟡")
                            st.metric("📊 Stochastic", f"{stoch:.1f} {stoch_color}")
                        
                        with col4:
                            bb_pos = analysis['indicators']['BB_Position']
                            bb_color = ("🔴" if bb_pos > 80 else "🟢" if bb_pos < 20 else "🟡")
                            st.metric("📊 BB Position", f"{bb_pos:.1f}% {bb_color}")
                        
                        # Smart Money Analysis
                        st.subheader("💰 تحلیل Smart Money")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🧠 سیگنال Smart Money", smart_analysis['smart_signal'])
                        
                        with col2:
                            st.metric("🏢 فعالیت نهادی", smart_analysis['institution_activity'])
                        
                        with col3:
                            st.metric("💹 جریان سفارش", smart_analysis['order_flow'])
                        
                        # Price levels
                        st.subheader("📊 سطوح مهم قیمت")
                        
                        levels = analysis['levels']
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("🔺 مقاومت", f"{levels['resistance']:.5f}")
                        
                        with col2:
                            st.metric("🔻 حمایت", f"{levels['support']:.5f}")
                        
                        with col3:
                            st.metric("📊 SMA 20", f"{levels['sma_20']:.5f}")
                        
                        with col4:
                            st.metric("📊 EMA 50", f"{levels['ema_50']:.5f}")
                        
                        # Advanced Chart
                        st.subheader("📈 نمودار پیشرفته")
                        
                        # Chart controls
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            show_ma = st.checkbox("📊 میانگین متحرک", value=True)
                        
                        with col2:
                            show_bb = st.checkbox("📊 Bollinger Bands", value=True)
                        
                        with col3:
                            show_volume = st.checkbox("📊 حجم", value=True)
                        
                        # Main price chart
                        fig = go.Figure()
                        
                        # Show last 100 candles
                        display_data = data.tail(100)
                        
                        # Candlestick
                        fig.add_trace(go.Candlestick(
                            x=display_data.index,
                            open=display_data['open'],
                            high=display_data['high'],
                            low=display_data['low'],
                            close=display_data['close'],
                            name=selected_symbol,
                            increasing_line_color='#26a69a',
                            decreasing_line_color='#ef5350'
                        ))
                        
                        # Moving averages
                        if show_ma:
                            fig.add_trace(go.Scatter(
                                x=display_data.index,
                                y=display_data['SMA_20'],
                                mode='lines',
                                name='SMA 20',
                                line=dict(color='blue', width=1)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=display_data.index,
                                y=display_data['SMA_50'],
                                mode='lines',
                                name='SMA 50',
                                line=dict(color='red', width=1)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=display_data.index,
                                y=display_data['EMA_20'],
                                mode='lines',
                                name='EMA 20',
                                line=dict(color='orange', width=1, dash='dash')
                            ))
                        
                        # Bollinger Bands
                        if show_bb:
                            fig.add_trace(go.Scatter(
                                x=display_data.index,
                                y=display_data['BB_Upper'],
                                mode='lines',
                                name='BB Upper',
                                line=dict(color='purple', width=1),
                                opacity=0.7
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=display_data.index,
                                y=display_data['BB_Lower'],
                                mode='lines',
                                name='BB Lower',
                                line=dict(color='purple', width=1),
                                fill='tonexty',
                                fillcolor='rgba(128, 0, 128, 0.1)',
                                opacity=0.7
                            ))
                        
                        # Support and Resistance lines
                        fig.add_hline(
                            y=levels['resistance'],
                            line_dash="dash",
                            line_color="red",
                            annotation_text="مقاومت"
                        )
                        
                        fig.add_hline(
                            y=levels['support'],
                            line_dash="dash",
                            line_color="green",
                            annotation_text="حمایت"
                        )
                        
                        fig.update_layout(
                            title=f"{selected_symbol} - {timeframe} | {analysis['overall_signal']} (اعتماد: {analysis['confidence']}%)",
                            height=600,
                            xaxis_title="زمان",
                            yaxis_title="قیمت",
                            template="plotly_white",
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Volume chart
                        if show_volume:
                            st.subheader("📊 تحلیل حجم")
                            
                            fig_volume = go.Figure()
                            
                            # Volume bars with color coding
                            colors = ['green' if row['close'] > row['open'] else 'red' 
                                     for _, row in display_data.iterrows()]
                            
                            fig_volume.add_trace(go.Bar(
                                x=display_data.index,
                                y=display_data['tick_volume'],
                                marker_color=colors,
                                name='Volume',
                                opacity=0.7
                            ))
                            
                            # Volume moving average
                            fig_volume.add_trace(go.Scatter(
                                x=display_data.index,
                                y=display_data['Volume_MA'],
                                mode='lines',
                                name='Volume MA',
                                line=dict(color='orange', width=2)
                            ))
                            
                            # High volume spikes
                            high_volume_data = display_data[display_data['Volume_Spike']]
                            if not high_volume_data.empty:
                                fig_volume.add_trace(go.Scatter(
                                    x=high_volume_data.index,
                                    y=high_volume_data['tick_volume'],
                                    mode='markers',
                                    marker=dict(color='yellow', size=8, symbol='star'),
                                    name='High Volume'
                                ))
                            
                            fig_volume.update_layout(
                                title="Volume Analysis with Smart Money Activity",
                                height=300,
                                xaxis_title="زمان",
                                yaxis_title="حجم"
                            )
                            
                            st.plotly_chart(fig_volume, use_container_width=True)
                        
                        # RSI and other oscillators
                        st.subheader("📊 اسیلاتورها")
                        
                        fig_rsi = go.Figure()
                        
                        # RSI
                        fig_rsi.add_trace(go.Scatter(
                            x=display_data.index,
                            y=display_data['RSI'],
                            mode='lines',
                            name='RSI',
                            line=dict(color='purple', width=2)
                        ))
                        
                        # RSI levels
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                                         annotation_text="اشباع خرید")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                                         annotation_text="اشباع فروش")
                        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", 
                                         annotation_text="خط میانی")
                        
                        fig_rsi.update_layout(
                            title="RSI (Relative Strength Index)",
                            height=300,
                            yaxis=dict(range=[0, 100])
                        )
                        
                        st.plotly_chart(fig_rsi, use_container_width=True)
                        
                        # MACD
                        fig_macd = go.Figure()
                        
                        fig_macd.add_trace(go.Scatter(
                            x=display_data.index,
                            y=display_data['MACD'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_macd.add_trace(go.Scatter(
                            x=display_data.index,
                            y=display_data['MACD_Signal'],
                            mode='lines',
                            name='Signal',
                            line=dict(color='red', width=2)
                        ))
                        
                        fig_macd.add_trace(go.Bar(
                            x=display_data.index,
                            y=display_data['MACD_Histogram'],
                            name='Histogram',
                            opacity=0.7
                        ))
                        
                        fig_macd.add_hline(y=0, line_dash="dot", line_color="gray")
                        
                        fig_macd.update_layout(
                            title="MACD (Moving Average Convergence Divergence)",
                            height=300
                        )
                        
                        st.plotly_chart(fig_macd, use_container_width=True)
                        
                        # Trading recommendation
                        st.subheader("💡 توصیه معاملاتی")
                        
                        recommendation_text = f"""
                        **تحلیل کلی:** {analysis['overall_signal']} با اعتماد {analysis['confidence']}%
                        
                        **نکات مهم:**
                        - روند: {analysis['trend']} (قدرت: {analysis['strength']})
                        - ریسک: {analysis['risk_level']}
                        - Smart Money: {smart_analysis['smart_signal']}
                        
                        **سطوح کلیدی:**
                        - مقاومت: {levels['resistance']:.5f}
                        - حمایت: {levels['support']:.5f}
                        
                        **اندیکاتورها:**
                        - RSI: {analysis['indicators']['RSI']:.1f} ({'اشباع خرید' if analysis['indicators']['RSI'] > 70 else 'اشباع فروش' if analysis['indicators']['RSI'] < 30 else 'عادی'})
                        - MACD: {'مثبت' if analysis['indicators']['MACD'] > 0 else 'منفی'}
                        """
                        
                        if analysis['confidence'] > 80:
                            st.success(recommendation_text)
                        elif analysis['confidence'] > 60:
                            st.info(recommendation_text)
                        else:
                            st.warning(recommendation_text)
                        
                        # Save analysis to history
                        analysis_record = {
                            'symbol': selected_symbol,
                            'timeframe': timeframe,
                            'analysis': analysis,
                            'smart_money': smart_analysis,
                            'timestamp': datetime.now()
                        }
                        
                        st.session_state.analysis_history.append(analysis_record)
                        if len(st.session_state.analysis_history) > 50:  # Keep last 50
                            st.session_state.analysis_history = st.session_state.analysis_history[-50:]
                    
                    else:
                        st.error("❌ خطا در دریافت داده یا داده کافی موجود نیست!")
        
        else:
            st.warning("⚠️ نمادی یافت نشد! لطفاً ابتدا به MT5 متصل شوید.")
    
    with tab3:
        st.header("🔍 اسکنر پیشرفته بازار")
        
        # Scanner settings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scan_timeframe = st.selectbox("⏰ تایم فریم اسکن:", 
                                        ['M15', 'H1', 'H4', 'D1'], 
                                        index=1, key="scan_timeframe")
        
        with col2:
            min_confidence = st.slider("🎯 حداقل اعتماد:", 50, 95, 70, 
                                     key="scan_confidence")
        
        with col3:
            signal_filter = st.selectbox("🔍 فیلتر سیگنال:", 
                                       ['همه', 'فقط خرید', 'فقط فروش', 'سیگنال‌های قوی', 'Smart Money'], 
                                       key="scan_filter")
        
        # Symbol categories for scanning
        st.subheader("📊 انتخاب دسته‌بندی نمادها")
        
        symbols_categories = get_symbol_categories()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scan_major_forex = st.checkbox("💱 فارکس اصلی", value=True)
            scan_gold = st.checkbox("✨ طلا و فلزات", value=True)
        
        with col2:
            scan_minor_forex = st.checkbox("💱 فارکس فرعی", value=False)
            scan_commodities = st.checkbox("🛢️ کامودیتی", value=False)
        
        with col3:
            scan_indices = st.checkbox("📈 شاخص‌ها", value=False)
            scan_watchlist = st.checkbox("👁️ واچ لیست", value=True)
        
        # Build scan list
        scan_symbols = []
        
        if scan_watchlist and st.session_state.watchlist:
            scan_symbols.extend(st.session_state.watchlist)
        
        if scan_major_forex and symbols_categories.get('forex_major'):
            scan_symbols.extend(symbols_categories['forex_major'][:10])
        
        if scan_gold and symbols_categories.get('gold_metals'):
            scan_symbols.extend(symbols_categories['gold_metals'][:5])
        
        if scan_minor_forex and symbols_categories.get('forex_minor'):
            scan_symbols.extend(symbols_categories['forex_minor'][:10])
        
        if scan_commodities and symbols_categories.get('commodities'):
            scan_symbols.extend(symbols_categories['commodities'][:5])
        
        if scan_indices and symbols_categories.get('indices'):
            scan_symbols.extend(symbols_categories['indices'][:5])
        
        # Remove duplicates
        scan_symbols = list(set(scan_symbols))
        
        st.write(f"📊 تعداد نمادهای انتخاب شده: {len(scan_symbols)}")
        
        if scan_symbols and st.button("🚀 شروع اسکن", type="primary", use_container_width=True):
            with st.spinner("در حال اسکن بازار..."):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                scan_results = []
                
                for i, symbol in enumerate(scan_symbols):
                    status_text.text(f"🔍 اسکن {symbol}... ({i+1}/{len(scan_symbols)})")
                    progress_bar.progress((i + 1) / len(scan_symbols))
                    
                    try:
                        data, success = get_market_data(symbol, scan_timeframe, 200)
                        
                        if success and data is not None:
                            data = calculate_all_indicators(data)
                            analysis = advanced_market_analysis(data, symbol)
                            smart_analysis = smart_money_analysis(data, symbol)
                            
                            # Apply filters
                            include_result = False
                            
                            if signal_filter == 'همه':
                                include_result = analysis['confidence'] >= min_confidence
                            elif signal_filter == 'فقط خرید':
                                include_result = ('خرید' in analysis['overall_signal'] and 
                                                analysis['confidence'] >= min_confidence)
                            elif signal_filter == 'فقط فروش':
                                include_result = ('فروش' in analysis['overall_signal'] and 
                                                analysis['confidence'] >= min_confidence)
                            elif signal_filter == 'سیگنال‌های قوی':
                                include_result = ('قوی' in analysis['overall_signal'] and 
                                                analysis['confidence'] >= min_confidence)
                            elif signal_filter == 'Smart Money':
                                include_result = ('Smart Money' in smart_analysis['smart_signal'] and 
                                                analysis['confidence'] >= min_confidence)
                            
                            if include_result:
                                live_price = get_live_price(symbol)
                                scan_results.append({
                                    'symbol': symbol,
                                    'analysis': analysis,
                                    'smart_money': smart_analysis,
                                    'live_price': live_price,
                                    'scan_time': datetime.now()
                                })
                    
                    except Exception as e:
                        continue
                
                progress_bar.progress(1.0)
                status_text.text("✅ اسکن کامل شد!")
                
                # Display results
                if scan_results:
                    st.success(f"✅ {len(scan_results)} فرصت معاملاتی یافت شد!")
                    
                    # Sort by confidence
                    scan_results.sort(key=lambda x: x['analysis']['confidence'], reverse=True)
                    
                    # Results table
                    st.subheader("📊 نتایج اسکن")
                    
                    scan_data = []
                    for result in scan_results:
                        symbol = result['symbol']
                        analysis = result['analysis']
                        smart_data = result['smart_money']
                        live_price = result.get('live_price')
                        
                        # Signal emoji
                        signal = analysis['overall_signal']
                        if 'قوی' in signal and 'خرید' in signal:
                            signal_emoji = "🟢🚀"
                        elif 'قوی' in signal and 'فروش' in signal:
                            signal_emoji = "🔴⬇️"
                        elif 'خرید' in signal:
                            signal_emoji = "🟡⬆️"
                        elif 'فروش' in signal:
                            signal_emoji = "🟠⬇️"
                        else:
                            signal_emoji = "⚪➡️"
                        
                        scan_data.append({
                            'رتبه': len(scan_data) + 1,
                            'نماد': symbol,
                            'قیمت': f"{live_price['bid']:.5f}" if live_price else "N/A",
                            'سیگنال': f"{signal_emoji} {signal}",
                            'اعتماد': f"{analysis['confidence']}%",
                            'قدرت': analysis['strength'],
                            'روند': analysis['trend'],
                            'ریسک': analysis['risk_level'],
                            'RSI': f"{analysis['indicators']['RSI']:.1f}",
                            'Smart Money': smart_data['smart_signal'],
                            'تغییر': f"{analysis.get('price_change_20', 0):.2f}%",
                            'زمان': result['scan_time'].strftime('%H:%M:%S')
                        })
                    
                    df_scan = pd.DataFrame(scan_data)
                    st.dataframe(df_scan, use_container_width=True, height=400)
                    
                    # Statistics
                    st.subheader("📈 آمار اسکن")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        buy_count = len([r for r in scan_results 
                                       if 'خرید' in r['analysis']['overall_signal']])
                        st.metric("🟢 فرصت خرید", buy_count)
                    
                    with col2:
                        sell_count = len([r for r in scan_results 
                                        if 'فروش' in r['analysis']['overall_signal']])
                        st.metric("🔴 فرصت فروش", sell_count)
                    
                    with col3:
                        strong_signals = len([r for r in scan_results 
                                            if 'قوی' in r['analysis']['overall_signal']])
                        st.metric("⭐ سیگنال قوی", strong_signals)
                    
                    with col4:
                        avg_conf = np.mean([r['analysis']['confidence'] for r in scan_results])
                        st.metric("🎯 میانگین اعتماد", f"{avg_conf:.1f}%")
                    
                    with col5:
                        smart_money_signals = len([r for r in scan_results 
                                                 if 'Smart Money' in r['smart_money']['smart_signal']])
                        st.metric("💰 Smart Money", smart_money_signals)
                    
                    # Top opportunities visualization
                    st.subheader("🎯 برترین فرصت‌ها")
                    
                    if len(scan_results) >= 5:
                        top_5 = scan_results[:5]
                        
                        symbols = [r['symbol'] for r in top_5]
                        confidences = [r['analysis']['confidence'] for r in top_5]
                        
                        fig = go.Figure(data=[
                            go.Bar(x=symbols, y=confidences, 
                                  marker_color=['green' if 'خرید' in r['analysis']['overall_signal'] 
                                              else 'red' if 'فروش' in r['analysis']['overall_signal'] 
                                              else 'gray' for r in top_5])
                        ])
                        
                        fig.update_layout(
                            title="5 فرصت برتر (بر اساس اعتماد)",
                            xaxis_title="نماد",
                            yaxis_title="درصد اعتماد",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed view for top result
                    if scan_results:
                        st.subheader("🔍 جزئیات برترین فرصت")
                        
                        top_result = scan_results[0]
                        symbol = top_result['symbol']
                        analysis = top_result['analysis']
                        smart_data = top_result['smart_money']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**💱 نماد:** {symbol}")
                            st.write(f"**🎯 سیگنال:** {analysis['overall_signal']}")
                            st.write(f"**🤖 اعتماد:** {analysis['confidence']}%")
                            st.write(f"**📈 روند:** {analysis['trend']}")
                            st.write(f"**⚠️ ریسک:** {analysis['risk_level']}")
                        
                        with col2:
                            st.write(f"**💰 Smart Money:** {smart_data['smart_signal']}")
                            st.write(f"**📊 RSI:** {analysis['indicators']['RSI']:.1f}")
                            st.write(f"**📈 MACD:** {analysis['indicators']['MACD']:.5f}")
                            st.write(f"**📊 تغییر 20 دوره:** {analysis.get('price_change_20', 0):.2f}%")
                            st.write(f"**🕒 زمان اسکن:** {top_result['scan_time'].strftime('%H:%M:%S')}")
                        
                        # Quick chart for top symbol
                        if st.button(f"📈 نمودار سریع {symbol}"):
                            with st.spinner("بارگذاری نمودار..."):
                                chart_data, success = get_market_data(symbol, scan_timeframe, 100)
                                
                                if success:
                                    chart_data = calculate_all_indicators(chart_data)
                                    
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Candlestick(
                                        x=chart_data.index,
                                        open=chart_data['open'],
                                        high=chart_data['high'],
                                        low=chart_data['low'],
                                        close=chart_data['close'],
                                        name=symbol
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=chart_data.index,
                                        y=chart_data['SMA_20'],
                                        mode='lines',
                                        name='SMA 20',
                                        line=dict(color='blue', width=1)
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"{symbol} - {analysis['overall_signal']} (اعتماد: {analysis['confidence']}%)",
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("⚠️ هیچ فرصت معاملاتی با معیارهای انتخاب شده یافت نشد!")
                    st.info("💡 سعی کنید معیارهای فیلتر را تغییر دهید یا حداقل اعتماد را کاهش دهید")
    
    with tab4:
        st.header("📈 نمودارهای زنده و مانیتورینگ")
        
        # Live chart settings
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            live_symbol = st.selectbox("💱 نماد:", st.session_state.watchlist, 
                                     key="live_symbol")
        
        with col2:
            live_timeframe = st.selectbox("⏰ تایم فریم:", 
                                        ['M1', 'M5', 'M15', 'H1', 'H4'], 
                                        index=2, key="live_timeframe")
        
        with col3:
            auto_update_charts = st.checkbox("🔄 بروزرسانی خودکار", key="auto_charts")
        
        with col4:
            update_interval = st.selectbox("⏰ فاصله:", [10, 30, 60, 120], index=1)
        
        # Chart controls
        st.subheader("🎛️ تنظیمات نمودار")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            show_ma_live = st.checkbox("📊 میانگین متحرک", value=True, key="live_ma")
        
        with col2:
            show_bb_live = st.checkbox("📊 Bollinger Bands", value=False, key="live_bb")
        
        with col3:
            show_volume_live = st.checkbox("📊 حجم", value=True, key="live_volume")
        
        with col4:
            show_rsi_live = st.checkbox("📊 RSI", value=True, key="live_rsi")
        
        # Live data display
        if st.button("📈 نمایش نمودار زنده") or auto_update_charts:
            with st.spinner("بارگذاری داده زنده..."):
                live_data, success = get_market_data(live_symbol, live_timeframe, 200)
                
                if success and live_data is not None:
                    live_data = calculate_all_indicators(live_data)
                    
                    # Current market status
                    current_price = live_data['close'].iloc[-1]
                    price_change = ((current_price / live_data['close'].iloc[-20]) - 1) * 100
                    current_rsi = live_data['RSI'].iloc[-1]
                    live_price_data = get_live_price(live_symbol)
                    
                    # Market status header
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("💰 قیمت فعلی", f"{current_price:.5f}")
                    
                    with col2:
                        delta_color = "normal" if abs(price_change) < 0.1 else None
                        st.metric("📊 تغییر 20 دوره", f"{price_change:.2f}%", 
                                delta=f"{price_change:.2f}%", delta_color=delta_color)
                    
                    with col3:
                        st.metric("📊 RSI", f"{current_rsi:.1f}")
                    
                    with col4:
                        if live_price_data:
                            st.metric("📊 Spread", f"{live_price_data['spread']:.1f} pips")
                    
                    with col5:
                        current_time = datetime.now()
                        st.metric("🕒 زمان", current_time.strftime("%H:%M:%S"))
                    
                    # Main price chart
                    fig_main = go.Figure()
                    
                    # Show last 100 candles for better performance
                    display_data = live_data.tail(100)
                    
                    # Candlestick chart
                    fig_main.add_trace(go.Candlestick(
                        x=display_data.index,
                        open=display_data['open'],
                        high=display_data['high'],
                        low=display_data['low'],
                        close=display_data['close'],
                        name=live_symbol,
                        increasing_line_color='#26a69a',
                        decreasing_line_color='#ef5350'
                    ))
                    
                    # Moving averages
                    if show_ma_live:
                        fig_main.add_trace(go.Scatter(
                            x=display_data.index,
                            y=display_data['SMA_20'],
                            mode='lines',
                            name='SMA 20',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_main.add_trace(go.Scatter(
                            x=display_data.index,
                            y=display_data['SMA_50'],
                            mode='lines',
                            name='SMA 50',
                            line=dict(color='red', width=2)
                        ))
                    
                    # Bollinger Bands
                    if show_bb_live:
                        fig_main.add_trace(go.Scatter(
                            x=display_data.index,
                            y=display_data['BB_Upper'],
                            mode='lines',
                            name='BB Upper',
                            line=dict(color='purple', width=1),
                            opacity=0.7
                        ))
                        
                        fig_main.add_trace(go.Scatter(
                            x=display_data.index,
                            y=display_data['BB_Lower'],
                            mode='lines',
                            name='BB Lower',
                            line=dict(color='purple', width=1),
                            fill='tonexty',
                            fillcolor='rgba(128, 0, 128, 0.1)',
                            opacity=0.7
                        ))
                    
                    fig_main.update_layout(
                        title=f"{live_symbol} - {live_timeframe} (زنده) | قیمت: {current_price:.5f} | تغییر: {price_change:.2f}%",
                        height=500,
                        xaxis_title="زمان",
                        yaxis_title="قیمت",
                        template="plotly_white",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_main, use_container_width=True)
                    
                    # Sub-charts
                    if show_volume_live or show_rsi_live:
                        col1, col2 = st.columns(2)
                        
                        # Volume chart
                        if show_volume_live:
                            with col1:
                                st.subheader("📊 حجم معاملات")
                                
                                fig_volume = go.Figure()
                                
                                colors = ['green' if row['close'] > row['open'] else 'red' 
                                         for _, row in display_data.iterrows()]
                                
                                fig_volume.add_trace(go.Bar(
                                    x=display_data.index,
                                    y=display_data['tick_volume'],
                                    marker_color=colors,
                                    name='Volume',
                                    opacity=0.7
                                ))
                                
                                fig_volume.add_trace(go.Scatter(
                                    x=display_data.index,
                                    y=display_data['Volume_MA'],
                                    mode='lines',
                                    name='Volume MA',
                                    line=dict(color='orange', width=2)
                                ))
                                
                                fig_volume.update_layout(
                                    title="حجم معاملات",
                                    height=300,
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig_volume, use_container_width=True)
                        
                        # RSI chart
                        if show_rsi_live:
                            with col2:
                                st.subheader("📊 RSI")
                                
                                fig_rsi = go.Figure()
                                
                                fig_rsi.add_trace(go.Scatter(
                                    x=display_data.index,
                                    y=display_data['RSI'],
                                    mode='lines',
                                    name='RSI',
                                    line=dict(color='purple', width=2)
                                ))
                                
                                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                                fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
                                
                                fig_rsi.update_layout(
                                    title=f"RSI: {current_rsi:.1f}",
                                    height=300,
                                    yaxis=dict(range=[0, 100]),
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # Quick analysis
                    st.subheader("🧠 تحلیل سریع")
                    
                    analysis = advanced_market_analysis(live_data, live_symbol)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        signal = analysis['overall_signal']
                        if 'قوی' in signal and 'خرید' in signal:
                            st.success(f"🎯 {signal}")
                        elif 'قوی' in signal and 'فروش' in signal:
                            st.error(f"🎯 {signal}")
                        else:
                            st.info(f"🎯 {signal}")
                        
                        st.write(f"**اعتماد:** {analysis['confidence']}%")
                    
                    with col2:
                        st.write(f"**📈 روند:** {analysis['trend']}")
                        st.write(f"**💪 قدرت:** {analysis['strength']}")
                        st.write(f"**⚠️ ریسک:** {analysis['risk_level']}")
                    
                    with col3:
                        st.write(f"**📊 RSI:** {analysis['indicators']['RSI']:.1f}")
                        st.write(f"**📈 MACD:** {analysis['indicators']['MACD']:.5f}")
                        st.write(f"**📊 BB Position:** {analysis['indicators']['BB_Position']:.1f}%")
                    
                    # Market alerts
                    st.subheader("🚨 هشدارهای لحظه‌ای")
                    
                    alerts = []
                    
                    # RSI alerts
                    if current_rsi > 80:
                        alerts.append("🔴 RSI اشباع خرید شدید (>80)")
                    elif current_rsi > 70:
                        alerts.append("🟠 RSI اشباع خرید (>70)")
                    elif current_rsi < 20:
                        alerts.append("🟢 RSI اشباع فروش شدید (<20)")
                    elif current_rsi < 30:
                        alerts.append("🟡 RSI اشباع فروش (<30)")
                    
                    # Price movement alerts
                    if abs(price_change) > 2:
                        direction = "صعودی" if price_change > 0 else "نزولی"
                        alerts.append(f"⚡ حرکت قوی {direction}: {abs(price_change):.2f}%")
                    
                    # Volume alerts
                    recent_volume_ratio = live_data['Volume_Ratio'].iloc[-1]
                    if recent_volume_ratio > 2:
                        alerts.append("📊 حجم بالای غیرعادی شناسایی شد")
                    
                    # High confidence signal alerts
                    if analysis['confidence'] > 85:
                        alerts.append(f"🎯 سیگنال با اعتماد بالا: {analysis['overall_signal']}")
                    
                    if alerts:
                        for alert in alerts:
                            st.warning(alert)
                    else:
                        st.info("✅ وضعیت عادی - هیچ هشدار خاصی وجود ندارد")
                    
                    # Auto refresh
                    if auto_update_charts:
                        time.sleep(update_interval)
                        st.rerun()
                
                else:
                    st.error("❌ خطا در دریافت داده زنده")
        
        # Multi-symbol monitoring
        st.subheader("👁️ مانیتورینگ چند نماد")
        
        if len(st.session_state.watchlist) > 1:
            if st.button("📊 مانیتورینگ واچ لیست", use_container_width=True):
                with st.spinner("بارگذاری داده همه نمادها..."):
                    monitoring_data = []
                    
                    for symbol in st.session_state.watchlist[:6]:  # حداکثر 6 نماد
                        try:
                            data, success = get_market_data(symbol, 'H1', 50)
                            if success:
                                data = calculate_all_indicators(data)
                                current_price = data['close'].iloc[-1]
                                current_rsi = data['RSI'].iloc[-1]
                                price_change = ((current_price / data['close'].iloc[-10]) - 1) * 100
                                
                                # Status determination
                                if current_rsi > 70:
                                    status = "🔴 اشباع خرید"
                                elif current_rsi < 30:
                                    status = "🟢 اشباع فروش"
                                elif abs(price_change) > 1:
                                    status = "⚡ حرکت قوی"
                                else:
                                    status = "⚪ عادی"
                                
                                monitoring_data.append({
                                    'نماد': symbol,
                                    'قیمت': f"{current_price:.5f}",
                                    'تغییر 10 دوره': f"{price_change:.2f}%",
                                    'RSI': f"{current_rsi:.1f}",
                                    'وضعیت': status,
                                    'زمان': datetime.now().strftime('%H:%M:%S')
                                })
                        except:
                            continue
                    
                    if monitoring_data:
                        df_monitoring = pd.DataFrame(monitoring_data)
                        st.dataframe(df_monitoring, use_container_width=True)
                        
                        # Summary
                        alert_count = len([d for d in monitoring_data if d['وضعیت'] != '⚪ عادی'])
                        st.write(f"🚨 تعداد نمادهای نیاز به توجه: {alert_count}")
                    
                    else:
                        st.warning("⚠️ خطا در دریافت داده برخی نمادها")
    
    with tab5:
        st.header("💰 Smart Money Analysis")
        
        st.markdown("""
        ### 🧠 تحلیل Smart Money چیست؟
        
        Smart Money به سرمایه‌گذاران بزرگ، نهادها، بانک‌ها و صندوق‌های سرمایه‌گذاری اطلاق می‌شود که:
        - حجم بالایی از معاملات انجام می‌دهند
        - اطلاعات بیشتری از بازار دارند  
        - می‌توانند جهت بازار را تغییر دهند
        - معمولاً زودتر از سایرین وارد یا خارج می‌شوند
        """)
        
        # Smart Money analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            sm_symbol = st.selectbox("💱 انتخاب نماد برای تحلیل Smart Money:", 
                                   st.session_state.watchlist, key="sm_symbol")
        
        with col2:
            sm_timeframe = st.selectbox("⏰ تایم فریم:", ['H1', 'H4', 'D1'], 
                                      index=1, key="sm_timeframe")
        
        if st.button("🧠 تحلیل Smart Money", type="primary", use_container_width=True):
            with st.spinner("در حال تحلیل فعالیت Smart Money..."):
                data, success = get_market_data(sm_symbol, sm_timeframe, 500)
                
                if success and data is not None:
                    data = calculate_all_indicators(data)
                    smart_analysis = smart_money_analysis(data, sm_symbol)
                    market_analysis = advanced_market_analysis(data, sm_symbol)
                    
                    # Smart Money signals
                    st.subheader("💰 سیگنال‌های Smart Money")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        signal = smart_analysis['smart_signal']
                        if 'قوی' in signal and 'خرید' in signal:
                            st.success(f"🧠 {signal}")
                        elif 'قوی' in signal and 'فروش' in signal:
                            st.error(f"🧠 {signal}")
                        else:
                            st.info(f"🧠 {signal}")
                    
                    with col2:
                        st.metric("🏢 فعالیت نهادی", smart_analysis['institution_activity'])
                        st.metric("💹 جریان سفارش", smart_analysis['order_flow'])
                    
                    with col3:
                        st.metric("📊 نسبت حجم", f"{smart_analysis['volume_spike_ratio']:.2f}x")
                        st.metric("💪 قدرت نهادی", smart_analysis['institution_strength'])
                    
                    # Smart Money signals detail
                    if smart_analysis.get('signals'):
                        st.subheader("📋 سیگنال‌های تشخیص داده شده")
                        for i, signal in enumerate(smart_analysis['signals'], 1):
                            st.write(f"{i}. {signal}")
                    
                    # Volume analysis chart
                    st.subheader("📊 تحلیل حجم Smart Money")
                    
                    display_data = data.tail(100)
                    
                    fig_smart = go.Figure()
                    
                    # Price
                    fig_smart.add_trace(go.Candlestick(
                        x=display_data.index,
                        open=display_data['open'],
                        high=display_data['high'],
                        low=display_data['low'],
                        close=display_data['close'],
                        name=sm_symbol,
                        yaxis='y2'
                    ))
                    
                    # Volume
                    colors = ['green' if row['close'] > row['open'] else 'red' 
                             for _, row in display_data.iterrows()]
                    
                    fig_smart.add_trace(go.Bar(
                        x=display_data.index,
                        y=display_data['tick_volume'],
                        marker_color=colors,
                        name='Volume',
                        opacity=0.7,
                        yaxis='y'
                    ))
                    
                    # Volume MA
                    fig_smart.add_trace(go.Scatter(
                        x=display_data.index,
                        y=display_data['Volume_MA'],
                        mode='lines',
                        name='Volume MA',
                        line=dict(color='orange', width=2),
                        yaxis='y'
                    ))
                    
                    # High volume spikes
                    high_volume = display_data[display_data['Volume_Spike']]
                    if not high_volume.empty:
                        fig_smart.add_trace(go.Scatter(
                            x=high_volume.index,
                            y=high_volume['tick_volume'],
                            mode='markers',
                            marker=dict(color='yellow', size=10, symbol='star'),
                            name='Smart Money Activity',
                            yaxis='y'
                        ))
                    
                    fig_smart.update_layout(
                        title=f"{sm_symbol} - Smart Money Analysis ({sm_timeframe})",
                        height=600,
                        yaxis=dict(title="Volume", side="left"),
                        yaxis2=dict(title="Price", side="right", overlaying="y"),
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig_smart, use_container_width=True)
                    
                    # Order flow analysis
                    st.subheader("💹 تحلیل جریان سفارش")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🟢 حجم خریداران (صعودی):**")
                        st.write(f"{smart_analysis.get('bullish_volume', 0):,}")
                        
                        st.write("**🔴 حجم فروشندگان (نزولی):**")
                        st.write(f"{smart_analysis.get('bearish_volume', 0):,}")
                    
                    with col2:
                        # Order flow chart
                        bullish_vol = smart_analysis.get('bullish_volume', 0)
                        bearish_vol = smart_analysis.get('bearish_volume', 0)
                        total_vol = bullish_vol + bearish_vol
                        
                        if total_vol > 0:
                            fig_flow = go.Figure(data=[go.Pie(
                                labels=['خریداران', 'فروشندگان'],
                                values=[bullish_vol, bearish_vol],
                                marker_colors=['green', 'red']
                            )])
                            
                            fig_flow.update_layout(
                                title="توزیع جریان سفارش",
                                height=300
                            )
                            
                            st.plotly_chart(fig_flow, use_container_width=True)
                    
                    # Combined analysis
                    st.subheader("🎯 تحلیل ترکیبی")
                    
                    combined_signals = []
                    confidence_factors = []
                    
                    # Market analysis
                    market_signal = market_analysis['overall_signal']
                    market_confidence = market_analysis['confidence']
                    
                    # Smart money analysis
                    smart_signal = smart_analysis['smart_signal']
                    institution_strength = abs(smart_analysis['institution_strength'])
                    
                    combined_signals.append(f"📊 تحلیل تکنیکال: {market_signal} (اعتماد: {market_confidence}%)")
                    combined_signals.append(f"💰 Smart Money: {smart_signal}")
                    
                    # Agreement analysis
                    market_bullish = 'خرید' in market_signal
                    smart_bullish = 'خرید' in smart_signal
                    
                    if market_bullish and smart_bullish:
                        st.success("✅ توافق کامل: هم تحلیل تکنیکال و هم Smart Money سیگنال خرید می‌دهند")
                        final_recommendation = "خرید قوی"
                    elif not market_bullish and not smart_bullish:
                        st.error("❌ توافق کامل: هم تحلیل تکنیکال و هم Smart Money سیگنال فروش می‌دهند")
                        final_recommendation = "فروش قوی"
                    elif market_bullish and not smart_bullish:
                        st.warning("⚠️ اختلاف نظر: تحلیل تکنیکال خرید، Smart Money فروش")
                        final_recommendation = "احتیاط - انتظار"
                    elif not market_bullish and smart_bullish:
                        st.warning("⚠️ اختلاف نظر: تحلیل تکنیکال فروش، Smart Money خرید")
                        final_recommendation = "احتیاط - انتظار"
                    else:
                        st.info("ℹ️ وضعیت خنثی")
                        final_recommendation = "خنثی"
                    
                    st.write(f"**🎯 توصیه نهایی:** {final_recommendation}")
                    
                    # Smart money patterns
                    st.subheader("🔍 الگوهای Smart Money")
                    
                    patterns = []
                    
                    # High volume analysis
                    recent_high_volume = len(display_data[display_data['Volume_Spike']])
                    if recent_high_volume >= 5:
                        patterns.append("📊 فعالیت نهادی بالا در دوره اخیر")
                    
                    # Strong body + high volume
                    strong_candles = display_data[(display_data['Body_Ratio'] > 0.7) & (display_data['Volume_Ratio'] > 1.5)]
                    if len(strong_candles) >= 3:
                        patterns.append("💪 کندل‌های قوی با حجم بالا (ورود Smart Money)")
                    
                    # Rejection patterns
                    hammer_count = len(display_data[display_data['Hammer']])
                    shooting_star_count = len(display_data[display_data['Shooting_Star']])
                    
                    if hammer_count >= 2:
                        patterns.append("🔨 الگوی Hammer - رد فروش توسط خریداران")
                    
                    if shooting_star_count >= 2:
                        patterns.append("⭐ الگوی Shooting Star - رد خرید توسط فروشندگان")
                    
                    if patterns:
                        st.write("**🔍 الگوهای تشخیص داده شده:**")
                        for pattern in patterns:
                            st.write(f"• {pattern}")
                    else:
                        st.write("• هیچ الگوی خاص Smart Money تشخیص داده نشد")
                
                else:
                    st.error("❌ خطا در دریافت داده")
        
        # Smart Money education
        with st.expander("📚 آموزش Smart Money"):
            st.markdown("""
            ### 🎓 مفاهیم کلیدی Smart Money:
            
            **1. 📊 Volume Spike (افزایش ناگهانی حجم):**
            - نشان‌دهنده ورود یا خروج Smart Money
            - معمولاً همراه با تغییر جهت قیمت
            
            **2. 🕯️ Strong Body Candles:**
            - کندل‌های با بدنه قوی + حجم بالا
            - نشان‌دهنده تصمیم‌گیری قاطع نهادها
            
            **3. 🔨 Rejection Patterns:**
            - Hammer: رد فروش توسط خریداران
            - Shooting Star: رد خرید توسط فروشندگان
            
            **4. 💹 Order Flow:**
            - تحلیل نسبت حجم خرید به فروش
            - تشخیص کنترل بازار توسط کدام گروه
            
            **5. ⚡ Institution Activity:**
            - سطح فعالیت نهادها
            - بر اساس حجم و الگوهای قیمت
            """)
    
    with tab6:
        st.header("🚨 هشدارها و تنظیمات پیشرفته")
        
        # Alert management
        st.subheader("🔔 مدیریت هشدارها")
        
        # Add new alert
        with st.expander("➕ افزودن هشدار جدید"):
            col1, col2 = st.columns(2)
            
            with col1:
                alert_symbol = st.selectbox("💱 نماد:", st.session_state.watchlist, 
                                          key="alert_symbol")
                alert_type = st.selectbox("📊 نوع هشدار:", [
                    'قیمت بالاتر از',
                    'قیمت پایین‌تر از', 
                    'RSI بالاتر از',
                    'RSI پایین‌تر از',
                    'سیگنال خرید',
                    'سیگنال فروش',
                    'Smart Money فعالیت'
                ], key="alert_type")
            
            with col2:
                if alert_type in ['قیمت بالاتر از', 'قیمت پایین‌تر از']:
                    alert_value = st.number_input("🎯 قیمت هدف:", value=1.0, step=0.0001, 
                                                format="%.5f", key="alert_price")
                elif alert_type in ['RSI بالاتر از', 'RSI پایین‌تر از']:
                    alert_value = st.slider("🎯 سطح RSI:", 0, 100, 70, key="alert_rsi")
                else:
                    alert_value = 0
                
                alert_active = st.checkbox("🔔 فعال", value=True, key="alert_active")
                alert_repeat = st.checkbox("🔄 تکرار", value=False, key="alert_repeat")
            
            if st.button("➕ افزودن هشدار", use_container_width=True):
                new_alert = {
                    'id': len(st.session_state.alerts) + 1,
                    'symbol': alert_symbol,
                    'type': alert_type,
                    'value': alert_value,
                    'active': alert_active,
                    'repeat': alert_repeat,
                    'created': datetime.now(),
                    'triggered': False,
                    'last_trigger': None
                }
                
                st.session_state.alerts.append(new_alert)
                st.success("✅ هشدار با موفقیت افزوده شد!")
                st.rerun()
        
        # Display existing alerts
        if st.session_state.alerts:
            st.subheader("📋 هشدارهای موجود")
            
            alerts_data = []
            for alert in st.session_state.alerts:
                status = "🟢 فعال" if alert['active'] else "🔴 غیرفعال"
                triggered = "✅ فعال شده" if alert['triggered'] else "⏳ انتظار"
                
                alerts_data.append({
                    'ID': alert['id'],
                    'نماد': alert['symbol'],
                    'نوع': alert['type'],
                    'مقدار': alert['value'] if alert['value'] else '-',
                    'وضعیت': status,
                    'فعالیت': triggered,
                    'تاریخ': alert['created'].strftime('%Y-%m-%d %H:%M')
                })
            
            df_alerts = pd.DataFrame(alerts_data)
            st.dataframe(df_alerts, use_container_width=True)
            
            # Alert actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔍 بررسی هشدارها"):
                    triggered_alerts = []
                    
                    for alert in st.session_state.alerts:
                        if not alert['active']:
                            continue
                        
                        try:
                            # Get current data
                            data, success = get_market_data(alert['symbol'], 'H1', 50)
                            
                            if success:
                                data = calculate_all_indicators(data)
                                current_price = data['close'].iloc[-1]
                                current_rsi = data['RSI'].iloc[-1]
                                
                                # Check alert conditions
                                triggered = False
                                
                                if alert['type'] == 'قیمت بالاتر از' and current_price > alert['value']:
                                    triggered = True
                                elif alert['type'] == 'قیمت پایین‌تر از' and current_price < alert['value']:
                                    triggered = True
                                elif alert['type'] == 'RSI بالاتر از' and current_rsi > alert['value']:
                                    triggered = True
                                elif alert['type'] == 'RSI پایین‌تر از' and current_rsi < alert['value']:
                                    triggered = True
                                elif alert['type'] == 'سیگنال خرید':
                                    analysis = advanced_market_analysis(data, alert['symbol'])
                                    if 'خرید' in analysis['overall_signal'] and analysis['confidence'] > 75:
                                        triggered = True
                                elif alert['type'] == 'سیگنال فروش':
                                    analysis = advanced_market_analysis(data, alert['symbol'])
                                    if 'فروش' in analysis['overall_signal'] and analysis['confidence'] > 75:
                                        triggered = True
                                elif alert['type'] == 'Smart Money فعالیت':
                                    smart_analysis = smart_money_analysis(data, alert['symbol'])
                                    if 'Smart Money' in smart_analysis['smart_signal']:
                                        triggered = True
                                
                                if triggered:
                                    alert['triggered'] = True
                                    alert['last_trigger'] = datetime.now()
                                    triggered_alerts.append(alert)
                                    
                                    if not alert['repeat']:
                                        alert['active'] = False
                        
                        except:
                            continue
                    
                    if triggered_alerts:
                        st.success(f"🚨 {len(triggered_alerts)} هشدار فعال شد!")
                        for alert in triggered_alerts:
                            st.warning(f"🔔 {alert['symbol']}: {alert['type']} - {alert['value']}")
                    else:
                        st.info("ℹ️ هیچ هشداری فعال نشد")
            
            with col2:
                selected_alert_id = st.selectbox("انتخاب هشدار:", 
                                               [alert['id'] for alert in st.session_state.alerts],
                                               key="selected_alert")
                
                if st.button("🗑️ حذف هشدار"):
                    st.session_state.alerts = [a for a in st.session_state.alerts if a['id'] != selected_alert_id]
                    st.success("✅ هشدار حذف شد!")
                    st.rerun()
            
            with col3:
                if st.button("🔄 فعال/غیرفعال کردن"):
                    for alert in st.session_state.alerts:
                        if alert['id'] == selected_alert_id:
                            alert['active'] = not alert['active']
                            status = "فعال" if alert['active'] else "غیرفعال"
                            st.success(f"✅ هشدار {status} شد!")
                            st.rerun()
                            break
        
        else:
            st.info("📝 هیچ هشداری تعریف نشده است")
        
        st.markdown("---")
        
        # System settings
        st.subheader("⚙️ تنظیمات سیستم")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎛️ تنظیمات عمومی")
            
            # Default timeframe
            default_timeframe = st.selectbox("⏰ تایم فریم پیش‌فرض:", 
                                           ['M15', 'H1', 'H4', 'D1'], 
                                           index=1, key="default_tf")
            
            # اگه تنظیمات بیشتری هست، اینجا اضافه کن
            # ...
    
    # بعد از بسته شدن tab6، tab7 شروع میشه
    with tab7:
        st.title("🤖 ربات معاملاتی هوشمند")
    
    # تنظیمات اولیه
    if 'bot_active' not in st.session_state:
        st.session_state.bot_active = False
    if 'ai_analysis' not in st.session_state:
        st.session_state.ai_analysis = {}
    
    # کنترل ربات
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        if st.session_state.bot_active:
            if st.button("⏸️ توقف ربات", type="secondary", use_container_width=True):
                st.session_state.bot_active = False
                st.warning("⚠️ ربات متوقف شد!")
                st.rerun()
        else:
            if st.button("🚀 شروع ربات هوشمند", type="primary", use_container_width=True):
                st.session_state.bot_active = True
                st.success("✅ ربات هوشمند فعال شد!")
                st.rerun()
    
    with control_col2:
        if st.button("🛑 بستن همه معاملات", use_container_width=True):
            st.error("🛑 همه معاملات بسته شد!")
    
    with control_col3:
        bot_status = "🟢 فعال" if st.session_state.bot_active else "🔴 غیرفعال"
        st.metric("🤖 وضعیت", bot_status)
    
    st.markdown("---")
    
    # تحلیل هوش مصنوعی
    if st.session_state.bot_active:
        st.markdown("### 🧠 تحلیل هوش مصنوعی")
        
        # شبیه‌سازی تحلیل AI
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "XAUUSD"]
        
        for symbol in symbols:
            with st.expander(f"💱 {symbol}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                # شبیه‌سازی درصدها
                import random
                ai_score = random.randint(65, 95)
                smart_money = random.randint(60, 90)
                technical = random.randint(55, 85)
                
                # محاسبه نمره کل
                total_score = (ai_score + smart_money + technical) / 3
                
                with col1:
                    st.metric("🧠 AI مدل", f"{ai_score}%")
                
                with col2:
                    st.metric("💰 Smart Money", f"{smart_money}%")
                
                with col3:
                    st.metric("📊 تحلیل تکنیکال", f"{technical}%")
                
                with col4:
                    if total_score >= 80:
                        signal_color = "🟢"
                        signal_text = "خرید قوی"
                    elif total_score >= 70:
                        signal_color = "🟡"
                        signal_text = "خرید متوسط"
                    elif total_score <= 40:
                        signal_color = "🔴"
                        signal_text = "فروش قوی"
                    else:
                        signal_color = "⚪"
                        signal_text = "صبر"
                    
                    st.metric("🎯 سیگنال", f"{signal_color} {signal_text}")
                
                # نمایش توضیحات
                if total_score >= 75:
                    st.success(f"✅ سیگنال ورود {total_score:.1f}% - بر اساس ترکیب AI + Smart Money + تحلیل تکنیکال")
                elif total_score >= 60:
                    st.warning(f"⚠️ سیگنال متوسط {total_score:.1f}% - احتیاط در ورود")
                else:
                    st.info(f"ℹ️ سیگنال ضعیف {total_score:.1f}% - انتظار سیگنال بهتر")
        
        # تنظیمات ریسک
        st.markdown("### ⚙️ مدیریت ریسک خودکار")
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            auto_lot = st.number_input("📊 سایز لات:", value=0.1, step=0.01)
        
        with risk_col2:
            auto_sl = st.number_input("🛑 Stop Loss:", value=50, step=5)
        
        with risk_col3:
            auto_tp = st.number_input("🎯 Take Profit:", value=100, step=10)
        
        # آمار عملکرد
        st.markdown("### 📊 عملکرد زنده")
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("💰 موجودی", "$1,000.00", "0%")
        
        with perf_col2:
            st.metric("📈 سود/زیان امروز", "+$0.00", "0%")
        
        with perf_col3:
            st.metric("📊 معاملات باز", "0")
        
        with perf_col4:
            st.metric("🎯 نرخ موفقیت", "0%")
    
    else:
        st.info("🤖 ربات غیرفعال - برای شروع تحلیل، ربات را فعال کنید")
        
        # پیش‌نمایش تنظیمات
        st.markdown("### ⚙️ تنظیمات پیش‌فرض")
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            st.info("🧠 **مدل AI**: ترکیب RSI+MACD+Smart Money")
            st.info("📊 **حداقل اعتماد**: 75% برای ورود")
        
        with settings_col2:
            st.info("💰 **مدیریت ریسک**: خودکار")
            st.info("🎯 **نمادها**: 7 ارز اصلی")
        
        # تنظیمات اولیه session state
        if 'bot_active' not in st.session_state:
            st.session_state.bot_active = False
        # باقی کد کامل ربات...
            
            # Default confidence threshold
            default_confidence = st.slider("🎯 حداقل اعتماد پیش‌فرض:", 50, 95, 70, 
                                         key="default_conf")
            
            # Auto-save settings
            auto_save = st.checkbox("💾 ذخیره خودکار تحلیل‌ها", value=True)
            
            # Theme settings
            theme_mode = st.selectbox("🎨 حالت نمایش:", ['روشن', 'تیره', 'خودکار'], 
                                    key="theme_mode")
        
        with col2:
            st.markdown("### 📊 تنظیمات تحلیل")
            
            # Analysis depth
            analysis_depth = st.selectbox("🔍 عمق تحلیل:", 
                                        ['سریع', 'متوسط', 'کامل'], 
                                        index=1, key="analysis_depth")
            
            # Smart Money sensitivity
            sm_sensitivity = st.slider("🧠 حساسیت Smart Money:", 1, 10, 5, 
                                     key="sm_sensitivity")
            
            # Chart update frequency
            chart_update = st.selectbox("📈 فرکانس بروزرسانی نمودار:", 
                                      ['10 ثانیه', '30 ثانیه', '1 دقیقه', '5 دقیقه'], 
                                      index=1, key="chart_update")
            
            # Risk management
            risk_mode = st.selectbox("⚠️ حالت مدیریت ریسک:", 
                                   ['محافظه‌کار', 'متعادل', 'تهاجمی'], 
                                   index=1, key="risk_mode")
        
        # Save settings
        if st.button("💾 ذخیره تنظیمات", type="primary", use_container_width=True):
            settings = {
                'default_timeframe': default_timeframe,
                'default_confidence': default_confidence,
                'auto_save': auto_save,
                'theme_mode': theme_mode,
                'analysis_depth': analysis_depth,
                'sm_sensitivity': sm_sensitivity,
                'chart_update': chart_update,
                'risk_mode': risk_mode,
                'saved_at': datetime.now()
            }
            
            # In a real app, you would save to a database or file
            st.session_state.system_settings = settings
            st.success("✅ تنظیمات با موفقیت ذخیره شد!")
        
        st.markdown("---")
        
        # Data management
        st.subheader("📊 مدیریت داده‌ها")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ پاک کردن تاریخچه تحلیل"):
                st.session_state.analysis_history = []
                st.success("✅ تاریخچه تحلیل پاک شد!")
        
        with col2:
            if st.button("🗑️ پاک کردن داده بازار"):
                st.session_state.market_data = {}
                st.success("✅ داده بازار پاک شد!")
        
        with col3:
            if st.button("🔄 ریست کامل سیستم"):
                for key in ['analysis_history', 'market_data', 'alerts', 'system_settings']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("✅ سیستم ریست شد!")
                st.rerun()
        
        # System status
        st.subheader("📊 وضعیت سیستم")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            analysis_count = len(st.session_state.get('analysis_history', []))
            st.metric("📊 تعداد تحلیل‌ها", analysis_count)
        
        with col2:
            market_data_count = len(st.session_state.get('market_data', {}))
            st.metric("💾 نمادهای ذخیره شده", market_data_count)
        
        with col3:
            alert_count = len(st.session_state.get('alerts', []))
            st.metric("🚨 تعداد هشدارها", alert_count)
        
        with col4:
            watchlist_count = len(st.session_state.get('watchlist', []))
            st.metric("👁️ نمادهای واچ لیست", watchlist_count)
        
        # Export/Import functionality
        st.subheader("📤 Export/Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export تنظیمات"):
                export_data = {
                    'watchlist': st.session_state.get('watchlist', []),
                    'alerts': st.session_state.get('alerts', []),
                    'system_settings': st.session_state.get('system_settings', {}),
                    'export_date': datetime.now().isoformat()
                }
                
                st.download_button(
                    label="💾 دانلود فایل تنظیمات",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False),
                    file_name=f"forex_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            st.write("📤 Import تنظیمات")
            uploaded_file = st.file_uploader("انتخاب فایل تنظیمات", type=['json'])
            
            if uploaded_file is not None:
                try:
                    import_data = json.loads(uploaded_file.read())
                    
                    if st.button("📥 Import کن"):
                        if 'watchlist' in import_data:
                            st.session_state.watchlist = import_data['watchlist']
                        if 'alerts' in import_data:
                            st.session_state.alerts = import_data['alerts']
                        if 'system_settings' in import_data:
                            st.session_state.system_settings = import_data['system_settings']
                        
                        st.success("✅ تنظیمات با موفقیت Import شد!")
                        st.rerun()
                
                except:
                    st.error("❌ خطا در خواندن فایل!")

else:
    # Not connected to MT5
    st.warning("⚠️ برای استفاده از سیستم، ابتدا به MT5 متصل شوید!")
    
    st.markdown("""
    ### 🚀 سیستم فارکس کامل و پیشرفته
    
    **ویژگی‌های کلیدی:**
    - 🧠 **تحلیل هوشمند AI** با اعتماد بالا
    - 💰 **Smart Money Analysis** برای تشخیص نهادها  
    - 🔍 **اسکنر پیشرفته بازار** با فیلترهای متنوع
    - 📈 **نمودارهای زنده** با اندیکاتورهای کامل
    - 🚨 **سیستم هشدار پیشرفته** با اعلان‌های هوشمند
    - 🤖 **ربات معاملاتی** با مدیریت ریسک
    - 📊 **گزارشات جامع** عملکرد و تحلیل
    
    **برای شروع:**
    1. از منوی کناری گزینه "اتصال سریع" را انتخاب کنید
    2. یا اطلاعات MT5 خود را وارد کنید
    3. پس از اتصال، تمامی امکانات در دسترس خواهد بود
    """)
    
    # Demo Features Preview
    st.markdown("### 🎮 پیش‌نمایش امکانات")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="padding: 1rem; border: 2px solid #blue; border-radius: 10px; text-align: center;">
            <h4>🧠 تحلیل AI</h4>
            <p>تحلیل هوشمند با 15+ اندیکاتور</p>
            <p><strong>اعتماد:</strong> تا 95%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 1rem; border: 2px solid #green; border-radius: 10px; text-align: center;">
            <h4>💰 Smart Money</h4>
            <p>تشخیص فعالیت نهادی</p>
            <p><strong>Order Blocks & Liquidity</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="padding: 1rem; border: 2px solid #orange; border-radius: 10px; text-align: center;">
            <h4>🤖 ربات معاملاتی</h4>
            <p>معاملات خودکار</p>
            <p><strong>مدیریت ریسک هوشمند</strong></p>
        </div>
        """, unsafe_allow_html=True)

# در اینجا کد دیگری وجود داشت که قطع شده بود
# حالا بخش صحیح account info اضافه می‌کنم

# Account Info Section (این بخش باید در قسمت MT5 connected باشه)
if st.session_state.mt5_connected:
    # Account info display
    account = st.session_state.account_info
    
    st.markdown(f"""
    <div style="padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin: 1rem 0;">
        <h4>📊 اطلاعات حساب</h4>
        <strong>🏢 شرکت:</strong> {account['company']}<br>
        <strong>💰 موجودی:</strong> ${account['balance']:,.2f}<br>
        <strong>💎 ارزش خالص:</strong> ${account['equity']:,.2f}<br>
        <strong>🎯 اهرم:</strong> 1:{account.get('leverage', 'N/A')}<br>
        <strong>💸 مارژین آزاد:</strong> ${account.get('free_margin', 0):,.2f}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ عملیات سریع")
    
    if st.button("📊 اسکن بازار", use_container_width=True):
        st.session_state.quick_scan = True
    
    if st.button("🔄 به‌روزرسانی قیمت‌ها", use_container_width=True):
        st.session_state.update_prices = True
    
    if st.button("🎯 تحلیل Smart Money", use_container_width=True):
        st.session_state.smart_analysis = True
    
    st.markdown("---")
    
    # Watchlist
    st.markdown("### 👀 لیست پیگیری")
    
    for symbol in st.session_state.watchlist[:5]:
        live_data = get_live_price(symbol)
        if live_data:
            st.write(f"**{symbol}:** {live_data['bid']:.5f}")
    
    # Add to watchlist
    new_symbol = st.text_input("➕ افزودن نماد:", key="add_watchlist")
    if st.button("➕ افزودن") and new_symbol:
        if new_symbol.upper() not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_symbol.upper())
            st.success(f"✅ {new_symbol} اضافه شد!")
            st.rerun()
    
    st.markdown("---")
    
    # System Status
    st.markdown("### 🔋 وضعیت سیستم")
    current_time = datetime.now()
    st.write(f"🕒 {current_time.strftime('%H:%M:%S')}")
    st.write(f"📅 {current_time.strftime('%Y-%m-%d')}")
    
    if 'market_data' in st.session_state and st.session_state.market_data:
        st.write(f"📊 داده آماده: {len(st.session_state.market_data)} نماد")
        
        if st.button("❌ قطع اتصال", type="secondary"):
            mt5.shutdown()
            st.session_state.mt5_connected = False
            st.session_state.clear()
            st.rerun()

# Main Application
if not st.session_state.mt5_connected:
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>🔌 لطفاً ابتدا به MT5 متصل شوید</h2>
        <p>برای استفاده از سیستم تحلیل فارکس، ابتدا باید به پلتفرم MetaTrader 5 متصل شوید.</p>
        <p>از منوی کناری گزینه "اتصال سریع" را انتخاب کنید.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Get available symbols
    symbol_categories = get_symbol_categories()
    
    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 داشبورد اصلی",
        "📊 تحلیل پیشرفته", 
        "💰 Smart Money",
        "📈 نمودارهای زنده",
        "🔔 هشدارها و اسکن",
        "🤖 ربات معاملاتی",
        "📋 گزارشات"
    ])
    
    with tab1:
        st.header("🎯 داشبورد اصلی")
        
        # Quick Overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>🔗 اتصال</h4>
                <h2 style="color: green;">✅ متصل</h2>
                <p>MT5 فعال</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            account_info = st.session_state.account_info
            st.markdown(f"""
            <div class="metric-card">
                <h4>💰 موجودی</h4>
                <h2>${account_info['balance']:,.0f}</h2>
                <p>ارزش: ${account_info['equity']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            active_signals = len(st.session_state.trading_signals)
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 سیگنال‌ها</h4>
                <h2>{active_signals}</h2>
                <p>سیگنال فعال</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            watched_symbols = len(st.session_state.watchlist)
            st.markdown(f"""
            <div class="metric-card">
                <h4>👀 پیگیری</h4>
                <h2>{watched_symbols}</h2>
                <p>نماد در لیست</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Analysis Section
        st.subheader("⚡ تحلیل سریع")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Symbol selection
            selected_category = st.selectbox(
                "📂 دسته‌بندی نمادها:",
                ['forex_major', 'gold_metals', 'forex_minor', 'commodities', 'indices'],
                format_func=lambda x: {
                    'forex_major': '💱 فارکس اصلی',
                    'gold_metals': '🥇 طلا و فلزات',
                    'forex_minor': '💸 فارکس فرعی',
                    'commodities': '🛢️ کالاها',
                    'indices': '📈 شاخص‌ها'
                }[x]
            )
            
            available_symbols = symbol_categories.get(selected_category, [])
            if available_symbols:
                quick_symbol = st.selectbox("💱 انتخاب نماد:", available_symbols)
            else:
                quick_symbol = st.text_input("💱 نماد (مثال: EURUSD):", value="EURUSD")
        
        with col2:
            quick_timeframe = st.selectbox(
                "⏰ تایم فریم:",
                ['M15', 'H1', 'H4', 'D1'],
                index=1
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 تحلیل سریع", type="primary", use_container_width=True):
                st.session_state.quick_analysis = {
                    'symbol': quick_symbol,
                    'timeframe': quick_timeframe,
                    'requested': True
                }
        
        # Quick Analysis Results
        if 'quick_analysis' in st.session_state and st.session_state.quick_analysis.get('requested'):
            analysis_data = st.session_state.quick_analysis
            
            with st.spinner(f"🔄 تحلیل {analysis_data['symbol']} در {analysis_data['timeframe']}..."):
                # Get data and analyze
                data, success = get_market_data(analysis_data['symbol'], analysis_data['timeframe'], 200)
                
                if success:
                    data = calculate_all_indicators(data)
                    analysis = advanced_market_analysis(data, analysis_data['symbol'])
                    smart_analysis = smart_money_analysis(data, analysis_data['symbol'])
                    live_price = get_live_price(analysis_data['symbol'])
                    
                    # Display results
                    st.markdown("### 📊 نتایج تحلیل سریع")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        signal_color = 'green' if 'خرید' in analysis['overall_signal'] else 'red' if 'فروش' in analysis['overall_signal'] else 'gray'
                        st.markdown(f"""
                        <div style="background: {signal_color}; color: white; padding: 1rem; border-radius: 5px; text-align: center;">
                            <h4>🎯 سیگنال اصلی</h4>
                            <h3>{analysis['overall_signal']}</h3>
                            <p>اعتماد: {analysis['confidence']}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("📈 روند", analysis['trend'], f"قدرت: {analysis['strength']}")
                    
                    with col3:
                        st.metric("⚠️ ریسک", analysis['risk_level'], f"نوسان: {analysis['volatility']:.3f}%")
                    
                    with col4:
                        if live_price:
                            st.metric("💰 قیمت فعلی", f"{live_price['bid']:.5f}", f"اسپرد: {live_price['spread']:.1f}")
                    
                    with col5:
                        st.metric("🧠 Smart Money", smart_analysis['smart_signal'][:15], smart_analysis['institution_activity'])
                    
                    # Key Indicators
                    st.markdown("### 📊 اندیکاتورهای کلیدی")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        rsi_value = analysis['indicators']['RSI']
                        rsi_color = 'red' if rsi_value > 70 else 'green' if rsi_value < 30 else 'blue'
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; border: 2px solid {rsi_color}; border-radius: 5px;">
                            <h4>RSI</h4>
                            <h2 style="color: {rsi_color};">{rsi_value:.1f}</h2>
                            <p>{'خرید بیش از حد' if rsi_value > 70 else 'فروش بیش از حد' if rsi_value < 30 else 'عادی'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        macd_value = analysis['indicators']['MACD']
                        macd_signal = analysis['indicators']['MACD_Signal']
                        macd_color = 'green' if macd_value > macd_signal else 'red'
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; border: 2px solid {macd_color}; border-radius: 5px;">
                            <h4>MACD</h4>
                            <h2 style="color: {macd_color};">{macd_value:.5f}</h2>
                            <p>{'صعودی' if macd_value > macd_signal else 'نزولی'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        bb_position = analysis['indicators']['BB_Position']
                        bb_color = 'red' if bb_position > 80 else 'green' if bb_position < 20 else 'blue'
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; border: 2px solid {bb_color}; border-radius: 5px;">
                            <h4>Bollinger Position</h4>
                            <h2 style="color: {bb_color};">{bb_position:.1f}%</h2>
                            <p>{'بالای باند' if bb_position > 80 else 'پایین باند' if bb_position < 20 else 'وسط'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        volume_ratio = analysis['indicators']['Volume_Ratio']
                        vol_color = 'orange' if volume_ratio > 1.5 else 'blue'
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; border: 2px solid {vol_color}; border-radius: 5px;">
                            <h4>Volume Ratio</h4>
                            <h2 style="color: {vol_color};">{volume_ratio:.2f}x</h2>
                            <p>{'حجم بالا' if volume_ratio > 1.5 else 'حجم عادی'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Quick Chart
                    st.markdown("### 📈 نمودار سریع")
                    
                    fig = go.Figure()
                    
                    # Display last 50 candles
                    chart_data = data.tail(50)
                    
                    # Candlestick
                    fig.add_trace(go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['open'],
                        high=chart_data['high'],
                        low=chart_data['low'],
                        close=chart_data['close'],
                        name=analysis_data['symbol']
                    ))
                    
                    # Add EMA 20
                    fig.add_trace(go.Scatter(
                        x=chart_data.index,
                        y=chart_data['EMA_20'],
                        mode='lines',
                        name='EMA 20',
                        line=dict(color='blue', width=1)
                    ))
                    
                    # Add EMA 50
                    fig.add_trace(go.Scatter(
                        x=chart_data.index,
                        y=chart_data['EMA_50'],
                        mode='lines',
                        name='EMA 50',
                        line=dict(color='orange', width=1)
                    ))
                    
                    fig.update_layout(
                        title=f"{analysis_data['symbol']} - {analysis_data['timeframe']} - {analysis['overall_signal']}",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save to session
                    st.session_state.quick_analysis['requested'] = False
                    
                    # Add to trading signals if strong
                    if analysis['confidence'] >= 75:
                        signal_entry = {
                            'symbol': analysis_data['symbol'],
                            'signal': analysis['overall_signal'],
                            'confidence': analysis['confidence'],
                            'timeframe': analysis_data['timeframe'],
                            'time': datetime.now(),
                            'price': live_price['bid'] if live_price else data['close'].iloc[-1],
                            'analysis': analysis
                        }
                        
                        # Check if already exists
                        existing = [s for s in st.session_state.trading_signals if s['symbol'] == analysis_data['symbol']]
                        if not existing:
                            st.session_state.trading_signals.append(signal_entry)
                            st.success(f"✅ سیگنال قوی {analysis_data['symbol']} به لیست اضافه شد!")
                
                else:
                    st.error(f"❌ خطا در دریافت داده {analysis_data['symbol']}")
        
        st.markdown("---")
        
        # Recent Signals
        if st.session_state.trading_signals:
            st.subheader("🎯 سیگنال‌های اخیر")
            
            signals_df = pd.DataFrame([
                {
                    'نماد': s['symbol'],
                    'سیگنال': s['signal'],
                    'اعتماد': f"{s['confidence']}%",
                    'تایم فریم': s['timeframe'],
                    'قیمت': f"{s['price']:.5f}",
                    'زمان': s['time'].strftime('%H:%M')
                }
                for s in st.session_state.trading_signals[-10:]  # Last 10
            ])
            
            st.dataframe(signals_df, use_container_width=True)
        
        # Market Overview
        st.markdown("---")
        st.subheader("🌍 نمای کلی بازار")
        
        if st.button("🔄 به‌روزرسانی نمای کلی", type="secondary"):
            with st.spinner("🔄 به‌روزرسانی..."):
                # Quick scan of major pairs
                major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD']
                
                overview_data = []
                for symbol in major_pairs:
                    live_price = get_live_price(symbol)
                    if live_price:
                        # Get quick data for trend
                        data, success = get_market_data(symbol, 'H1', 50)
                        if success:
                            current_price = live_price['bid']
                            ema_20 = data['close'].ewm(span=20).mean().iloc[-1]
                            change_24h = ((current_price / data['close'].iloc[-24]) - 1) * 100 if len(data) >= 24 else 0
                            
                            trend = "📈" if current_price > ema_20 else "📉"
                            
                            overview_data.append({
                                'نماد': symbol,
                                'قیمت': f"{current_price:.5f}",
                                'تغییر 24h': f"{change_24h:+.2f}%",
                                'روند': trend,
                                'اسپرد': f"{live_price['spread']:.1f}"
                            })
                
                if overview_data:
                    overview_df = pd.DataFrame(overview_data)
                    st.dataframe(overview_df, use_container_width=True)
                else:
                    st.warning("⚠️ داده نمای کلی دریافت نشد")
    
    with tab2:
        st.header("📊 تحلیل پیشرفته")
        
        # Symbol and Timeframe Selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            adv_category = st.selectbox(
                "📂 دسته نماد:",
                ['forex_major', 'gold_metals', 'forex_minor', 'commodities', 'indices'],
                format_func=lambda x: {
                    'forex_major': '💱 فارکس اصلی',
                    'gold_metals': '🥇 طلا و فلزات',
                    'forex_minor': '💸 فارکس فرعی',
                    'commodities': '🛢️ کالاها',
                    'indices': '📈 شاخص‌ها'
                }[x],
                key="adv_category"
            )
        
        with col2:
            adv_symbols = symbol_categories.get(adv_category, ['EURUSD'])
            if adv_symbols:
                adv_symbol = st.selectbox("💱 نماد:", adv_symbols, key="adv_symbol")
            else:
                adv_symbol = st.text_input("💱 نماد:", value="EURUSD", key="adv_symbol_input")
        
        with col3:
            adv_timeframe = st.selectbox(
                "⏰ تایم فریم:",
                ['M15', 'H1', 'H4', 'D1', 'W1'],
                index=1,
                key="adv_timeframe"
            )
        
        # Analysis Options
        st.markdown("### ⚙️ گزینه‌های تحلیل")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_indicators = st.multiselect(
                "📊 اندیکاتورها:",
                ['RSI', 'MACD', 'Bollinger Bands', 'Stochastic', 'Williams %R', 'ATR'],
                default=['RSI', 'MACD', 'Bollinger Bands']
            )
        
        with col2:
            show_ma = st.multiselect(
                "📈 میانگین متحرک:",
                ['SMA 20', 'SMA 50', 'SMA 200', 'EMA 20', 'EMA 50', 'EMA 200'],
                default=['EMA 20', 'EMA 50']
            )
        
        with col3:
            analysis_period = st.slider("📊 تعداد کندل:", 100, 1000, 500, step=50)
        
        # Run Advanced Analysis
        if st.button("🚀 تحلیل پیشرفته کامل", type="primary"):
            with st.spinner(f"🔄 تحلیل پیشرفته {adv_symbol}..."):
                # Get comprehensive data
                data, success = get_market_data(adv_symbol, adv_timeframe, analysis_period)
                
                if success:
                    # Calculate all indicators
                    data = calculate_all_indicators(data)
                    
                    # Perform advanced analysis
                    advanced_result = advanced_market_analysis(data, adv_symbol)
                    smart_result = smart_money_analysis(data, adv_symbol)
                    live_price = get_live_price(adv_symbol)
                    
                    # Store results
                    st.session_state.advanced_analysis = {
                        'symbol': adv_symbol,
                        'timeframe': adv_timeframe,
                        'data': data,
                        'analysis': advanced_result,
                        'smart_money': smart_result,
                        'live_price': live_price,
                        'timestamp': datetime.now()
                    }
                    
                    st.success("✅ تحلیل پیشرفته کامل شد!")
                else:
                    st.error(f"❌ خطا در دریافت داده {adv_symbol}")
        
        # Display Advanced Analysis Results
        if 'advanced_analysis' in st.session_state:
            result = st.session_state.advanced_analysis
            
            st.markdown("---")
            st.markdown(f"### 📊 نتایج تحلیل پیشرفته - {result['symbol']} ({result['timeframe']})")
            
            # Summary Cards
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                signal = result['analysis']['overall_signal']
                confidence = result['analysis']['confidence']
                signal_color = 'green' if 'خرید' in signal else 'red' if 'فروش' in signal else 'gray'
                
                st.markdown(f"""
                <div class="signal-{'bullish' if 'خرید' in signal else 'bearish' if 'فروش' in signal else 'neutral'}">
                    <h4>🎯 سیگنال نهایی</h4>
                    <h3>{signal}</h3>
                    <p>اعتماد: {confidence}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                trend = result['analysis']['trend']
                strength = result['analysis']['strength']
                st.metric("📈 روند بازار", trend, f"قدرت: {strength}")
            
            with col3:
                risk = result['analysis']['risk_level']
                volatility = result['analysis']['volatility']
                st.metric("⚠️ ریسک", risk, f"نوسان: {volatility:.3f}%")
            
            with col4:
                if result['live_price']:
                    current_price = result['live_price']['bid']
                    spread = result['live_price']['spread']
                    st.metric("💰 قیمت زنده", f"{current_price:.5f}", f"اسپرد: {spread:.1f}")
            
            with col5:
                smart_signal = result['smart_money']['smart_signal']
                institution = result['smart_money']['institution_activity']
                st.metric("🧠 Smart Money", smart_signal[:15], f"نهادی: {institution}")
            
            # Detailed Indicators
            st.markdown("### 📊 اندیکاتورهای تفصیلی")
            
            indicators = result['analysis']['indicators']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("#### 📈 اسیلاتورها")
                
                # RSI
                rsi_val = indicators['RSI']
                rsi_status = "خرید بیش از حد" if rsi_val > 70 else "فروش بیش از حد" if rsi_val < 30 else "عادی"
                st.write(f"**RSI:** {rsi_val:.1f} - {rsi_status}")
                
                # Stochastic
                stoch_val = indicators['Stoch_K']
                stoch_status = "خرید بیش از حد" if stoch_val > 80 else "فروش بیش از حد" if stoch_val < 20 else "عادی"
                st.write(f"**Stochastic:** {stoch_val:.1f} - {stoch_status}")
                
                # Williams %R (if available)
                st.write(f"**Williams %R:** در حال محاسبه...")
            
            with col2:
                st.markdown("#### 📊 MACD")
                
                macd_val = indicators['MACD']
                macd_signal = indicators['MACD_Signal']
                macd_status = "صعودی" if macd_val > macd_signal else "نزولی"
                
                st.write(f"**MACD:** {macd_val:.5f}")
                st.write(f"**Signal:** {macd_signal:.5f}")
                st.write(f"**وضعیت:** {macd_status}")
                
                # Histogram
                histogram = macd_val - macd_signal
                st.write(f"**Histogram:** {histogram:.5f}")
            
            with col3:
                st.markdown("#### 🎯 Bollinger Bands")
                
                bb_position = indicators['BB_Position']
                bb_status = "نزدیک بالا" if bb_position > 80 else "نزدیک پایین" if bb_position < 20 else "وسط"
                
                st.write(f"**Position:** {bb_position:.1f}%")
                st.write(f"**وضعیت:** {bb_status}")
                
                # Levels
                levels = result['analysis']['levels']
                st.write(f"**حمایت:** {levels['support']:.5f}")
                st.write(f"**مقاومت:** {levels['resistance']:.5f}")
            
            with col4:
                st.markdown("#### 📊 حجم و نوسان")
                
                volume_ratio = indicators['Volume_Ratio']
                volume_status = "بالا" if volume_ratio > 1.5 else "پایین" if volume_ratio < 0.8 else "عادی"
                
                st.write(f"**Volume Ratio:** {volume_ratio:.2f}x")
                st.write(f"**وضعیت حجم:** {volume_status}")
                
                atr_val = indicators['ATR']
                st.write(f"**ATR:** {atr_val:.5f}")
                st.write(f"**نوسان:** {result['analysis']['volatility']:.3f}%")
            
            # Price Levels
            st.markdown("### 🎯 سطوح کلیدی قیمت")
            
            levels = result['analysis']['levels']
            current_price = levels['current_price']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💰 قیمت فعلی", f"{current_price:.5f}")
            
            with col2:
                resistance = levels['resistance']
                distance_resistance = ((resistance / current_price) - 1) * 100
                st.metric("🔴 مقاومت", f"{resistance:.5f}", f"{distance_resistance:+.2f}%")
            
            with col3:
                support = levels['support']
                distance_support = ((support / current_price) - 1) * 100
                st.metric("🟢 حمایت", f"{support:.5f}", f"{distance_support:+.2f}%")
            
            with col4:
                risk_reward = abs(distance_resistance) / abs(distance_support) if distance_support != 0 else 0
                st.metric("⚖️ ریسک/بازده", f"1:{risk_reward:.2f}")
            
            # Moving Averages Table
            st.markdown("### 📈 میانگین‌های متحرک")
            
            ma_data = {
                'نوع': ['EMA 20', 'EMA 50', 'SMA 20', 'SMA 50'],
                'مقدار': [
                    f"{levels['ema_20']:.5f}",
                    f"{levels['ema_50']:.5f}",
                    f"{levels['sma_20']:.5f}",
                    f"{levels['sma_50']:.5f}"
                ],
                'فاصله از قیمت': [
                    f"{((levels['ema_20'] / current_price) - 1) * 100:+.2f}%",
                    f"{((levels['ema_50'] / current_price) - 1) * 100:+.2f}%",
                    f"{((levels['sma_20'] / current_price) - 1) * 100:+.2f}%",
                    f"{((levels['sma_50'] / current_price) - 1) * 100:+.2f}%"
                ],
                'وضعیت': [
                    "🟢 بالاتر" if current_price > levels['ema_20'] else "🔴 پایین‌تر",
                    "🟢 بالاتر" if current_price > levels['ema_50'] else "🔴 پایین‌تر",
                    "🟢 بالاتر" if current_price > levels['sma_20'] else "🔴 پایین‌تر",
                    "🟢 بالاتر" if current_price > levels['sma_50'] else "🔴 پایین‌تر"
                ]
            }
            
            st.dataframe(pd.DataFrame(ma_data), use_container_width=True)
            
            # Advanced Chart
            st.markdown("### 📈 نمودار پیشرفته")
            
            # Chart controls
            chart_col1, chart_col2, chart_col3 = st.columns(3)
            
            with chart_col1:
                chart_candles = st.slider("تعداد کندل نمودار:", 50, 200, 100)
            
            with chart_col2:
                chart_indicators = st.multiselect(
                    "اندیکاتورها روی نمودار:",
                    ['EMA 20', 'EMA 50', 'SMA 20', 'SMA 50', 'Bollinger Bands'],
                    default=['EMA 20', 'EMA 50'],
                    key="chart_indicators"
                )
            
            with chart_col3:
                show_volume = st.checkbox("نمایش حجم", value=True)
            
            # Create advanced chart
            chart_data = result['data'].tail(chart_candles)
            
            # Main price chart
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=chart_data.index,
                open=chart_data['open'],
                high=chart_data['high'],
                low=chart_data['low'],
                close=chart_data['close'],
                name=result['symbol'],
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
            
            # Add selected indicators
            for indicator in chart_indicators:
                if indicator == 'EMA 20':
                    fig.add_trace(go.Scatter(
                        x=chart_data.index, y=chart_data['EMA_20'],
                        mode='lines', name='EMA 20',
                        line=dict(color='blue', width=1.5)
                    ))
                elif indicator == 'EMA 50':
                    fig.add_trace(go.Scatter(
                        x=chart_data.index, y=chart_data['EMA_50'],
                        mode='lines', name='EMA 50',
                        line=dict(color='orange', width=1.5)
                    ))
                elif indicator == 'SMA 20':
                    fig.add_trace(go.Scatter(
                        x=chart_data.index, y=chart_data['SMA_20'],
                        mode='lines', name='SMA 20',
                        line=dict(color='purple', width=1)
                    ))
                elif indicator == 'SMA 50':
                    fig.add_trace(go.Scatter(
                        x=chart_data.index, y=chart_data['SMA_50'],
                        mode='lines', name='SMA 50',
                        line=dict(color='brown', width=1)
                    ))
                elif indicator == 'Bollinger Bands':
                    fig.add_trace(go.Scatter(
                        x=chart_data.index, y=chart_data['BB_Upper'],
                        mode='lines', name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=chart_data.index, y=chart_data['BB_Lower'],
                        mode='lines', name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
                    ))
            
            # Add support/resistance lines
            fig.add_hline(y=levels['resistance'], line_dash="solid", line_color="red", 
                         annotation_text="مقاومت", annotation_position="left")
            fig.add_hline(y=levels['support'], line_dash="solid", line_color="green",
                         annotation_text="حمایت", annotation_position="left")
            
            fig.update_layout(
                title=f"{result['symbol']} - {result['timeframe']} | {result['analysis']['overall_signal']} (اعتماد: {result['analysis']['confidence']}%)",
                xaxis_title="زمان",
                yaxis_title="قیمت",
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume Chart
            if show_volume:
                st.markdown("### 📊 نمودار حجم")
                
                fig_vol = go.Figure()
                
                # Volume bars
                colors = ['green' if row['Bullish'] else 'red' for _, row in chart_data.iterrows()]
                
                fig_vol.add_trace(go.Bar(
                    x=chart_data.index,
                    y=chart_data['tick_volume'],
                    marker_color=colors,
                    opacity=0.7,
                    name='Volume'
                ))
                
                # Volume MA
                fig_vol.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data['Volume_MA'],
                    mode='lines',
                    name='Volume MA',
                    line=dict(color='blue', width=2)
                ))
                
                # High volume markers
                high_vol_data = chart_data[chart_data['Volume_Ratio'] > 1.5]
                if not high_vol_data.empty:
                    fig_vol.add_trace(go.Scatter(
                        x=high_vol_data.index,
                        y=high_vol_data['tick_volume'],
                        mode='markers',
                        marker=dict(color='yellow', size=8, symbol='star'),
                        name='High Volume'
                    ))
                
                fig_vol.update_layout(
                    title="Volume Analysis",
                    height=250,
                    showlegend=True
                )
                
                st.plotly_chart(fig_vol, use_container_width=True)
            
            # Detailed Analysis Summary
            st.markdown("### 📋 خلاصه تحلیل تفصیلی")
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown("#### 🎯 نقاط قوت تحلیل")
                
                analysis_signals = result['analysis']['signals']
                positive_signals = []
                
                for signal_type, signals in analysis_signals.items():
                    for signal in signals:
                        if any(word in signal for word in ['bullish', 'strong', 'up', 'buy', 'support']):
                            positive_signals.append(f"✅ {signal}")
                
                if positive_signals:
                    for signal in positive_signals[:5]:  # Show top 5
                        st.write(signal)
                else:
                    st.write("🔍 در حال بررسی...")
            
            with summary_col2:
                st.markdown("#### ⚠️ نقاط ضعف و ریسک")
                
                negative_signals = []
                
                for signal_type, signals in analysis_signals.items():
                    for signal in signals:
                        if any(word in signal for word in ['bearish', 'weak', 'down', 'sell', 'resistance']):
                            negative_signals.append(f"❌ {signal}")
                
                if negative_signals:
                    for signal in negative_signals[:5]:  # Show top 5
                        st.write(signal)
                else:
                    st.write("🔍 در حال بررسی...")
            
            # Trading Recommendation
            st.markdown("### 🎯 توصیه معاملاتی")
            
            recommendation_col1, recommendation_col2, recommendation_col3 = st.columns(3)
            
            with recommendation_col1:
                action = result['analysis']['action']
                confidence = result['analysis']['confidence']
                
                if action in ['STRONG_BUY', 'BUY']:
                    st.markdown(f"""
                    <div class="signal-bullish">
                        <h4>📈 توصیه: خرید</h4>
                        <p>اعتماد: {confidence}%</p>
                        <p>نوع: {'قوی' if action == 'STRONG_BUY' else 'متوسط'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif action in ['STRONG_SELL', 'SELL']:
                    st.markdown(f"""
                    <div class="signal-bearish">
                        <h4>📉 توصیه: فروش</h4>
                        <p>اعتماد: {confidence}%</p>
                        <p>نوع: {'قوی' if action == 'STRONG_SELL' else 'متوسط'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="signal-neutral">
                        <h4>⏸️ توصیه: انتظار</h4>
                        <p>اعتماد: {confidence}%</p>
                        <p>نوع: خنثی</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with recommendation_col2:
                st.markdown("#### 🎯 سطوح ورود و خروج")
                
                if action in ['STRONG_BUY', 'BUY']:
                    entry_level = levels['support'] + (levels['current_price'] - levels['support']) * 0.3
                    stop_loss = levels['support'] - (levels['current_price'] - levels['support']) * 0.2
                    take_profit = levels['resistance']
                    
                    st.write(f"**ورود:** {entry_level:.5f}")
                    st.write(f"**ضرر:** {stop_loss:.5f}")
                    st.write(f"**هدف:** {take_profit:.5f}")
                    
                elif action in ['STRONG_SELL', 'SELL']:
                    entry_level = levels['resistance'] - (levels['resistance'] - levels['current_price']) * 0.3
                    stop_loss = levels['resistance'] + (levels['resistance'] - levels['current_price']) * 0.2
                    take_profit = levels['support']
                    
                    st.write(f"**ورود:** {entry_level:.5f}")
                    st.write(f"**ضرر:** {stop_loss:.5f}")
                    st.write(f"**هدف:** {take_profit:.5f}")
                
                else:
                    st.write("⏸️ منتظر سیگنال قوی‌تر بمانید")
            
            with recommendation_col3:
                st.markdown("#### ⚖️ مدیریت ریسک")
                
                risk_level = result['analysis']['risk_level']
                volatility = result['analysis']['volatility']
                
                if risk_level == 'بالا':
                    position_size = "1-2% سرمایه"
                    leverage = "1:50 یا کمتر"
                elif risk_level == 'متوسط':
                    position_size = "2-3% سرمایه"
                    leverage = "1:100"
                else:
                    position_size = "3-5% سرمایه"
                    leverage = "1:200"
                
                st.write(f"**ریسک:** {risk_level}")
                st.write(f"**سایز پوزیشن:** {position_size}")
                st.write(f"**اهرم پیشنهادی:** {leverage}")
                st.write(f"**نوسان:** {volatility:.3f}%")
        
        else:
            st.info("🔍 برای مشاهده تحلیل پیشرفته، ابتدا روی دکمه 'تحلیل پیشرفته کامل' کلیک کنید.")
    
    with tab3:
        st.header("💰 Smart Money Analysis")
        
        st.markdown("""
        ### 🧠 تحلیل Smart Money چیست؟
        
        Smart Money تحلیلی است که فعالیت‌های نهادی و بانک‌های بزرگ را شناسایی می‌کند:
        
        - **💼 Order Blocks:** مناطقی که نهادها سفارشات بزرگ دارند
        - **💧 Liquidity Zones:** نقاط تجمع نقدینگی
        - **📊 Volume Analysis:** تحلیل حجم برای تشخیص فعالیت نهادی
        - **🔄 Order Flow:** جریان سفارشات خرید و فروش
        - **🎯 Market Structure:** ساختار بازار و تغییرات آن
        """)
        
        # Smart Money Controls
        smart_col1, smart_col2, smart_col3 = st.columns(3)
        
        with smart_col1:
            smart_category = st.selectbox(
                "📂 دسته نماد:",
                ['forex_major', 'gold_metals', 'forex_minor', 'commodities'],
                format_func=lambda x: {
                    'forex_major': '💱 فارکس اصلی',
                    'gold_metals': '🥇 طلا و فلزات',
                    'forex_minor': '💸 فارکس فرعی',
                    'commodities': '🛢️ کالاها'
                }[x],
                key="smart_category"
            )
        
        with smart_col2:
            smart_symbols = symbol_categories.get(smart_category, ['EURUSD'])
            if smart_symbols:
                smart_symbol = st.selectbox("💱 نماد:", smart_symbols, key="smart_symbol")
            else:
                smart_symbol = st.text_input("💱 نماد:", value="EURUSD", key="smart_symbol_input")
        
        with smart_col3:
            smart_timeframe = st.selectbox(
                "⏰ تایم فریم:",
                ['H1', 'H4', 'D1'],
                index=0,
                key="smart_timeframe"
            )
        
        # Smart Money Settings
        st.markdown("### ⚙️ تنظیمات Smart Money")
        
        settings_col1, settings_col2, settings_col3 = st.columns(3)
        
        with settings_col1:
            volume_threshold = st.slider("🔊 حد آستانه حجم:", 1.2, 3.0, 1.5, 0.1)
            st.caption("حجم بالاتر از میانگین برای تشخیص فعالیت نهادی")
        
        with settings_col2:
            body_ratio_threshold = st.slider("📊 نسبت بدنه کندل:", 0.5, 0.9, 0.7, 0.05)
            st.caption("نسبت بدنه کندل به کل رنج برای Order Block")
        
        with settings_col3:
            lookback_period = st.slider("🔍 دوره بررسی:", 20, 100, 50, 10)
            st.caption("تعداد کندل برای شناسایی الگوها")
        
        # Run Smart Money Analysis
        if st.button("🧠 تحلیل Smart Money کامل", type="primary"):
            with st.spinner(f"🔄 تحلیل Smart Money {smart_symbol}..."):
                # Get data
                data, success = get_market_data(smart_symbol, smart_timeframe, 300)
                
                if success:
                    # Calculate indicators
                    data = calculate_all_indicators(data)
                    
                    # Enhanced Smart Money Analysis
                    smart_result = smart_money_analysis(data, smart_symbol)
                    general_analysis = advanced_market_analysis(data, smart_symbol)
                    live_price = get_live_price(smart_symbol)
                    
                    # Advanced Smart Money Calculations
                    
                    # 1. Order Blocks Detection
                    order_blocks = []
                    for i in range(lookback_period, len(data) - 1):
                        current = data.iloc[i]
                        
                        # Bullish Order Block
                        if (current['close'] > current['open'] and  # Bullish candle
                            current['Body'] / (current['high'] - current['low']) > body_ratio_threshold and  # Strong body
                            current['Volume_Ratio'] > volume_threshold):  # High volume
                            
                            # Check for price return
                            future_data = data.iloc[i+1:i+10]
                            if len(future_data) > 0 and future_data['low'].min() < current['low']:
                                order_blocks.append({
                                    'type': 'Bullish',
                                    'time': current.name,
                                    'high': current['high'],
                                    'low': current['low'],
                                    'volume_ratio': current['Volume_Ratio'],
                                    'body_ratio': current['Body'] / (current['high'] - current['low']),
                                    'strength': 'Strong' if current['Volume_Ratio'] > 2.0 else 'Medium'
                                })
                        
                        # Bearish Order Block
                        elif (current['close'] < current['open'] and  # Bearish candle
                              current['Body'] / (current['high'] - current['low']) > body_ratio_threshold and  # Strong body
                              current['Volume_Ratio'] > volume_threshold):  # High volume
                            
                            # Check for price return
                            future_data = data.iloc[i+1:i+10]
                            if len(future_data) > 0 and future_data['high'].max() > current['high']:
                                order_blocks.append({
                                    'type': 'Bearish',
                                    'time': current.name,
                                    'high': current['high'],
                                    'low': current['low'],
                                    'volume_ratio': current['Volume_Ratio'],
                                    'body_ratio': current['Body'] / (current['high'] - current['low']),
                                    'strength': 'Strong' if current['Volume_Ratio'] > 2.0 else 'Medium'
                                })
                    
                    # 2. Liquidity Zones
                    liquidity_zones = []
                    
                    # Find swing highs and lows
                    for i in range(2, len(data) - 2):
                        current = data.iloc[i]
                        
                        # Swing High
                        if (current['high'] > data.iloc[i-1]['high'] and 
                            current['high'] > data.iloc[i-2]['high'] and
                            current['high'] > data.iloc[i+1]['high'] and 
                            current['high'] > data.iloc[i+2]['high']):
                            
                            # Check for multiple touches
                            touches = 0
                            for j in range(max(0, i-20), min(len(data), i+20)):
                                if abs(data.iloc[j]['high'] - current['high']) < (current['high'] * 0.001):
                                    touches += 1
                            
                            if touches >= 2:
                                liquidity_zones.append({
                                    'type': 'High',
                                    'level': current['high'],
                                    'time': current.name,
                                    'touches': touches,
                                    'strength': 'Strong' if touches >= 3 else 'Medium'
                                })
                        
                        # Swing Low
                        elif (current['low'] < data.iloc[i-1]['low'] and 
                              current['low'] < data.iloc[i-2]['low'] and
                              current['low'] < data.iloc[i+1]['low'] and 
                              current['low'] < data.iloc[i+2]['low']):
                            
                            # Check for multiple touches
                            touches = 0
                            for j in range(max(0, i-20), min(len(data), i+20)):
                                if abs(data.iloc[j]['low'] - current['low']) < (current['low'] * 0.001):
                                    touches += 1
                            
                            if touches >= 2:
                                liquidity_zones.append({
                                    'type': 'Low',
                                    'level': current['low'],
                                    'time': current.name,
                                    'touches': touches,
                                    'strength': 'Strong' if touches >= 3 else 'Medium'
                                })
                    
                    # 3. Market Structure Analysis
                    current_price = data['close'].iloc[-1]
                    
                    # Trend structure
                    recent_highs = [lz['level'] for lz in liquidity_zones if lz['type'] == 'High'][-3:]
                    recent_lows = [lz['level'] for lz in liquidity_zones if lz['type'] == 'Low'][-3:]
                    
                    if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                        if recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]:
                            market_structure = "Higher Highs & Higher Lows - صعودی"
                        elif recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]:
                            market_structure = "Lower Highs & Lower Lows - نزولی"
                        else:
                            market_structure = "Mixed Structure - مخلوط"
                    else:
                        market_structure = "Not enough data - داده کافی نیست"
                    
                    # 4. Institutional Activity Score
                    institutional_score = 0
                    
                    # Volume analysis
                    recent_volume = data['Volume_Ratio'].tail(10).mean()
                    if recent_volume > 1.5:
                        institutional_score += 3
                    elif recent_volume > 1.2:
                        institutional_score += 1
                    
                    # Order blocks
                    recent_obs = [ob for ob in order_blocks if (datetime.now() - ob['time']).days < 7]
                    institutional_score += len(recent_obs) * 2
                    
                    # Price rejection
                    recent_wicks = data.tail(5)
                    long_wicks = 0
                    for _, candle in recent_wicks.iterrows():
                        upper_wick = candle['high'] - max(candle['open'], candle['close'])
                        lower_wick = min(candle['open'], candle['close']) - candle['low']
                        total_range = candle['high'] - candle['low']
                        
                        if upper_wick > total_range * 0.4 or lower_wick > total_range * 0.4:
                            long_wicks += 1
                    
                    institutional_score += long_wicks
                    
                    # Final score classification
                    if institutional_score >= 10:
                        activity_level = "بسیار بالا"
                    elif institutional_score >= 7:
                        activity_level = "بالا"
                    elif institutional_score >= 4:
                        activity_level = "متوسط"
                    else:
                        activity_level = "پایین"
                    
                    # Store results
                    st.session_state.smart_money_results = {
                        'symbol': smart_symbol,
                        'timeframe': smart_timeframe,
                        'data': data,
                        'order_blocks': order_blocks,
                        'liquidity_zones': liquidity_zones,
                        'market_structure': market_structure,
                        'institutional_score': institutional_score,
                        'activity_level': activity_level,
                        'smart_analysis': smart_result,
                        'general_analysis': general_analysis,
                        'live_price': live_price,
                        'settings': {
                            'volume_threshold': volume_threshold,
                            'body_ratio_threshold': body_ratio_threshold,
                            'lookback_period': lookback_period
                        },
                        'timestamp': datetime.now()
                    }
                    
                    st.success("✅ تحلیل Smart Money کامل شد!")
                
                else:
                    st.error(f"❌ خطا در دریافت داده {smart_symbol}")
        
        # Display Smart Money Results
        if 'smart_money_results' in st.session_state:
            results = st.session_state.smart_money_results
            
            st.markdown("---")
            st.markdown(f"### 🧠 نتایج Smart Money - {results['symbol']} ({results['timeframe']})")
            
            # Summary Dashboard
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎯 سیگنال Smart Money</h4>
                    <h3>{results['smart_analysis']['smart_signal'][:20]}</h3>
                    <p>فعالیت: {results['activity_level']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("📊 Order Blocks", len(results['order_blocks']), f"قوی: {len([ob for ob in results['order_blocks'] if ob['strength'] == 'Strong'])}")
            
            with col3:
                st.metric("💧 Liquidity Zones", len(results['liquidity_zones']), f"قوی: {len([lz for lz in results['liquidity_zones'] if lz['strength'] == 'Strong'])}")
            
            with col4:
                st.metric("🏛️ امتیاز نهادی", results['institutional_score'], f"سطح: {results['activity_level']}")
            
            with col5:
                if results['live_price']:
                    st.metric("💰 قیمت زنده", f"{results['live_price']['bid']:.5f}", f"اسپرد: {results['live_price']['spread']:.1f}")
            
            # Market Structure
            st.markdown("### 🏗️ ساختار بازار")
            st.info(f"📊 **Market Structure:** {results['market_structure']}")
            
            # Order Flow Analysis
            st.markdown("### 🔄 تحلیل Order Flow")
            
            flow_col1, flow_col2, flow_col3 = st.columns(3)
            
            with flow_col1:
                st.markdown("#### 💰 حجم خرید vs فروش")
                
                bullish_volume = results['smart_analysis']['bullish_volume']
                bearish_volume = results['smart_analysis']['bearish_volume']
                total_volume = bullish_volume + bearish_volume
                
                if total_volume > 0:
                    bull_ratio = (bullish_volume / total_volume) * 100
                    bear_ratio = (bearish_volume / total_volume) * 100
                    
                    st.write(f"🟢 **خرید:** {bull_ratio:.1f}% ({bullish_volume:,.0f})")
                    st.write(f"🔴 **فروش:** {bear_ratio:.1f}% ({bearish_volume:,.0f})")
                    
                    # Progress bars
                    st.progress(bull_ratio / 100)
                    st.caption("نسبت خرید")
                    
                    st.progress(bear_ratio / 100)
                    st.caption("نسبت فروش")
            
            with flow_col2:
                st.markdown("#### 📈 تحلیل حجم")
                
                volume_spike_ratio = results['smart_analysis']['volume_spike_ratio']
                
                st.write(f"**نسبت حجم:** {volume_spike_ratio:.2f}x")
                
                if volume_spike_ratio > 2:
                    st.success("🔥 حجم فوق‌العاده بالا - فعالیت نهادی قوی")
                elif volume_spike_ratio > 1.5:
                    st.warning("⚡ حجم بالاتر از معمول - فعالیت نهادی")
                else:
                    st.info("📊 حجم عادی")
                
                st.write(f"**وضعیت:** {results['smart_analysis']['order_flow']}")
            
            with flow_col3:
                st.markdown("#### 🎯 سیگنال‌های تشخیص داده شده")
                
                if results['smart_analysis']['signals']:
                    for i, signal in enumerate(results['smart_analysis']['signals'][:5], 1):
                        st.write(f"{i}. {signal}")
                else:
                    st.write("🔍 هیچ سیگنال خاصی تشخیص داده نشد")
            
            # Order Blocks Table
            if results['order_blocks']:
                st.markdown("### 📊 Order Blocks")
                
                ob_data = []
                for ob in results['order_blocks'][-10:]:  # Show last 10
                    ob_data.append({
                        'نوع': f"{'🟢' if ob['type'] == 'Bullish' else '🔴'} {ob['type']}",
                        'زمان': ob['time'].strftime('%Y-%m-%d %H:%M'),
                        'قیمت بالا': f"{ob['high']:.5f}",
                        'قیمت پایین': f"{ob['low']:.5f}",
                        'نسبت حجم': f"{ob['volume_ratio']:.2f}x",
                        'نسبت بدنه': f"{ob['body_ratio']:.2%}",
                        'قدرت': ob['strength']
                    })
                
                st.dataframe(pd.DataFrame(ob_data), use_container_width=True)
            else:
                st.info("🔍 Order Block یافت نشد با تنظیمات فعلی")
            
            # Liquidity Zones Table
            if results['liquidity_zones']:
                st.markdown("### 💧 Liquidity Zones")
                
                lz_data = []
                current_price = results['data']['close'].iloc[-1]
                
                for lz in results['liquidity_zones'][-10:]:  # Show last 10
                    distance = ((lz['level'] / current_price) - 1) * 100
                    
                    lz_data.append({
                        'نوع': f"{'🔴' if lz['type'] == 'High' else '🟢'} {lz['type']}",
                        'سطح': f"{lz['level']:.5f}",
                        'فاصله از قیمت': f"{distance:+.2f}%",
                        'تعداد تماس': lz['touches'],
                        'قدرت': lz['strength'],
                        'زمان': lz['time'].strftime('%Y-%m-%d %H:%M')
                    })
                
                st.dataframe(pd.DataFrame(lz_data), use_container_width=True)
            else:
                st.info("🔍 Liquidity Zone یافت نشد با تنظیمات فعلی")
            
            # Smart Money Chart
            st.markdown("### 📈 نمودار Smart Money")
            
            chart_data = results['data'].tail(100)
            
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=chart_data.index,
                open=chart_data['open'],
                high=chart_data['high'],
                low=chart_data['low'],
                close=chart_data['close'],
                name=results['symbol'],
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
            
            # Add EMA for trend
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data['EMA_20'],
                mode='lines',
                name='EMA 20',
                line=dict(color='blue', width=1)
            ))
            
            # Order Blocks
            for ob in results['order_blocks'][-5:]:  # Show last 5
                if ob['time'] in chart_data.index or ob['time'] >= chart_data.index[0]:
                    color = 'rgba(0, 255, 0, 0.3)' if ob['type'] == 'Bullish' else 'rgba(255, 0, 0, 0.3)'
                    
                    fig.add_shape(
                        type="rect",
                        x0=ob['time'],
                        x1=chart_data.index[-1],
                        y0=ob['low'],
                        y1=ob['high'],
                        fillcolor=color,
                        line=dict(width=1, color=color.replace('0.3', '0.7')),
                        layer="below"
                    )
                    
                    # Add annotation
                    fig.add_annotation(
                        x=ob['time'],
                        y=ob['high'] if ob['type'] == 'Bearish' else ob['low'],
                        text=f"{ob['type']} OB ({ob['strength']})",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=color.replace('0.3', '1'),
                        bgcolor=color,
                        bordercolor=color.replace('0.3', '1'),
                        borderwidth=1
                    )
            
            # Liquidity Zones
            for lz in results['liquidity_zones'][-5:]:  # Show last 5
                color = 'blue' if lz['type'] == 'High' else 'purple'
                line_style = 'solid' if lz['strength'] == 'Strong' else 'dash'
                
                fig.add_hline(
                    y=lz['level'],
                    line_dash=line_style,
                    line_color=color,
                    line_width=2,
                    annotation_text=f"LZ {lz['type']} ({lz['touches']})",
                    annotation_position="top right"
                )
            
            fig.update_layout(
                title=f"Smart Money Analysis - {results['symbol']} ({results['timeframe']})",
                height=600,
                showlegend=True,
                xaxis_title="Time",
                yaxis_title="Price"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume Analysis for Smart Money
            st.markdown("### 📊 تحلیل حجم Smart Money")
            
            fig_vol = go.Figure()
            
            # Volume bars with smart money highlighting
            colors = []
            for _, row in chart_data.iterrows():
                if row['Volume_Ratio'] > volume_threshold and row['Body_Ratio'] > body_ratio_threshold:
                    colors.append('yellow')  # Smart money activity
                elif row['Bullish']:
                    colors.append('green')
                else:
                    colors.append('red')
            
            fig_vol.add_trace(go.Bar(
                x=chart_data.index,
                y=chart_data['tick_volume'],
                marker_color=colors,
                opacity=0.7,
                name='Volume'
            ))
            
            # Volume MA
            fig_vol.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data['Volume_MA'],
                mode='lines',
                name='Volume MA',
                line=dict(color='orange', width=2)
            ))
            
            # Threshold line
            threshold_line = chart_data['Volume_MA'] * volume_threshold
            fig_vol.add_trace(go.Scatter(
                x=chart_data.index,
                y=threshold_line,
                mode='lines',
                name=f'Smart Money Threshold ({volume_threshold}x)',
                line=dict(color='red', width=1, dash='dash')
            ))
            
            fig_vol.update_layout(
                title="Volume Analysis - Smart Money Detection",
                height=300,
                yaxis_title="Volume",
                showlegend=True
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Smart Money Trading Strategy
            st.markdown("### 🎯 استراتژی معاملاتی Smart Money")
            
            strategy_col1, strategy_col2 = st.columns(2)
            
            with strategy_col1:
                st.markdown("#### 📈 سیگنال‌های خرید Smart Money")
                
                buy_signals = [
                    "✅ Order Block صعودی فعال شود",
                    "✅ قیمت از Liquidity Low برگردد",
                    "✅ حجم بالا همراه با کندل صعودی قوی",
                    "✅ رد فروش با Long Lower Wick",
                    "✅ Higher Lows در Market Structure"
                ]
                
                for signal in buy_signals:
                    st.write(signal)
                
                # Current buy conditions
                current_conditions = []
                current_price = results['data']['close'].iloc[-1]
                
                # Check Order Blocks
                active_bullish_obs = [ob for ob in results['order_blocks'] 
                                    if ob['type'] == 'Bullish' and 
                                    abs(current_price - ob['low']) / current_price < 0.005]
                
                if active_bullish_obs:
                    current_conditions.append("🟢 نزدیک Bullish Order Block")
                
                # Check Liquidity Zones
                active_low_zones = [lz for lz in results['liquidity_zones']
                                  if lz['type'] == 'Low' and
                                  abs(current_price - lz['level']) / current_price < 0.005]
                
                if active_low_zones:
                    current_conditions.append("🟢 نزدیک Liquidity Low")
                
                # Check volume
                recent_volume = results['data']['Volume_Ratio'].iloc[-1]
                if recent_volume > volume_threshold:
                    current_conditions.append("🟢 حجم بالا فعال")
                
                if current_conditions:
                    st.markdown("#### 🎯 شرایط فعلی خرید:")
                    for condition in current_conditions:
                        st.write(condition)
                else:
                    st.info("⏳ منتظر شرایط خرید مناسب")
            
            with strategy_col2:
                st.markdown("#### 📉 سیگنال‌های فروش Smart Money")
                
                sell_signals = [
                    "❌ Order Block نزولی فعال شود",
                    "❌ قیمت از Liquidity High برگردد", 
                    "❌ حجم بالا همراه با کندل نزولی قوی",
                    "❌ رد خرید با Long Upper Wick",
                    "❌ Lower Highs در Market Structure"
                ]
                
                for signal in sell_signals:
                    st.write(signal)
                
                # Current sell conditions
                current_conditions = []
                
                # Check Order Blocks
                active_bearish_obs = [ob for ob in results['order_blocks'] 
                                    if ob['type'] == 'Bearish' and 
                                    abs(current_price - ob['high']) / current_price < 0.005]
                
                if active_bearish_obs:
                    current_conditions.append("🔴 نزدیک Bearish Order Block")
                
                # Check Liquidity Zones
                active_high_zones = [lz for lz in results['liquidity_zones']
                                   if lz['type'] == 'High' and
                                   abs(current_price - lz['level']) / current_price < 0.005]
                
                if active_high_zones:
                    current_conditions.append("🔴 نزدیک Liquidity High")
                
                if current_conditions:
                    st.markdown("#### 🎯 شرایط فعلی فروش:")
                    for condition in current_conditions:
                        st.write(condition)
                else:
                    st.info("⏳ منتظر شرایط فروش مناسب")
            
            # Risk Management for Smart Money
            st.markdown("### ⚖️ مدیریت ریسک Smart Money")
            
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            
            with risk_col1:
                st.markdown("#### 🛡️ Stop Loss")
                st.write("• **برای خرید:** زیر آخرین Order Block صعودی")
                st.write("• **برای فروش:** بالای آخرین Order Block نزولی")
                st.write("• **فاصله معمول:** 20-30 پیپ از Order Block")
            
            with risk_col2:
                st.markdown("#### 🎯 Take Profit")
                st.write("• **هدف اول:** نزدیک‌ترین Liquidity Zone")
                st.write("• **هدف دوم:** Liquidity Zone بعدی")
                st.write("• **نسبت ریسک/ریوارد:** حداقل 1:2")
            
            with risk_col3:
                st.markdown("#### 💰 Position Sizing")
                activity_level = results['activity_level']
                if activity_level == "بسیار بالا":
                    st.write("• **سایز پوزیشن:** 3-5% سرمایه")
                    st.write("• **اعتماد:** بالا")
                elif activity_level == "بالا":
                    st.write("• **سایز پوزیشن:** 2-3% سرمایه")
                    st.write("• **اعتماد:** متوسط تا بالا")
                else:
                    st.write("• **سایز پوزیشن:** 1-2% سرمایه")
                    st.write("• **اعتماد:** پایین")
        
        else:
            st.info("🧠 برای مشاهده تحلیل Smart Money، روی دکمه 'تحلیل Smart Money کامل' کلیک کنید.")
    
    with tab4:
        st.header("📈 نمودارهای زنده")
        
        st.markdown("### 📊 Real-time Price Charts")
        
        # Chart Configuration
        chart_col1, chart_col2, chart_col3, chart_col4 = st.columns(4)
        
        with chart_col1:
            live_category = st.selectbox(
                "📂 دسته:",
                ['forex_major', 'gold_metals', 'forex_minor'],
                key="live_category"
            )
        
        with chart_col2:
            live_symbols = symbol_categories.get(live_category, ['EURUSD'])
            if live_symbols:
                live_symbol = st.selectbox("💱 نماد:", live_symbols, key="live_symbol_charts")  # تغییر کلید
            else:
                live_symbol = st.text_input("💱 نماد:", value="EURUSD", key="live_symbol_input")
        
        with chart_col3:
            live_timeframe = st.selectbox("⏰ تایم فریم:", ['M1', 'M5', 'M15', 'H1'], index=2, key="live_timeframe_charts")
        
        with chart_col4:
            auto_refresh = st.checkbox("🔄 به‌روزرسانی خودکار", value=False)
        
        # Chart Settings
        st.markdown("### ⚙️ تنظیمات نمودار")
        
        settings_col1, settings_col2, settings_col3 = st.columns(3)
        
        with settings_col1:
            chart_style = st.selectbox("🎨 سبک نمودار:", ['Candlestick', 'OHLC', 'Line'], key="chart_style")
            show_volume_chart = st.checkbox("📊 نمایش حجم", value=True)
        
        with settings_col2:
            chart_indicators = st.multiselect(
                "📈 اندیکاتورها:",
                ['EMA 20', 'EMA 50', 'SMA 20', 'SMA 50', 'Bollinger Bands', 'VWAP'],
                default=['EMA 20', 'EMA 50'],
                key="live_chart_indicators"
            )
        
        with settings_col3:
            chart_candles = st.slider("🕯️ تعداد کندل:", 50, 500, 100, key="live_chart_candles")
            show_alerts = st.checkbox("🔔 نمایش هشدارها", value=True)
        
        # Live Chart Container
        live_chart_container = st.container()
        
        # Start Live Chart
        if st.button("📈 شروع نمودار زنده", type="primary") or auto_refresh:
            
            # Auto refresh mechanism
            if auto_refresh:
                refresh_placeholder = st.empty()
                for countdown in range(30, 0, -1):
                    refresh_placeholder.write(f"🔄 به‌روزرسانی در {countdown} ثانیه...")
                    time.sleep(1)
                refresh_placeholder.empty()
            
            with live_chart_container:
                with st.spinner(f"📊 بارگذاری نمودار زنده {live_symbol}..."):
                    # Get live data
                    live_data, success = get_market_data(live_symbol, live_timeframe, chart_candles)
                    
                    if success:
                        # Calculate indicators
                        live_data = calculate_all_indicators(live_data)
                        
                        # Get live price
                        current_tick = get_live_price(live_symbol)
                        
                        # Create main chart
                        fig = go.Figure()
                        
                        # Price chart based on style
                        if chart_style == 'Candlestick':
                            fig.add_trace(go.Candlestick(
                                x=live_data.index,
                                open=live_data['open'],
                                high=live_data['high'],
                                low=live_data['low'],
                                close=live_data['close'],
                                name=live_symbol,
                                increasing_line_color='green',
                                decreasing_line_color='red'
                            ))
                        elif chart_style == 'OHLC':
                            fig.add_trace(go.Ohlc(
                                x=live_data.index,
                                open=live_data['open'],
                                high=live_data['high'],
                                low=live_data['low'],
                                close=live_data['close'],
                                name=live_symbol
                            ))
                        else:  # Line chart
                            fig.add_trace(go.Scatter(
                                x=live_data.index,
                                y=live_data['close'],
                                mode='lines',
                                name=f'{live_symbol} Close',
                                line=dict(color='blue', width=2)
                            ))
                        
                        # Add indicators
                        for indicator in chart_indicators:
                            if indicator == 'EMA 20':
                                fig.add_trace(go.Scatter(
                                    x=live_data.index, y=live_data['EMA_20'],
                                    mode='lines', name='EMA 20',
                                    line=dict(color='blue', width=1.5)
                                ))
                            elif indicator == 'EMA 50':
                                fig.add_trace(go.Scatter(
                                    x=live_data.index, y=live_data['EMA_50'],
                                    mode='lines', name='EMA 50',
                                    line=dict(color='orange', width=1.5)
                                ))
                            elif indicator == 'SMA 20':
                                fig.add_trace(go.Scatter(
                                    x=live_data.index, y=live_data['SMA_20'],
                                    mode='lines', name='SMA 20',
                                    line=dict(color='purple', width=1)
                                ))
                            elif indicator == 'SMA 50':
                                fig.add_trace(go.Scatter(
                                    x=live_data.index, y=live_data['SMA_50'],
                                    mode='lines', name='SMA 50',
                                    line=dict(color='brown', width=1)
                                ))
                            elif indicator == 'Bollinger Bands':
                                fig.add_trace(go.Scatter(
                                    x=live_data.index, y=live_data['BB_Upper'],
                                    mode='lines', name='BB Upper',
                                    line=dict(color='gray', width=1, dash='dash')
                                ))
                                fig.add_trace(go.Scatter(
                                    x=live_data.index, y=live_data['BB_Lower'],
                                    mode='lines', name='BB Lower',
                                    line=dict(color='gray', width=1, dash='dash'),
                                    fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
                                ))
                            elif indicator == 'VWAP':
                                # Simple VWAP calculation
                                vwap = (live_data['close'] * live_data['tick_volume']).cumsum() / live_data['tick_volume'].cumsum()
                                fig.add_trace(go.Scatter(
                                    x=live_data.index, y=vwap,
                                    mode='lines', name='VWAP',
                                    line=dict(color='yellow', width=2)
                                ))
                        
                        # Add current price line
                        if current_tick:
                            current_price = current_tick['bid']
                            fig.add_hline(
                                y=current_price,
                                line_dash="solid",
                                line_color="red",
                                line_width=2,
                                annotation_text=f"Live: {current_price:.5f}",
                                annotation_position="right"
                            )
                        
                        # Chart layout
                        fig.update_layout(
                            title=f"📈 {live_symbol} - {live_timeframe} | Live Price: {current_price:.5f}" if current_tick else f"📈 {live_symbol} - {live_timeframe}",
                            height=500,
                            showlegend=True,
                            xaxis_title="Time",
                            yaxis_title="Price"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Volume Chart
                        if show_volume_chart:
                            fig_vol = go.Figure()
                            
                            # Volume bars
                            colors = ['green' if row['Bullish'] else 'red' for _, row in live_data.iterrows()]
                            fig_vol.add_trace(go.Bar(
                                x=live_data.index,
                                y=live_data['tick_volume'],
                                marker_color=colors,
                                opacity=0.7,
                                name='Volume'
                            ))
                            
                            # Volume MA
                            fig_vol.add_trace(go.Scatter(
                                x=live_data.index,
                                y=live_data['Volume_MA'],
                                mode='lines',
                                name='Volume MA',
                                line=dict(color='blue', width=2)
                            ))
                            
                            fig_vol.update_layout(
                                title="📊 Volume Analysis",
                                height=200,
                                showlegend=True,
                                yaxis_title="Volume"
                            )
                            
                            st.plotly_chart(fig_vol, use_container_width=True)
                        
                        # Live Price Info
                        if current_tick:
                            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                            
                            with info_col1:
                                st.metric("💰 Bid", f"{current_tick['bid']:.5f}")
                            
                            with info_col2:
                                st.metric("💰 Ask", f"{current_tick['ask']:.5f}")
                            
                            with info_col3:
                                st.metric("📊 Spread", f"{current_tick['spread']:.1f} pips")
                            
                            with info_col4:
                                st.metric("🕒 Update Time", current_tick['time'].strftime('%H:%M:%S'))
                        
                        # Technical Summary
                        st.markdown("### 📊 خلاصه تکنیکال")
                        
                        current_price = live_data['close'].iloc[-1]
                        ema_20 = live_data['EMA_20'].iloc[-1]
                        ema_50 = live_data['EMA_50'].iloc[-1]
                        rsi = live_data['RSI'].iloc[-1]
                        
                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                        
                        with summary_col1:
                            # Trend
                            if current_price > ema_20 > ema_50:
                                trend_status = "📈 صعودی"
                                trend_color = "green"
                            elif current_price < ema_20 < ema_50:
                                trend_status = "📉 نزولی"
                                trend_color = "red"
                            else:
                                trend_status = "➡️ رنج"
                                trend_color = "gray"
                            
                            st.markdown(f"""
                            <div style="border: 2px solid {trend_color}; padding: 1rem; border-radius: 5px; text-align: center;">
                                <h4>روند</h4>
                                <h3 style="color: {trend_color};">{trend_status}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with summary_col2:
                            # RSI Status
                            if rsi > 70:
                                rsi_status = "خرید بیش از حد"
                                rsi_color = "red"
                            elif rsi < 30:
                                rsi_status = "فروش بیش از حد"
                                rsi_color = "green"
                            else:
                                rsi_status = "عادی"
                                rsi_color = "blue"
                            
                            st.markdown(f"""
                            <div style="border: 2px solid {rsi_color}; padding: 1rem; border-radius: 5px; text-align: center;">
                                <h4>RSI</h4>
                                <h3 style="color: {rsi_color};">{rsi:.1f}</h3>
                                <p>{rsi_status}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with summary_col3:
                            # Volume Status
                            volume_ratio = live_data['Volume_Ratio'].iloc[-1]
                            if volume_ratio > 1.5:
                                volume_status = "حجم بالا"
                                volume_color = "orange"
                            elif volume_ratio < 0.8:
                                volume_status = "حجم پایین"
                                volume_color = "gray"
                            else:
                                volume_status = "حجم عادی"
                                volume_color = "blue"
                            
                            st.markdown(f"""
                            <div style="border: 2px solid {volume_color}; padding: 1rem; border-radius: 5px; text-align: center;">
                                <h4>Volume</h4>
                                <h3 style="color: {volume_color};">{volume_ratio:.2f}x</h3>
                                <p>{volume_status}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Quick Analysis
                        quick_analysis = advanced_market_analysis(live_data, live_symbol)
                        
                        st.markdown("### 🎯 تحلیل سریع")
                        st.info(f"**سیگنال:** {quick_analysis['overall_signal']} | **اعتماد:** {quick_analysis['confidence']}% | **ریسک:** {quick_analysis['risk_level']}")
                        
                        if auto_refresh:
                            st.rerun()
                    
                    else:
                        st.error(f"❌ خطا در دریافت داده زنده {live_symbol}")
        
        # Chart Controls
        st.markdown("---")
        st.markdown("### 🎮 کنترل‌های نمودار")
        
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            if st.button("⏸️ توقف به‌روزرسانی"):
                auto_refresh = False
                st.info("🛑 به‌روزرسانی خودکار متوقف شد")
        
        with control_col2:
            if st.button("🔄 به‌روزرسانی دستی"):
                st.rerun()
        
        with control_col3:
            if st.button("💾 ذخیره نمودار"):
                if 'live_data' in locals() and live_data is not None:
                    # Save chart data to session
                    st.session_state.saved_charts = st.session_state.get('saved_charts', [])
                    st.session_state.saved_charts.append({
                        'symbol': live_symbol,
                        'timeframe': live_timeframe,
                        'data': live_data.tail(50),  # Save last 50 candles
                        'saved_time': datetime.now()
                    })
                    st.success("💾 نمودار ذخیره شد!")
                else:
                    st.warning("⚠️ ابتدا نمودار را بارگذاری کنید")
        
        # Saved Charts
        if 'saved_charts' in st.session_state and st.session_state.saved_charts:
            st.markdown("### 💾 نمودارهای ذخیره شده")
            
            for i, saved_chart in enumerate(st.session_state.saved_charts[-5:]):  # Show last 5
                with st.expander(f"📊 {saved_chart['symbol']} - {saved_chart['timeframe']} | {saved_chart['saved_time'].strftime('%H:%M:%S')}"):
                    
                    # Quick chart
                    fig_saved = go.Figure()
                    
                    chart_data = saved_chart['data']
                    fig_saved.add_trace(go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['open'],
                        high=chart_data['high'],
                        low=chart_data['low'],
                        close=chart_data['close'],
                        name=saved_chart['symbol']
                    ))
                    
                    fig_saved.update_layout(
                        title=f"Saved: {saved_chart['symbol']} - {saved_chart['timeframe']}",
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_saved, use_container_width=True)
                    
                    # Chart stats
                    price_change = ((chart_data['close'].iloc[-1] / chart_data['close'].iloc[0]) - 1) * 100
                    high_price = chart_data['high'].max()
                    low_price = chart_data['low'].min()
                    
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    
                    with stat_col1:
                        st.metric("📈 تغییر", f"{price_change:+.2f}%")
                    
                    with stat_col2:
                        st.metric("🔺 بالاترین", f"{high_price:.5f}")
                    
                    with stat_col3:
                        st.metric("🔻 پایین‌ترین", f"{low_price:.5f}")
    
    with tab5:
        st.header("🔔 هشدارها و اسکن بازار")
        
        # Market Scanner
        st.markdown("### 📡 اسکن بازار")
        
        scanner_col1, scanner_col2, scanner_col3 = st.columns(3)
        
        with scanner_col1:
            scan_categories = st.multiselect(
                "📂 دسته‌های اسکن:",
                ['forex_major', 'gold_metals', 'forex_minor', 'commodities'],
                default=['forex_major', 'gold_metals'],
                format_func=lambda x: {
                    'forex_major': '💱 فارکس اصلی',
                    'gold_metals': '🥇 طلا و فلزات',
                    'forex_minor': '💸 فارکس فرعی',
                    'commodities': '🛢️ کالاها'
                }[x]
            )
        
        with scanner_col2:
            scan_timeframe = st.selectbox("⏰ تایم فریم اسکن:", ['H1', 'H4', 'D1'], index=0)
            min_confidence = st.slider("🎯 حداقل اعتماد:", 50, 90, 70, 5)
        
        with scanner_col3:
            scan_type = st.selectbox(
                "🔍 نوع اسکن:",
                ['all', 'strong_signals', 'smart_money', 'breakouts'],
                format_func=lambda x: {
                    'all': 'همه سیگنال‌ها',
                    'strong_signals': 'سیگنال‌های قوی',
                    'smart_money': 'Smart Money',
                    'breakouts': 'شکست‌ها'
                }[x]
            )
        
        # Run Market Scan
        if st.button("🚀 شروع اسکن بازار", type="primary"):
            
            # Collect all symbols to scan
            symbols_to_scan = []
            for category in scan_categories:
                symbols_to_scan.extend(symbol_categories.get(category, []))
            
            # Remove duplicates
            symbols_to_scan = list(set(symbols_to_scan))
            
            if symbols_to_scan:
                st.info(f"🔍 در حال اسکن {len(symbols_to_scan)} نماد...")
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                scan_results = []
                
                for i, symbol in enumerate(symbols_to_scan):
                    status_text.text(f"اسکن {symbol}... ({i+1}/{len(symbols_to_scan)})")
                    progress_bar.progress((i + 1) / len(symbols_to_scan))
                    
                    try:
                        # Get data
                        data, success = get_market_data(symbol, scan_timeframe, 200)
                        
                        if success:
                            # Analyze
                            data = calculate_all_indicators(data)
                            analysis = advanced_market_analysis(data, symbol)
                            smart_analysis = smart_money_analysis(data, symbol)
                            live_price = get_live_price(symbol)
                            
                            # Filter based on scan type and confidence
                            include_result = False
                            
                            if scan_type == 'all':
                                include_result = analysis['confidence'] >= min_confidence
                            elif scan_type == 'strong_signals':
                                include_result = (analysis['confidence'] >= min_confidence and 
                                                analysis['action'] in ['STRONG_BUY', 'STRONG_SELL'])
                            elif scan_type == 'smart_money':
                                include_result = (analysis['confidence'] >= min_confidence and
                                                'Smart Money' in smart_analysis['smart_signal'])
                            elif scan_type == 'breakouts':
                                # Check for breakout conditions
                                recent_high = data['high'].tail(20).max()
                                recent_low = data['low'].tail(20).min()
                                current_price = data['close'].iloc[-1]
                                
                                breakout_up = current_price > recent_high * 1.001
                                breakout_down = current_price < recent_low * 0.999
                                
                                include_result = ((breakout_up or breakout_down) and 
                                                analysis['confidence'] >= min_confidence)
                            
                            if include_result:
                                scan_results.append({
                                    'symbol': symbol,
                                    'signal': analysis['overall_signal'],
                                    'confidence': analysis['confidence'],
                                    'action': analysis['action'],
                                    'trend': analysis['trend'],
                                    'risk': analysis['risk_level'],
                                    'smart_money': smart_analysis['smart_signal'],
                                    'price': live_price['bid'] if live_price else data['close'].iloc[-1],
                                    'rsi': analysis['indicators']['RSI'],
                                    'volume_ratio': analysis['indicators']['Volume_Ratio'],
                                    'analysis': analysis,
                                    'smart_analysis': smart_analysis,
                                    'scan_time': datetime.now()
                                })
                    
                    except Exception as e:
                        continue
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Store results
                st.session_state.scan_results = scan_results
                
                st.success(f"✅ اسکن تکمیل شد! {len(scan_results)} نماد یافت شد.")
            
            else:
                st.warning("⚠️ لطفاً حداقل یک دسته برای اسکن انتخاب کنید.")
        
        # Display Scan Results
        if 'scan_results' in st.session_state and st.session_state.scan_results:
            st.markdown("---")
            st.markdown(f"### 📊 نتایج اسکن ({len(st.session_state.scan_results)} نماد)")
            
            # Sort results by confidence
            sorted_results = sorted(st.session_state.scan_results, key=lambda x: x['confidence'], reverse=True)
            
            # Summary Stats
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                buy_signals = len([r for r in sorted_results if 'خرید' in r['signal']])
                st.metric("📈 سیگنال خرید", buy_signals)
            
            with summary_col2:
                sell_signals = len([r for r in sorted_results if 'فروش' in r['signal']])
                st.metric("📉 سیگنال فروش", sell_signals)
            
            with summary_col3:
                strong_signals = len([r for r in sorted_results if r['action'] in ['STRONG_BUY', 'STRONG_SELL']])
                st.metric("💪 سیگنال قوی", strong_signals)
            
            with summary_col4:
                avg_confidence = np.mean([r['confidence'] for r in sorted_results])
                st.metric("🎯 میانگین اعتماد", f"{avg_confidence:.1f}%")
            
            # Results Table
            st.markdown("### 📋 جدول نتایج")
            
            results_data = []
            for result in sorted_results:
                # Signal emoji
                if 'خرید' in result['signal']:
                    signal_emoji = "🟢"
                elif 'فروش' in result['signal']:
                    signal_emoji = "🔴"
                else:
                    signal_emoji = "🟡"
                
                # Risk emoji
                risk_emoji = {"بالا": "🔴", "متوسط": "🟡", "پایین": "🟢"}.get(result['risk'], "❓")
                
                results_data.append({
                    'نماد': result['symbol'],
                    'سیگنال': f"{signal_emoji} {result['signal']}",
                    'اعتماد': f"{result['confidence']:.1f}%",
                    'روند': result['trend'],
                    'ریسک': f"{risk_emoji} {result['risk']}",
                    'قیمت': f"{result['price']:.5f}",
                    'RSI': f"{result['rsi']:.1f}",
                    'حجم': f"{result['volume_ratio']:.2f}x",
                    'Smart Money': result['smart_money'][:15] + "..." if len(result['smart_money']) > 15 else result['smart_money']
                })
            
            # Create DataFrame and display
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Detailed Results
            st.markdown("### 🔍 نتایج تفصیلی")
            
            # Show top 5 results in detail
            for i, result in enumerate(sorted_results[:5]):
                with st.expander(f"📊 {result['symbol']} - {result['signal']} ({result['confidence']:.1f}%)"):
                    
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    
                    with detail_col1:
                        st.markdown("#### 🎯 تحلیل کلی")
                        st.write(f"**سیگنال:** {result['signal']}")
                        st.write(f"**اعتماد:** {result['confidence']:.1f}%")
                        st.write(f"**روند:** {result['trend']}")
                        st.write(f"**ریسک:** {result['risk']}")
                        st.write(f"**قیمت:** {result['price']:.5f}")
                    
                    with detail_col2:
                        st.markdown("#### 📊 اندیکاتورها")
                        indicators = result['analysis']['indicators']
                        st.write(f"**RSI:** {indicators['RSI']:.1f}")
                        st.write(f"**MACD:** {indicators['MACD']:.5f}")
                        st.write(f"**BB Position:** {indicators['BB_Position']:.1f}%")
                        st.write(f"**Volume Ratio:** {indicators['Volume_Ratio']:.2f}x")
                        st.write(f"**ATR:** {indicators['ATR']:.5f}")
                    
                    with detail_col3:
                        st.markdown("#### 🧠 Smart Money")
                        st.write(f"**سیگنال:** {result['smart_money']}")
                        st.write(f"**فعالیت نهادی:** {result['smart_analysis']['institution_activity']}")
                        st.write(f"**Order Flow:** {result['smart_analysis']['order_flow']}")
                        st.write(f"**Volume Spike:** {result['smart_analysis']['volume_spike_ratio']:.2f}x")
                    
                    # Action button
                    if st.button(f"📈 نمودار {result['symbol']}", key=f"chart_{result['symbol']}_{i}"):
                        # Store for chart display
                        st.session_state.chart_symbol = result['symbol']
                        st.session_state.chart_timeframe = scan_timeframe
                        st.info(f"📊 نمودار {result['symbol']} در تب 'نمودارهای زنده' آماده شد.")
        
        # Alert Settings
        st.markdown("---")
        st.markdown("### 🔔 تنظیمات هشدار")
        
        alert_col1, alert_col2 = st.columns(2)
        
        with alert_col1:
            st.markdown("#### ➕ افزودن هشدار جدید")
            
            with st.form("new_alert"):
                alert_symbol = st.selectbox("💱 نماد:", ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD'])
                alert_type = st.selectbox("🎯 نوع هشدار:", ['price_above', 'price_below', 'rsi_overbought', 'rsi_oversold', 'strong_signal'])
                
                if alert_type in ['price_above', 'price_below']:
                    alert_value = st.number_input("💰 قیمت هدف:", min_value=0.0, format="%.5f")
                elif alert_type in ['rsi_overbought', 'rsi_oversold']:
                    alert_value = st.number_input("📊 RSI هدف:", min_value=0.0, max_value=100.0, value=70.0 if alert_type == 'rsi_overbought' else 30.0)
                else:
                    alert_value = st.number_input("🎯 حداقل اعتماد:", min_value=50, max_value=100, value=80)
                
                alert_enabled = st.checkbox("✅ فعال", value=True)
                
                if st.form_submit_button("➕ افزودن هشدار"):
                    new_alert = {
                        'id': len(st.session_state.alerts),
                        'symbol': alert_symbol,
                        'type': alert_type,
                        'value': alert_value,
                        'enabled': alert_enabled,
                        'created': datetime.now(),
                        'triggered': False
                    }
                    
                    st.session_state.alerts.append(new_alert)
                    st.success(f"✅ هشدار برای {alert_symbol} اضافه شد!")
        
        with alert_col2:
            st.markdown("#### 📋 هشدارهای فعال")
            
            if st.session_state.alerts:
                for alert in st.session_state.alerts:
                    if alert['enabled'] and not alert['triggered']:
                        alert_status = "🟢 فعال"
                    elif alert['triggered']:
                        alert_status = "🔔 اجرا شده"
                    else:
                        alert_status = "⏸️ غیرفعال"
                    
                    type_desc = {
                        'price_above': f"قیمت بالای {alert['value']:.5f}",
                        'price_below': f"قیمت زیر {alert['value']:.5f}",
                        'rsi_overbought': f"RSI بالای {alert['value']:.1f}",
                        'rsi_oversold': f"RSI زیر {alert['value']:.1f}",
                        'strong_signal': f"سیگنال قوی (اعتماد >{alert['value']:.0f}%)"
                    }.get(alert['type'], 'نامشخص')
                    
                    st.write(f"**{alert['symbol']}:** {type_desc} - {alert_status}")
                    
                    # Quick controls
                    alert_control_col1, alert_control_col2 = st.columns(2)
                    
                    with alert_control_col1:
                        if st.button(f"❌ حذف", key=f"delete_alert_{alert['id']}"):
                            st.session_state.alerts = [a for a in st.session_state.alerts if a['id'] != alert['id']]
                            st.rerun()
                    
                    with alert_control_col2:
                        if st.button(f"{'⏸️ غیرفعال' if alert['enabled'] else '▶️ فعال'}", key=f"toggle_alert_{alert['id']}"):
                            alert['enabled'] = not alert['enabled']
                            st.rerun()
                    
                    st.markdown("---")
            
            else:
                st.info("📝 هیچ هشداری تنظیم نشده است.")
        
        # Check Alerts
        if st.button("🔍 بررسی هشدارها", type="secondary"):
            if st.session_state.alerts:
                triggered_alerts = []
                
                for alert in st.session_state.alerts:
                    if alert['enabled'] and not alert['triggered']:
                        try:
                            # Get current data
                            data, success = get_market_data(alert['symbol'], 'H1', 50)
                            
                            if success:
                                data = calculate_all_indicators(data)
                                current_price = data['close'].iloc[-1]
                                current_rsi = data['RSI'].iloc[-1]
                                
                                # Check alert conditions
                                triggered = False
                                
                                if alert['type'] == 'price_above' and current_price > alert['value']:
                                    triggered = True
                                elif alert['type'] == 'price_below' and current_price < alert['value']:
                                    triggered = True
                                elif alert['type'] == 'rsi_overbought' and current_rsi > alert['value']:
                                    triggered = True
                                elif alert['type'] == 'rsi_oversold' and current_rsi < alert['value']:
                                    triggered = True
                                elif alert['type'] == 'strong_signal':
                                    analysis = advanced_market_analysis(data, alert['symbol'])
                                    if analysis['confidence'] >= alert['value']:
                                        triggered = True
                                
                                if triggered:
                                    alert['triggered'] = True
                                    alert['triggered_time'] = datetime.now()
                                    triggered_alerts.append(alert)
                        
                        except:
                            continue
                
                if triggered_alerts:
                    st.success(f"🔔 {len(triggered_alerts)} هشدار اجرا شد!")
                    
                    for alert in triggered_alerts:
                        st.warning(f"🚨 هشدار {alert['symbol']}: شرایط تحقق یافت!")
                else:
                    st.info("✅ همه هشدارها بررسی شد. هیچ شرطی تحقق نیافت.")
            
            else:
                st.info("📝 هیچ هشداری برای بررسی وجود ندارد.")
    
    with tab6:
        st.header("🤖 ربات معاملاتی")
        
        st.markdown("""
        ### 🚀 ربات معاملاتی هوشمند
        
        این ربات بر اساس تحلیل‌های پیشرفته و Smart Money عمل می‌کند:
        
        - **🎯 تشخیص سیگنال‌های قوی** با اعتماد بالا
        - **💰 مدیریت ریسک** خودکار
        - **🧠 Smart Money** تحلیل
        - **📊 Multi-timeframe** آنالیز
        - **🔔 هشدارهای Real-time**
        """)
        
        # Bot Configuration
        st.markdown("### ⚙️ تنظیمات ربات")
        
        bot_col1, bot_col2, bot_col3 = st.columns(3)
        
        with bot_col1:
            st.markdown("#### 💱 انتخاب نمادها")
            
            bot_symbols = st.multiselect(
                "نمادها:",
                ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD', 'USDCAD', 'USDCHF', 'NZDUSD'],
                default=['EURUSD', 'GBPUSD', 'XAUUSD']
            )
            
            bot_timeframes = st.multiselect(
                "تایم فریم‌ها:",
                ['M15', 'H1', 'H4', 'D1'],
                default=['H1', 'H4']
            )
        
        with bot_col2:
            st.markdown("#### 🎯 تنظیمات سیگنال")
            
            min_confidence_bot = st.slider("حداقل اعتماد:", 60, 95, 75)
            include_smart_money = st.checkbox("🧠 Smart Money فعال", value=True)
            require_trend_confirmation = st.checkbox("📈 تأیید روند", value=True)
            max_signals_per_day = st.slider("حداکثر سیگنال روزانه:", 5, 50, 20)
        
        with bot_col3:
            st.markdown("#### ⚖️ مدیریت ریسک")
            
            risk_per_trade = st.slider("ریسک هر معامله (%):", 1.0, 5.0, 2.0, 0.5)
            stop_loss_pips = st.slider("Stop Loss (pips):", 10, 100, 30)
            take_profit_ratio = st.slider("Take Profit نسبت:", 1.5, 5.0, 2.0, 0.5)
            max_open_trades = st.slider("حداکثر معاملات باز:", 3, 10, 5)
        
        # Bot Status
        if 'bot_active' not in st.session_state:
            st.session_state.bot_active = False
        
        if 'bot_trades' not in st.session_state:
            st.session_state.bot_trades = []
        
        if 'bot_stats' not in st.session_state:
            st.session_state.bot_stats = {
                'total_signals': 0,
                'successful_trades': 0,
                'failed_trades': 0,
                'total_pips': 0,
                'win_rate': 0
            }
        
        # Bot Control
        st.markdown("---")
        st.markdown("### 🎮 کنترل ربات")
        
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            if not st.session_state.bot_active:
                if st.button("🚀 شروع ربات", type="primary", use_container_width=True):
                    if bot_symbols and bot_timeframes:
                        st.session_state.bot_active = True
                        st.session_state.bot_start_time = datetime.now()
                        st.success("✅ ربات فعال شد!")
                        st.rerun()
                    else:
                        st.error("❌ لطفاً نماد و تایم فریم انتخاب کنید!")
            else:
                if st.button("⏹️ توقف ربات", type="secondary", use_container_width=True):
                    st.session_state.bot_active = False
                    st.info("🛑 ربات متوقف شد")
                    st.rerun()
        
        with control_col2:
            if st.button("🧹 پاک کردن آمار", use_container_width=True):
                st.session_state.bot_trades = []
                st.session_state.bot_stats = {
                    'total_signals': 0,
                    'successful_trades': 0,
                    'failed_trades': 0,
                    'total_pips': 0,
                    'win_rate': 0
                }
                st.success("✅ آمار پاک شد!")
                st.rerun()
        
        with control_col3:
            if st.button("💾 ذخیره تنظیمات", use_container_width=True):
                bot_settings = {
                    'symbols': bot_symbols,
                    'timeframes': bot_timeframes,
                    'min_confidence': min_confidence_bot,
                    'include_smart_money': include_smart_money,
                    'risk_per_trade': risk_per_trade,
                    'stop_loss_pips': stop_loss_pips,
                    'take_profit_ratio': take_profit_ratio
                }
                st.session_state.bot_settings = bot_settings
                st.success("💾 تنظیمات ذخیره شد!")
        
        # Bot Status Display
        if st.session_state.bot_active:
            st.markdown("### 🔋 وضعیت ربات")
            
            status_col1, status_col2, status_col3, status_col4 = st.columns(4)
            
            with status_col1:
                st.markdown("""
                <div style="background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%); padding: 1rem; border-radius: 5px; text-align: center; color: white;">
                    <h4>🟢 ربات فعال</h4>
                    <p>در حال اسکن</p>
                </div>
                """, unsafe_allow_html=True)
            
# تعریف ستون‌ها قبل از استفاده
status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.metric("🤖 وضعیت ربات", "فعال" if st.session_state.get('bot_active', False) else "غیرفعال")

with status_col2:
    if 'bot_start_time' in st.session_state:
        active_duration = datetime.now() - st.session_state.bot_start_time
        hours = int(active_duration.total_seconds() // 3600)
        minutes = int((active_duration.total_seconds() % 3600) // 60)
        st.metric("⏰ مدت فعالیت", f"{hours}h {minutes}m")
    else:
        st.metric("⏰ مدت فعالیت", "0h 0m")

with status_col3:
    alerts_count = len(st.session_state.alerts) if st.session_state.alerts else 0
    st.metric("🔔 هشدارهای فعال", alerts_count)

with status_col4:
    session_time = datetime.now()
    st.metric("⏰ زمان جلسه", session_time.strftime('%H:%M'))

# Real-time Bot Scanning
if st.button("🔄 اسکن جدید", type="primary"):
    with st.spinner("🤖 ربات در حال اسکن..."):
        new_signals = []

        for symbol in bot_symbols:
            for timeframe in bot_timeframes:
                try:
                    # Get data
                    data, success = get_market_data(symbol, timeframe, 200)

                    if success:
                        # Analyze
                        data = calculate_all_indicators(data)
                        analysis = advanced_market_analysis(data, symbol)
                        
                        # Get current price and levels
                        current_price = data['close'].iloc[-1]
                        levels = analysis['levels']

                        # Smart Money analysis if enabled
                        smart_score = 0
                        if include_smart_money:
                            smart_analysis = smart_money_analysis(data, symbol)
                            if 'خرید' in smart_analysis['smart_signal']:
                                smart_score += 1

                        # Check if signal meets criteria
                        if analysis['confidence'] >= min_confidence:
                            # Create signal
                            signal = {
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'action': analysis['action'],
                                'confidence': analysis['confidence'],
                                'price': current_price,
                                'time': datetime.now(),
                                'smart_score': smart_score,
                                'analysis': analysis
                            }
                            
                            new_signals.append(signal)

                except Exception as e:
                    st.error(f"خطا در تحلیل {symbol}: {str(e)}")
                    continue

        # Display results
        if new_signals:
            st.success(f"✅ {len(new_signals)} سیگنال جدید پیدا شد!")
            
            # Add to bot signals
            st.session_state.bot_signals.extend(new_signals)
            st.session_state.bot_stats['total_signals'] += len(new_signals)
            
            # Show signals
            for signal in new_signals[-5:]:  # Show last 5
                if signal['action'] in ['BUY', 'STRONG_BUY']:
                    st.success(f"🟢 {signal['symbol']} ({signal['timeframe']}) - خرید - اعتماد: {signal['confidence']}%")
                elif signal['action'] in ['SELL', 'STRONG_SELL']:
                    st.error(f"🔴 {signal['symbol']} ({signal['timeframe']}) - فروش - اعتماد: {signal['confidence']}%")
        else:
            st.info("ℹ️ سیگنال جدیدی پیدا نشد")

# Moving Averages Analysis (if needed separately)
if 'analysis' in locals() and 'levels' in locals():
    st.markdown("### 📈 تحلیل میانگین‌های متحرک")
    
    ma_data = {
        'نوع': ['EMA 20', 'EMA 50', 'SMA 20', 'SMA 50'],
        'مقدار': [
            f"{levels['ema_20']:.5f}",
            f"{levels['ema_50']:.5f}",
            f"{levels['sma_20']:.5f}",
            f"{levels['sma_50']:.5f}"
        ],
        'فاصله': [
            f"{((levels['ema_20'] / current_price) - 1) * 100:+.2f}%",
            f"{((levels['ema_50'] / current_price) - 1) * 100:+.2f}%",
            f"{((levels['sma_20'] / current_price) - 1) * 100:+.2f}%",
            f"{((levels['sma_50'] / current_price) - 1) * 100:+.2f}%"
        ],
        'سیگنال': [
            "🟢 صعودی" if current_price > levels['ema_20'] else "🔴 نزولی",
            "🟢 صعودی" if current_price > levels['ema_50'] else "🔴 نزولی",
            "🟢 صعودی" if current_price > levels['sma_20'] else "🔴 نزولی",
            "🟢 صعودی" if current_price > levels['sma_50'] else "🔴 نزولی"
        ]
    }

    ma_df = pd.DataFrame(ma_data)
    st.dataframe(ma_df, use_container_width=True)

    # Detailed Chart
    st.markdown("### 📈 نمودار تفصیلی")
    chart_data = result['data'].tail(200)  # فرض: result قبلاً تعریف شده

    # Create subplots
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'Volume', 'RSI', 'MACD'),
        row_heights=[0.4, 0.2, 0.2, 0.2]  # تغییر از row_width به row_heights
    )

    # Price Chart with Moving Averages
    fig.add_trace(go.Candlestick(
        x=chart_data.index,
        open=chart_data['open'],
        high=chart_data['high'],
        low=chart_data['low'],
        close=chart_data['close'],
        name=result['symbol']
    ), row=1, col=1)

    # Moving Averages
    if 'EMA 20' in show_ma:
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['EMA_20'],
            mode='lines',
            name='EMA 20',
            line=dict(color='blue', width=1)
        ), row=1, col=1)

    if 'EMA 50' in show_ma:
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['EMA_50'],
            mode='lines',
            name='EMA 50',
            line=dict(color='orange', width=1)
        ), row=1, col=1)

    if 'SMA 20' in show_ma:
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='green', width=1, dash='dash')
        ), row=1, col=1)

    # Bollinger Bands
    if 'Bollinger Bands' in show_indicators:
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1),
            opacity=0.5
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            opacity=0.5
        ), row=1, col=1)

    # Volume Chart
    colors = ['green' if row['Bullish'] else 'red' for _, row in chart_data.iterrows()]
    fig.add_trace(go.Bar(
        x=chart_data.index,
        y=chart_data['tick_volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.7
    ), row=2, col=1)

    # Volume MA
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data['Volume_MA'],
        mode='lines',
        name='Volume MA',
        line=dict(color='orange', width=1)
    ), row=2, col=1)

    # RSI
    if 'RSI' in show_indicators:
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=3, col=1)

        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)

    # MACD
    if 'MACD' in show_indicators:
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['MACD_Signal'],
            mode='lines',
            name='Signal',
            line=dict(color='red', width=1)
        ), row=4, col=1)

        # MACD Histogram
        histogram_colors = ['green' if val > 0 else 'red' for val in chart_data['MACD_Histogram']]
        fig.add_trace(go.Bar(
            x=chart_data.index,
            y=chart_data['MACD_Histogram'],
            name='Histogram',
            marker_color=histogram_colors,
            opacity=0.6
        ), row=4, col=1)

    fig.update_layout(
        title=f"{result['symbol']} - تحلیل تفصیلی ({result['timeframe']})",
        height=800,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Signal Summary
    st.markdown("### 📋 خلاصه سیگنال‌ها")

    signals = result['analysis']['signals']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 سیگنال‌های صعودی")
        bullish_signals = []

        if 'strong_uptrend' in signals['trend']:
            bullish_signals.append("📈 روند صعودی قوی")
        elif 'uptrend' in signals['trend']:
            bullish_signals.append("📈 روند صعودی")

        if 'rsi_oversold' in signals['oscillator']:
            bullish_signals.append("📊 RSI فروش بیش از حد")

        if 'macd_bullish' in signals['oscillator']:
            bullish_signals.append("📊 MACD صعودی")

        if 'near_bb_lower' in signals['support_resistance']:
            bullish_signals.append("🎯 نزدیک پایین Bollinger")

        if bullish_signals:
            for signal in bullish_signals:
                st.success(signal)
        else:
            st.info("هیچ سیگنال صعودی قوی")

    with col2:
        st.markdown("#### 📉 سیگنال‌های نزولی")
        bearish_signals = []

        if 'strong_downtrend' in signals['trend']:
            bearish_signals.append("📉 روند نزولی قوی")
        elif 'downtrend' in signals['trend']:
            bearish_signals.append("📉 روند نزولی")

        if 'rsi_overbought' in signals['oscillator']:
            bearish_signals.append("📊 RSI خرید بیش از حد")

        if 'macd_bearish' in signals['oscillator']:
            bearish_signals.append("📊 MACD نزولی")

        if 'near_bb_upper' in signals['support_resistance']:
            bearish_signals.append("🎯 نزدیک بالای Bollinger")

        if bearish_signals:
            for signal in bearish_signals:
                st.error(signal)
        else:
            st.info("هیچ سیگنال نزولی قوی")
    
    with tab3:
        st.header("💰 Smart Money Analysis")
        
        st.markdown("""
        ### 🧠 تحلیل Smart Money چیست؟
        
        Smart Money تحلیل فعالیت‌های نهادی و بانک‌های بزرگ در بازار است که شامل:
        - **Order Blocks:** مناطقی که نهادها سفارشات بزرگ دارند
        - **Liquidity Zones:** نقاط جمع‌آوری نقدینگی
        - **Volume Analysis:** تحلیل حجم معاملات نهادی
        - **Market Structure:** ساختار بازار و تغییرات آن
        """)
        
        # Smart Money Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sm_symbol = st.selectbox(
                "💱 نماد Smart Money:",
                st.session_state.watchlist,
                key="sm_symbol"
            )
        
        with col2:
            sm_timeframe = st.selectbox(
                "⏰ تایم فریم:",
                ['M15', 'H1', 'H4', 'D1'],
                index=1,
                key="sm_timeframe"
            )
        
        with col3:
            sm_lookback = st.slider(
                "🔍 دوره بررسی:",
                50, 500, 200,
                key="sm_lookback"
            )
        
        # Smart Money Analysis Button
        if st.button("🧠 تحلیل Smart Money کامل", type="primary"):
            with st.spinner(f"🔄 تحلیل Smart Money {sm_symbol}..."):
                
                # Get data
                data, success = get_market_data(sm_symbol, sm_timeframe, sm_lookback)
                
                if success:
                    # Calculate indicators
                    data = calculate_all_indicators(data)
                    
                    # Smart Money Analysis
                    smart_result = smart_money_analysis(data, sm_symbol)
                    
                    # Additional Smart Money calculations
                    # Order Flow Analysis
                    recent_data = data.tail(50)
                    
                    # Volume Profile (simplified)
                    price_levels = np.linspace(recent_data['low'].min(), recent_data['high'].max(), 20)
                    volume_profile = []
                    
                    for i in range(len(price_levels) - 1):
                        level_volume = recent_data[
                            (recent_data['low'] <= price_levels[i+1]) & 
                            (recent_data['high'] >= price_levels[i])
                        ]['tick_volume'].sum()
                        
                        volume_profile.append({
                            'price_level': (price_levels[i] + price_levels[i+1]) / 2,
                            'volume': level_volume
                        })
                    
                    # Find high volume nodes
                    volume_profile_df = pd.DataFrame(volume_profile)
                    high_volume_nodes = volume_profile_df.nlargest(5, 'volume')
                    
                    # Institutional Activity Detection
                    # Large body candles with high volume
                    institution_candles = recent_data[
                        (recent_data['Body_Ratio'] > 0.6) & 
                        (recent_data['Volume_Ratio'] > 1.5)
                    ]
                    
                    # Absorption patterns (long wicks with high volume)
                    absorption_patterns = recent_data[
                        ((recent_data['Upper_Wick_Ratio'] > 0.4) | 
                         (recent_data['Lower_Wick_Ratio'] > 0.4)) &
                        (recent_data['Volume_Ratio'] > 1.3)
                    ]
                    
                    # Store results
                    st.session_state.smart_money_analysis = {
                        'symbol': sm_symbol,
                        'timeframe': sm_timeframe,
                        'data': data,
                        'smart_result': smart_result,
                        'volume_profile': volume_profile_df,
                        'high_volume_nodes': high_volume_nodes,
                        'institution_candles': institution_candles,
                        'absorption_patterns': absorption_patterns,
                        'timestamp': datetime.now()
                    }
                    
                    st.success("✅ تحلیل Smart Money کامل شد!")
                
                else:
                    st.error(f"❌ خطا در دریافت داده {sm_symbol}")
        
        # Display Smart Money Results
        if 'smart_money_analysis' in st.session_state:
            sm_result = st.session_state.smart_money_analysis
            
            st.markdown("---")
            st.markdown(f"### 🧠 نتایج Smart Money - {sm_result['symbol']} ({sm_result['timeframe']})")
            
            # Smart Money Summary
            smart_data = sm_result['smart_result']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="signal-{'bullish' if 'خرید' in smart_data['smart_signal'] else 'bearish' if 'فروش' in smart_data['smart_signal'] else 'neutral'}">
                    <h4>🧠 Smart Money Signal</h4>
                    <h3>{smart_data['smart_signal']}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "🏛️ فعالیت نهادی",
                    smart_data['institution_activity'],
                    f"قدرت: {smart_data['institution_strength']}"
                )
            
            with col3:
                st.metric(
                    "📊 Order Flow",
                    smart_data['order_flow'],
                    f"Volume Spike: {smart_data['volume_spike_ratio']}x"
                )
            
            with col4:
                bullish_vol = smart_data['bullish_volume']
                bearish_vol = smart_data['bearish_volume']
                total_vol = bullish_vol + bearish_vol
                
                if total_vol > 0:
                    bullish_percent = (bullish_vol / total_vol) * 100
                    st.metric(
                        "⚖️ حجم خریداران",
                        f"{bullish_percent:.1f}%",
                        f"vs {100-bullish_percent:.1f}% فروشندگان"
                    )
                else:
                    st.metric("⚖️ حجم خریداران", "N/A")
            
            # Smart Money Signals Detail
            if smart_data['signals']:
                st.markdown("### 📋 سیگنال‌های Smart Money")
                
                for i, signal in enumerate(smart_data['signals'], 1):
                    if 'خرید' in signal:
                        st.success(f"{i}. {signal}")
                    elif 'فروش' in signal:
                        st.error(f"{i}. {signal}")
                    else:
                        st.info(f"{i}. {signal}")
            
            # Volume Profile
            st.markdown("### 📊 Volume Profile")
            
            vol_profile = sm_result['volume_profile']
            high_vol_nodes = sm_result['high_volume_nodes']
            
            if not vol_profile.empty:
                fig_volume_profile = go.Figure()
                
                # Volume Profile bars
                fig_volume_profile.add_trace(go.Bar(
                    y=vol_profile['price_level'],
                    x=vol_profile['volume'],
                    orientation='h',
                    name='Volume Profile',
                    marker_color='blue',
                    opacity=0.6
                ))
                
                # High Volume Nodes
                for _, node in high_vol_nodes.iterrows():
                    fig_volume_profile.add_hline(
                        y=node['price_level'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"HVN: {node['volume']:.0f}",
                        annotation_position="top left"
                    )
                
                fig_volume_profile.update_layout(
                    title="Volume Profile - High Volume Nodes",
                    xaxis_title="Volume",
                    yaxis_title="Price Level",
                    height=400
                )
                
                st.plotly_chart(fig_volume_profile, use_container_width=True)
            
            # Institutional Activity
            st.markdown("### 🏛️ فعالیت نهادی")
            
            institution_candles = sm_result['institution_candles']
            absorption_patterns = sm_result['absorption_patterns']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 کندل‌های نهادی")
                st.write(f"**تعداد:** {len(institution_candles)}")
                
                if len(institution_candles) > 0:
                    bullish_inst = institution_candles['Bullish'].sum()
                    bearish_inst = len(institution_candles) - bullish_inst
                    
                    st.write(f"**صعودی:** {bullish_inst}")
                    st.write(f"**نزولی:** {bearish_inst}")
                    
                    if bullish_inst > bearish_inst:
                        st.success("🟢 نهادها بیشتر خریده‌اند")
                    elif bearish_inst > bullish_inst:
                        st.error("🔴 نهادها بیشتر فروخته‌اند")
                    else:
                        st.info("🔵 فعالیت نهادی متعادل")
            
            with col2:
                st.markdown("#### 🛡️ الگوهای جذب")
                st.write(f"**تعداد:** {len(absorption_patterns)}")
                
                if len(absorption_patterns) > 0:
                    upper_absorption = absorption_patterns['Upper_Wick_Ratio'] > 0.4
                    lower_absorption = absorption_patterns['Lower_Wick_Ratio'] > 0.4
                    
                    upper_count = upper_absorption.sum()
                    lower_count = lower_absorption.sum()
                    
                    st.write(f"**جذب بالا:** {upper_count}")
                    st.write(f"**جذب پایین:** {lower_count}")
                    
                    if lower_count > upper_count:
                        st.success("🟢 جذب فروش - احتمال صعود")
                    elif upper_count > lower_count:
                        st.error("🔴 جذب خرید - احتمال نزول")
                    else:
                        st.info("🔵 جذب متعادل")
            
            # Smart Money Chart
            st.markdown("### 📈 نمودار Smart Money")
            
            chart_data = sm_result['data'].tail(100)
            
            fig_smart = go.Figure()
            
            # Candlestick
            fig_smart.add_trace(go.Candlestick(
                x=chart_data.index,
                open=chart_data['open'],
                high=chart_data['high'],
                low=chart_data['low'],
                close=chart_data['close'],
                name=sm_result['symbol']
            ))
            
            # Volume colored by institutional activity
            colors = []
            for _, row in chart_data.iterrows():
                if row['Volume_Ratio'] > 1.5 and row['Body_Ratio'] > 0.6:
                    colors.append('yellow')  # Institutional activity
                elif row['Bullish']:
                    colors.append('green')
                else:
                    colors.append('red')
            
            # Add volume subplot
            fig_smart_with_volume = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Smart Money Price Action', 'Institutional Volume'),
                row_width=[0.7, 0.3]
            )
            
            # Price chart
            fig_smart_with_volume.add_trace(go.Candlestick(
                x=chart_data.index,
                open=chart_data['open'],
                high=chart_data['high'],
                low=chart_data['low'],
                close=chart_data['close'],
                name=sm_result['symbol']
            ), row=1, col=1)
            
            # EMA 20 for trend
            fig_smart_with_volume.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data['EMA_20'],
                mode='lines',
                name='EMA 20',
                line=dict(color='blue', width=1)
            ), row=1, col=1)
            
            # High Volume Nodes as horizontal lines
            for _, node in high_vol_nodes.iterrows():
                fig_smart_with_volume.add_hline(
                    y=node['price_level'],
                    line_dash="dash",
                    line_color="purple",
                    annotation_text="HVN",
                    annotation_position="right",
                    row=1, col=1
                )
            
            # Volume with institutional highlighting
            fig_smart_with_volume.add_trace(go.Bar(
                x=chart_data.index,
                y=chart_data['tick_volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)
            
            # Volume MA
            fig_smart_with_volume.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data['Volume_MA'],
                mode='lines',
                name='Volume MA',
                line=dict(color='orange', width=1)
            ), row=2, col=1)
            
            fig_smart_with_volume.update_layout(
                title=f"{sm_result['symbol']} - Smart Money Analysis",
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig_smart_with_volume, use_container_width=True)
            
            # Market Structure Analysis
            st.markdown("### 🏗️ ساختار بازار")
            
            # Calculate market structure
            recent_highs = chart_data['high'].rolling(10).max()
            recent_lows = chart_data['low'].rolling(10).min()
            
            # Higher highs and higher lows (uptrend)
            higher_highs = (chart_data['high'] > recent_highs.shift(1)).sum()
            higher_lows = (chart_data['low'] > recent_lows.shift(1)).sum()
            
            # Lower highs and lower lows (downtrend)
            lower_highs = (chart_data['high'] < recent_highs.shift(1)).sum()
            lower_lows = (chart_data['low'] < recent_lows.shift(1)).sum()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if higher_highs > lower_highs and higher_lows > lower_lows:
                    st.success("📈 ساختار صعودی - Higher Highs & Higher Lows")
                elif lower_highs > higher_highs and lower_lows > higher_lows:
                    st.error("📉 ساختار نزولی - Lower Highs & Lower Lows")
                else:
                    st.info("📊 ساختار رنج - بدون روند مشخص")
            
            with col2:
                st.metric("📈 Higher Highs", higher_highs)
                st.metric("📈 Higher Lows", higher_lows)
            
            with col3:
                st.metric("📉 Lower Highs", lower_highs)
                st.metric("📉 Lower Lows", lower_lows)
    
    with tab4:
        st.header("📈 نمودارهای زنده")
        
        st.markdown("### 🔴 قیمت‌های زنده")
        
        # Live price controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            live_symbols = st.multiselect(
                "💱 نمادهای زنده:",
                st.session_state.watchlist,
                default=st.session_state.watchlist[:3],
                key="live_symbols"
            )
        
        with col2:
            refresh_interval = st.selectbox(
                "🔄 بازه به‌روزرسانی:",
                [5, 10, 15, 30, 60],
                index=1,
                format_func=lambda x: f"{x} ثانیه"
            )
        
        with col3:
            auto_refresh = st.checkbox("🔄 به‌روزرسانی خودکار", value=True)
        
        # Live prices display
        if live_symbols:
            # Create placeholder for live data
            live_placeholder = st.empty()
            
            # Manual refresh button
            if st.button("🔄 به‌روزرسانی دستی") or auto_refresh:
                
                live_data = []
                
                for symbol in live_symbols:
                    price_data = get_live_price(symbol)
                    if price_data:
                        # Get recent change
                        recent_data, success = get_market_data(symbol, 'M15', 10)
                        if success and len(recent_data) >= 2:
                            prev_close = recent_data['close'].iloc[-2]
                            current_price = price_data['bid']
                            change = ((current_price / prev_close) - 1) * 100
                            change_text = f"{change:+.2f}%"
                            change_color = "green" if change > 0 else "red" if change < 0 else "gray"
                        else:
                            change_text = "N/A"
                            change_color = "gray"
                        
                        live_data.append({
                            'symbol': symbol,
                            'bid': price_data['bid'],
                            'ask': price_data['ask'],
                            'spread': price_data['spread'],
                            'change': change_text,
                            'change_color': change_color,
                            'time': price_data['time'].strftime('%H:%M:%S')
                        })
                
                # Display live prices
                with live_placeholder.container():
                    cols = st.columns(len(live_data))
                    
                    for i, data in enumerate(live_data):
                        with cols[i]:
                            st.markdown(f"""
                            <div style="border: 2px solid {data['change_color']}; padding: 1rem; border-radius: 10px; text-align: center;">
                                <h3>{data['symbol']}</h3>
                                <h2 style="color: {data['change_color']};">{data['bid']:.5f}</h2>
                                <p><strong>Ask:</strong> {data['ask']:.5f}</p>
                                <p><strong>Spread:</strong> {data['spread']:.1f}</p>
                                <p style="color: {data['change_color']}; font-weight: bold;">{data['change']}</p>
                                <p><small>🕒 {data['time']}</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Auto-refresh logic
                if auto_refresh:
                    time.sleep(refresh_interval)
                    st.rerun()
        
        st.markdown("---")
        
        # Real-time Chart
        st.markdown("### 📊 نمودار زنده")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chart_symbol = st.selectbox(
                "💱 نماد نمودار:",
                live_symbols if live_symbols else st.session_state.watchlist,
                key="chart_symbol"
            )
        
        with col2:
            chart_timeframe = st.selectbox(
                "⏰ تایم فریم:",
                ['M1', 'M5', 'M15', 'H1'],
                index=2,
                key="chart_timeframe"
            )
        
        if chart_symbol:
            # Get real-time data
            rt_data, success = get_market_data(chart_symbol, chart_timeframe, 100)
            
            if success:
                # Calculate basic indicators
                rt_data = calculate_all_indicators(rt_data)
                
                # Create real-time chart
                fig_rt = go.Figure()
                
                # Candlestick
                fig_rt.add_trace(go.Candlestick(
                    x=rt_data.index,
                    open=rt_data['open'],
                    high=rt_data['high'],
                    low=rt_data['low'],
                    close=rt_data['close'],
                    name=chart_symbol
                ))
                
                # EMA 20
                fig_rt.add_trace(go.Scatter(
                    x=rt_data.index,
                    y=rt_data['EMA_20'],
                    mode='lines',
                    name='EMA 20',
                    line=dict(color='blue', width=1)
                ))
                
                # Current price line
                current_price = rt_data['close'].iloc[-1]
                fig_rt.add_hline(
                    y=current_price,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Current: {current_price:.5f}",
                    annotation_position="left"
                )
                
                fig_rt.update_layout(
                    title=f"{chart_symbol} - Live Chart ({chart_timeframe})",
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig_rt, use_container_width=True)
                
                # Live stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    daily_range = rt_data['high'].max() - rt_data['low'].min()
                    st.metric("📊 Daily Range", f"{daily_range:.5f}")
                
                with col2:
                    avg_volume = rt_data['tick_volume'].mean()
                    current_volume = rt_data['tick_volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume
                    st.metric("📊 Volume Ratio", f"{volume_ratio:.2f}x")
                
                with col3:
                    volatility = rt_data['ATR'].iloc[-1] / current_price * 100
                    st.metric("⚡ Volatility", f"{volatility:.3f}%")
                
                with col4:
                    rsi_current = rt_data['RSI'].iloc[-1]
                    st.metric("📈 RSI", f"{rsi_current:.1f}")
    
    with tab5:
        st.header("🔔 هشدارها و اسکن بازار")
        
        # Market Scanner
        st.markdown("### 🔍 اسکن بازار")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            scan_categories = st.multiselect(
                "📂 دسته‌های اسکن:",
                ['forex_major', 'gold_metals', 'forex_minor', 'commodities', 'indices'],
                default=['forex_major', 'gold_metals'],
                format_func=lambda x: {
                    'forex_major': '💱 فارکس اصلی',
                    'gold_metals': '🥇 طلا و فلزات',
                    'forex_minor': '💸 فارکس فرعی',
                    'commodities': '🛢️ کالاها',
                    'indices': '📈 شاخص‌ها'
                }[x]
            )
        
        with col2:
            scan_timeframe = st.selectbox(
                "⏰ تایم فریم اسکن:",
                ['M15', 'H1', 'H4', 'D1'],
                index=1
            )
        
        with col3:
            min_confidence = st.slider(
                "🎯 حداقل اعتماد:",
                50, 95, 75,
                step=5
            )
        
        # Scan button
        if st.button("🚀 شروع اسکن بازار", type="primary"):
            
            # Collect symbols to scan
            symbols_to_scan = []
            for category in scan_categories:
                symbols_to_scan.extend(symbol_categories.get(category, []))
            
            if symbols_to_scan:
                with st.spinner(f"🔄 اسکن {len(symbols_to_scan)} نماد..."):
                    
                    scan_results = scan_market(symbols_to_scan, scan_timeframe, min_confidence)
                    
                    st.session_state.scan_results = {
                        'results': scan_results,
                        'timeframe': scan_timeframe,
                        'min_confidence': min_confidence,
                        'timestamp': datetime.now()
                    }
                    
                    st.success(f"✅ اسکن کامل شد! {len(scan_results)} سیگنال یافت شد.")
            
            else:
                st.warning("⚠️ لطفاً حداقل یک دسته انتخاب کنید")
        
        # Display scan results
        if 'scan_results' in st.session_state:
            scan_data = st.session_state.scan_results
            results = scan_data['results']
            
            if results:
                st.markdown(f"### 📊 نتایج اسکن - {len(results)} سیگنال")
                st.write(f"**تایم فریم:** {scan_data['timeframe']} | **حداقل اعتماد:** {scan_data['min_confidence']}% | **زمان:** {scan_data['timestamp'].strftime('%H:%M:%S')}")
                
                # Sort by confidence
                results_sorted = sorted(results, key=lambda x: x['analysis']['confidence'], reverse=True)
                
                # Display results table
                scan_table_data = []
                
                for result in results_sorted:
                    analysis = result['analysis']
                    smart_money = result['smart_money']
                    live_price = result['live_price']
                    
                    scan_table_data.append({
                        'نماد': result['symbol'],
                        'سیگنال': analysis['overall_signal'],
                        'اعتماد': f"{analysis['confidence']}%",
                        'روند': analysis['trend'],
                        'قدرت': analysis['strength'],
                        'Smart Money': smart_money['smart_signal'][:20],
                        'قیمت': f"{live_price['bid']:.5f}" if live_price else 'N/A',
                        'اسپرد': f"{live_price['spread']:.1f}" if live_price else 'N/A',
                        'ریسک': analysis['risk_level']
                    })
                
                scan_df = pd.DataFrame(scan_table_data)
                st.dataframe(scan_df, use_container_width=True)
                
                # Top signals
                st.markdown("### 🏆 برترین سیگنال‌ها")
                
                top_signals = results_sorted[:3]
                
                cols = st.columns(len(top_signals))
                
                for i, result in enumerate(top_signals):
                    with cols[i]:
                        analysis = result['analysis']
                        symbol = result['symbol']
                        confidence = analysis['confidence']
                        signal = analysis['overall_signal']
                        
                        signal_color = 'green' if 'خرید' in signal else 'red' if 'فروش' in signal else 'gray'
                        
                        st.markdown(f"""
                        <div style="border: 3px solid {signal_color}; padding: 1rem; border-radius: 10px; text-align: center;">
                            <h3>🏆 #{i+1}</h3>
                            <h2>{symbol}</h2>
                            <h3 style="color: {signal_color};">{signal}</h3>
                            <p><strong>اعتماد:</strong> {confidence}%</p>
                            <p><strong>روند:</strong> {analysis['trend']}</p>
                            <p><strong>ریسک:</strong> {analysis['risk_level']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            else:
                st.info("ℹ️ هیچ سیگنالی با این معیارها یافت نشد")
        
        st.markdown("---")
        
        # Alert System
        st.markdown("### 🔔 سیستم هشدار")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ➕ افزودن هشدار جدید")
            
            with st.form("add_alert"):
                alert_symbol = st.selectbox("💱 نماد:", st.session_state.watchlist)
                alert_type = st.selectbox("🔔 نوع هشدار:", ['قیمت', 'RSI', 'سیگنال تحلیل'])
                
                if alert_type == 'قیمت':
                    alert_condition = st.selectbox("📊 شرط:", ['بالاتر از', 'پایین‌تر از'])
                    alert_value = st.number_input("💰 مقدار:", value=1.0, step=0.00001, format="%.5f")
                
                elif alert_type == 'RSI':
                    alert_condition = st.selectbox("📊 شرط:", ['بالاتر از', 'پایین‌تر از'])
                    alert_value = st.number_input("📈 RSI:", value=70.0, min_value=0.0, max_value=100.0)
                
                else:  # سیگنال تحلیل
                    alert_condition = st.selectbox("📊 سیگنال:", ['خرید قوی', 'فروش قوی', 'هر سیگنال'])
                    alert_value = st.slider("🎯 حداقل اعتماد:", 50, 95, 80)
                
                alert_comment = st.text_input("💬 توضیحات (اختیاری):")
                
                if st.form_submit_button("✅ افزودن هشدار"):
                    new_alert = {
                        'symbol': alert_symbol,
                        'type': alert_type,
                        'condition': alert_condition,
                        'value': alert_value,
                        'comment': alert_comment,
                        'active': True,
                        'created': datetime.now(),
                        'triggered': False
                    }
                    
                    st.session_state.alerts.append(new_alert)
                    st.success("✅ هشدار اضافه شد!")
        
        with col2:
            st.markdown("#### 📋 هشدارهای فعال")
            
            if st.session_state.alerts:
                active_alerts = [alert for alert in st.session_state.alerts if alert['active']]
                
                for i, alert in enumerate(active_alerts):
                    with st.expander(f"🔔 {alert['symbol']} - {alert['type']}"):
                        st.write(f"**شرط:** {alert['condition']} {alert['value']}")
                        if alert['comment']:
                            st.write(f"**توضیحات:** {alert['comment']}")
                        st.write(f"**ایجاد شده:** {alert['created'].strftime('%Y-%m-%d %H:%M')}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button(f"❌ حذف", key=f"delete_alert_{i}"):
                                st.session_state.alerts.remove(alert)
                                st.rerun()
                        
                        with col_b:
                            if st.button(f"⏸️ توقف", key=f"pause_alert_{i}"):
                                alert['active'] = False
                                st.rerun()
            
            else:
                st.info("هیچ هشدار فعالی وجود ندارد")
        
        # Check alerts
        if st.button("🔍 بررسی هشدارها"):
            triggered_alerts = []
            
            for alert in st.session_state.alerts:
                if not alert['active'] or alert['triggered']:
                    continue
                
                try:
                    if alert['type'] == 'قیمت':
                        live_price = get_live_price(alert['symbol'])
                        if live_price:
                            current_price = live_price['bid']
                            
                            if alert['condition'] == 'بالاتر از' and current_price > alert['value']:
                                triggered_alerts.append(alert)
                                alert['triggered'] = True
                            elif alert['condition'] == 'پایین‌تر از' and current_price < alert['value']:
                                triggered_alerts.append(alert)
                                alert['triggered'] = True
                    
                    elif alert['type'] == 'RSI':
                        data, success = get_market_data(alert['symbol'], 'H1', 50)
                        if success:
                            data = calculate_all_indicators(data)
                            current_rsi = data['RSI'].iloc[-1]
                            
                            if alert['condition'] == 'بالاتر از' and current_rsi > alert['value']:
                                triggered_alerts.append(alert)
                                alert['triggered'] = True
                            elif alert['condition'] == 'پایین‌تر از' and current_rsi < alert['value']:
                                triggered_alerts.append(alert)
                                alert['triggered'] = True
                    
                    elif alert['type'] == 'سیگنال تحلیل':
                        data, success = get_market_data(alert['symbol'], 'H1', 200)
                        if success:
                            data = calculate_all_indicators(data)
                            analysis = advanced_market_analysis(data, alert['symbol'])
                            
                            if analysis['confidence'] >= alert['value']:
                                signal = analysis['overall_signal']
                                
                                trigger_conditions = {
                                    'خرید قوی': 'خرید قوی' in signal,
                                    'فروش قوی': 'فروش قوی' in signal,
                                    'هر سیگنال': True
                                }
                                
                                if trigger_conditions.get(alert['condition'], False):
                                    triggered_alerts.append(alert)
                                    alert['triggered'] = True
                
                except Exception as e:
                    continue
            
            if triggered_alerts:
                st.markdown("### 🚨 هشدارهای فعال شده")
                
                for alert in triggered_alerts:
                    st.error(f"🚨 **{alert['symbol']}** - {alert['type']}: {alert['condition']} {alert['value']}")
                    
                    # Add to notifications
                    notification = {
                        'type': 'alert',
                        'symbol': alert['symbol'],
                        'message': f"{alert['type']}: {alert['condition']} {alert['value']}",
                        'time': datetime.now()
                    }
                    
                    if 'notifications' not in st.session_state:
                        st.session_state.notifications = []
                    
                    st.session_state.notifications.append(notification)
            
            else:
                st.success("✅ هیچ هشدار جدیدی فعال نشده")
    
    with tab6:
        st.header("🤖 ربات معاملاتی (شبیه‌سازی)")
        
        st.markdown("""
        ### ⚠️ توجه: این بخش فقط شبیه‌سازی است
        
        همه معاملات فقط برای تست و یادگیری هستند و هیچ معامله واقعی انجام نمی‌شود.
        """)
        
        # Bot Configuration
        st.markdown("### ⚙️ تنظیمات ربات")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bot_symbols = st.multiselect(
                "💱 نمادهای ربات:",
                st.session_state.watchlist,
                default=['EURUSD', 'GBPUSD'],
                key="bot_symbols"
            )
        
        with col2:
            bot_timeframe = st.selectbox(
                "⏰ تایم فریم:",
                ['M15', 'H1', 'H4', 'D1'],
                index=1,
                key="bot_timeframe"
            )
        
        with col3:
            bot_confidence = st.slider(
                "🎯 حداقل اعتماد:",
                60, 95, 80,
                key="bot_confidence"
            )
        
        # Risk Management
        st.markdown("### ⚠️ مدیریت ریسک")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            initial_balance = st.number_input(
                "💰 موجودی اولیه:",
                value=10000.0,
                min_value=100.0,
                step=100.0
            )
        
        with col2:
            risk_per_trade = st.slider(
                "📊 ریسک هر معامله (%):",
                0.5, 5.0, 2.0,
                step=0.1
            )
        
        with col3:
            max_positions = st.slider(
                "📈 حداکثر پوزیشن:",
                1, 10, 3
            )
        
        with col4:
            stop_loss_atr = st.slider(
                "🛑 Stop Loss (ATR):",
                1.0, 5.0, 2.0,
                step=0.1
            )
        
        # Initialize bot session
        if 'trading_bot' not in st.session_state:
            st.session_state.trading_bot = {
                'active': False,
                'balance': initial_balance,
                'initial_balance': initial_balance,
                'positions': [],
                'trades_history': [],
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
        
        # Bot Controls
        st.markdown("### 🎮 کنترل ربات")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not st.session_state.trading_bot['active']:
                if st.button("🚀 شروع ربات", type="primary"):
                    st.session_state.trading_bot['active'] = True
                    st.session_state.trading_bot['balance'] = initial_balance
                    st.session_state.trading_bot['initial_balance'] = initial_balance
                    st.success("✅ ربات شروع شد!")
                    st.rerun()
            else:
                if st.button("⏹️ توقف ربات", type="secondary"):
                    st.session_state.trading_bot['active'] = False
                    st.warning("⏸️ ربات متوقف شد!")
                    st.rerun()
        
        with col2:
            if st.button("🔄 ریست ربات"):
                st.session_state.trading_bot = {
                    'active': False,
                    'balance': initial_balance,
                    'initial_balance': initial_balance,
                    'positions': [],
                    'trades_history': [],
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0
                }
                st.info("🔄 ربات ریست شد!")
                st.rerun()
        
        with col3:
            if st.session_state.trading_bot['active'] and st.button("🔍 جستجوی معامله"):
                with st.spinner("🔄 جستجوی فرصت معامله..."):
                    # Search for trading opportunities
                    for symbol in bot_symbols:
                        try:
                            data, success = get_market_data(symbol, bot_timeframe, 200)
                            
                            if success:
                                data = calculate_all_indicators(data)
                                analysis = advanced_market_analysis(data, symbol)
                                
                                if analysis['confidence'] >= bot_confidence:
                                    # Check if already have position
                                    existing_positions = [p for p in st.session_state.trading_bot['positions'] if p['symbol'] == symbol]
                                    
                                    if not existing_positions and len(st.session_state.trading_bot['positions']) < max_positions:
                                        # Calculate position size
                                        current_balance = st.session_state.trading_bot['balance']
                                        risk_amount = current_balance * (risk_per_trade / 100)
                                        
                                        # Get current price and ATR
                                        live_price = get_live_price(symbol)
                                        if live_price:
                                            current_price = live_price['bid']
                                            atr = data['ATR'].iloc[-1]
                                            
                                            # Calculate lot size (simplified)
                                            pip_value = 1  # Simplified
                                            stop_loss_pips = atr * stop_loss_atr * 10000  # Convert to pips
                                            lot_size = risk_amount / (stop_loss_pips * pip_value)
                                            lot_size = max(0.01, min(lot_size, 1.0))  # Limit lot size
                                            
                                            # Determine action
                                            if 'خرید' in analysis['overall_signal']:
                                                action = 'BUY'
                                                stop_loss = current_price - (atr * stop_loss_atr)
                                                take_profit = current_price + (atr * stop_loss_atr * 2)  # 1:2 R/R
                                            elif 'فروش' in analysis['overall_signal']:
                                                action = 'SELL'
                                                stop_loss = current_price + (atr * stop_loss_atr)
                                                take_profit = current_price - (atr * stop_loss_atr * 2)  # 1:2 R/R
                                            else:
                                                continue
                                            
                                            # Create new position
                                            new_position = {
                                                'symbol': symbol,
                                                'action': action,
                                                'lot_size': lot_size,
                                                'entry_price': current_price,
                                                'stop_loss': stop_loss,
                                                'take_profit': take_profit,
                                                'open_time': datetime.now(),
                                                'confidence': analysis['confidence'],
                                                'analysis': analysis
                                            }
                                            
                                            st.session_state.trading_bot['positions'].append(new_position)
                                            
                                            st.success(f"✅ پوزیشن جدید: {action} {symbol} - {lot_size:.2f} لات")
                        
                        except Exception as e:
                            continue
                    
                    if not any(st.session_state.trading_bot['positions']):
                        st.info("ℹ️ فرصت معامله‌ای یافت نشد")
        
        # Bot Status
        if st.session_state.trading_bot['active']:
            st.success("🟢 ربات فعال است")
        else:
            st.error("🔴 ربات غیرفعال است")
        
        # Bot Statistics
        st.markdown("### 📊 آمار ربات")
        
        bot_data = st.session_state.trading_bot
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            current_balance = bot_data['balance']
            initial_balance = bot_data['initial_balance']
            profit_loss = current_balance - initial_balance
            profit_loss_percent = (profit_loss / initial_balance) * 100
            
            st.metric(
                "💰 موجودی فعلی",
                f"${current_balance:,.2f}",
                f"{profit_loss:+.2f} ({profit_loss_percent:+.1f}%)"
            )
        
        with col2:
            total_trades = bot_data['total_trades']
            st.metric("📊 کل معاملات", total_trades)
        
        with col3:
            winning_trades = bot_data['winning_trades']
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            st.metric("🎯 نرخ برد", f"{win_rate:.1f}%", f"{winning_trades} برد")
        
        with col4:
            losing_trades = bot_data['losing_trades']
            st.metric("❌ معاملات ضرر", losing_trades)
        
        with col5:
            open_positions = len(bot_data['positions'])
            st.metric("📈 پوزیشن‌های باز", open_positions)
        
        # Open Positions
        if bot_data['positions']:
            st.markdown("### 📈 پوزیشن‌های باز")
            
            positions_data = []
            for pos in bot_data['positions']:
                # Get current price for P&L calculation
                live_price = get_live_price(pos['symbol'])
                if live_price:
                    current_price = live_price['bid'] if pos['action'] == 'SELL' else live_price['ask']
                    
                    if pos['action'] == 'BUY':
                        unrealized_pnl = (current_price - pos['entry_price']) * pos['lot_size'] * 100000  # Simplified
                    else:
                        unrealized_pnl = (pos['entry_price'] - current_price) * pos['lot_size'] * 100000  # Simplified
                    
                    positions_data.append({
                        'نماد': pos['symbol'],
                        'عمل': pos['action'],
                        'حجم': f"{pos['lot_size']:.2f}",
                        'قیمت ورود': f"{pos['entry_price']:.5f}",
                        'قیمت فعلی': f"{current_price:.5f}",
                        'S/L': f"{pos['stop_loss']:.5f}",
                        'T/P': f"{pos['take_profit']:.5f}",
                        'سود/ضرر': f"${unrealized_pnl:.2f}",
                        'اعتماد': f"{pos['confidence']}%",
                        'زمان': pos['open_time'].strftime('%H:%M')
                    })
            
            if positions_data:
                positions_df = pd.DataFrame(positions_data)
                st.dataframe(positions_df, use_container_width=True)
        
        # Trading History
        if bot_data['trades_history']:
            st.markdown("### 📋 تاریخچه معاملات")
            
            history_data = []
            for trade in bot_data['trades_history'][-10:]:  # Last 10 trades
                history_data.append({
                    'نماد': trade['symbol'],
                    'عمل': trade['action'],
                    'حجم': f"{trade['lot_size']:.2f}",
                    'ورود': f"{trade['entry_price']:.5f}",
                    'خروج': f"{trade['exit_price']:.5f}",
                    'سود/ضرر': f"${trade['profit_loss']:.2f}",
                    'نتیجه': '✅ برد' if trade['profit_loss'] > 0 else '❌ ضرر',
                    'زمان ورود': trade['open_time'].strftime('%m/%d %H:%M'),
                    'زمان خروج': trade['close_time'].strftime('%m/%d %H:%M')
                })
            
            if history_data:
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True)
    
    with tab7:
        st.header("📋 گزارشات و آمار")
        
        # Performance Summary
        st.markdown("### 📊 خلاصه عملکرد")
        
        if st.session_state.trading_signals or 'scan_results' in st.session_state:
            
            # Signal Analysis
            if st.session_state.trading_signals:
                signals = st.session_state.trading_signals
                
                # Count signals by type
                buy_signals = len([s for s in signals if 'خرید' in s['signal']])
                sell_signals = len([s for s in signals if 'فروش' in s['signal']])
                hold_signals = len(signals) - buy_signals - sell_signals
                
                # Average confidence
                avg_confidence = np.mean([s['confidence'] for s in signals])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎯 کل سیگنال‌ها", len(signals))
                
                with col2:
                    st.metric("📈 خرید", buy_signals, f"{buy_signals/len(signals)*100:.1f}%")
                
                with col3:
                    st.metric("📉 فروش", sell_signals, f"{sell_signals/len(signals)*100:.1f}%")
                
                with col4:
                    st.metric("🔵 خنثی", hold_signals, f"میانگین اعتماد: {avg_confidence:.1f}%")
                
                # Signals by symbol
                st.markdown("### 📊 سیگنال‌ها بر اساس نماد")
                
                symbol_counts = {}
                for signal in signals:
                    symbol = signal['symbol']
                    if symbol not in symbol_counts:
                        symbol_counts[symbol] = {'total': 0, 'buy': 0, 'sell': 0}
                    
                    symbol_counts[symbol]['total'] += 1
                    if 'خرید' in signal['signal']:
                        symbol_counts[symbol]['buy'] += 1
                    elif 'فروش' in signal['signal']:
                        symbol_counts[symbol]['sell'] += 1
                
                # Create chart
                symbols = list(symbol_counts.keys())
                buy_counts = [symbol_counts[s]['buy'] for s in symbols]
                sell_counts = [symbol_counts[s]['sell'] for s in symbols]
                
                fig_signals = go.Figure()
                
                fig_signals.add_trace(go.Bar(
                    x=symbols,
                    y=buy_counts,
                    name='خرید',
                    marker_color='green'
                ))
                
                fig_signals.add_trace(go.Bar(
                    x=symbols,
                    y=sell_counts,
                    name='فروش',
                    marker_color='red'
                ))
                
                fig_signals.update_layout(
                    title="توزیع سیگنال‌ها بر اساس نماد",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_signals, use_container_width=True)
        
        else:
            st.info("ℹ️ هنوز سیگنال یا اسکنی انجام نشده است")
        
        # Market Overview Report
        st.markdown("### 🌍 گزارش نمای کلی بازار")
        
        if st.button("📊 تولید گزارش جامع"):
            with st.spinner("📄 تولید گزارش..."):
                
                # Analyze major pairs
                major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD']
                market_report = []
                
                for symbol in major_pairs:
                    try:
                        data, success = get_market_data(symbol, 'D1', 30)  # 30 days
                        
                        if success:
                            data = calculate_all_indicators(data)
                            analysis = advanced_market_analysis(data, symbol)
                            smart_analysis = smart_money_analysis(data, symbol)
                            
                            # Calculate additional metrics
                            price_change_30d = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
                            volatility_30d = data['close'].pct_change().std() * np.sqrt(30) * 100
                            
                            market_report.append({
                                'symbol': symbol,
                                'analysis': analysis,
                                'smart_money': smart_analysis,
                                'price_change_30d': price_change_30d,
                                'volatility_30d': volatility_30d
                            })
                    
                    except Exception as e:
                        continue
                
                if market_report:
                    # Market sentiment
                    bullish_count = len([r for r in market_report if 'خرید' in r['analysis']['overall_signal']])
                    bearish_count = len([r for r in market_report if 'فروش' in r['analysis']['overall_signal']])
                    neutral_count = len(market_report) - bullish_count - bearish_count
                    
                    st.markdown("#### 📈 حال کلی بازار")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📈 صعودی", f"{bullish_count}/{len(market_report)}", f"{bullish_count/len(market_report)*100:.1f}%")
                    
                    with col2:
                        st.metric("📉 نزولی", f"{bearish_count}/{len(market_report)}", f"{bearish_count/len(market_report)*100:.1f}%")
                    
                    with col3:
                        st.metric("🔵 خنثی", f"{neutral_count}/{len(market_report)}", f"{neutral_count/len(market_report)*100:.1f}%")
                    
                    # Market strength pie chart
                    fig_sentiment = go.Figure(data=[go.Pie(
                        labels=['صعودی', 'نزولی', 'خنثی'],
                        values=[bullish_count, bearish_count, neutral_count],
                        marker_colors=['green', 'red', 'gray']
                    )])
                    
                    fig_sentiment.update_layout(
                        title="حالت کلی بازار",
                        height=400
                    )
                    
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                    
                    # Detailed report table
                    st.markdown("#### 📋 گزارش تفصیلی")
                    
                    report_data = []
                    for item in market_report:
                        report_data.append({
                            'نماد': item['symbol'],
                            'سیگنال': item['analysis']['overall_signal'],
                            'اعتماد': f"{item['analysis']['confidence']}%",
                            'روند': item['analysis']['trend'],
                            'Smart Money': item['smart_money']['smart_signal'][:15] + '...',
                            'تغییر 30 روز': f"{item['price_change_30d']:+.2f}%",
                            'نوسان': f"{item['volatility_30d']:.2f}%",
                            'ریسک': item['analysis']['risk_level']
                        })
                    
                    report_df = pd.DataFrame(report_data)
                    st.dataframe(report_df, use_container_width=True)
                    
                    # Export functionality
                    st.markdown("#### 💾 خروجی گزارش")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📥 دانلود CSV"):
                            csv = report_df.to_csv(index=False)
                            st.download_button(
                                label="📁 دانلود فایل CSV",
                                data=csv,
                                file_name=f"forex_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                    
                    with col2:
                        # Summary text report
                        summary_text = f"""
🔹 تاریخ گزارش: {datetime.now().strftime('%Y-%m-%d %H:%M')}
🔹 تعداد نمادهای بررسی شده: {len(market_report)}
🔹 حالت کلی بازار: {'صعودی' if bullish_count > bearish_count else 'نزولی' if bearish_count > bullish_count else 'خنثی'}
🔹 نمادهای صعودی: {bullish_count} ({bullish_count/len(market_report)*100:.1f}%)
🔹 نمادهای نزولی: {bearish_count} ({bearish_count/len(market_report)*100:.1f}%)
🔹 میانگین اعتماد: {np.mean([r['analysis']['confidence'] for r in market_report]):.1f}%

📊 توصیه کلی:
{
'بازار در حالت صعودی است. فرصت‌های خرید را در نظر بگیرید.' if bullish_count > bearish_count
else 'بازار در حالت نزولی است. احتیاط کنید و فرصت‌های فروش را بررسی کنید.' if bearish_count > bullish_count
else 'بازار در حالت خنثی است. منتظر سیگنال‌های قوی‌تر باشید.'
}
                        """
                        
                        if st.button("📄 دانلود خلاصه متنی"):
                            st.download_button(
                                label="📁 دانلود خلاصه",
                                data=summary_text,
                                file_name=f"forex_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                mime="text/plain"
                            )
                
                else:
                    st.error("❌ خطا در تولید گزارش")
        
# System Statistics
st.markdown("---")
st.markdown("### 🔧 آمار سیستم")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔗 وضعیت اتصال", "✅ متصل" if st.session_state.mt5_connected else "❌ قطع")

with col2:
    analysis_count = len(st.session_state.analysis_history) if 'analysis_history' in st.session_state else 0
    st.metric("📊 تحلیل‌های انجام شده", analysis_count)

if st.session_state.mt5_connected:
    with col3:
        alerts_count = len(st.session_state.alerts) if st.session_state.alerts else 0
        st.metric("🔔 هشدارهای فعال", alerts_count)

    with col4:
        session_time = datetime.now()
        st.metric("⏰ زمان جلسه", session_time.strftime('%H:%M'))
else:
    st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>🔌 اتصال به MT5 الزامی است</h2>
            <p>برای استفاده از تمامی امکانات سیستم، لطفاً ابتدا به MetaTrader 5 متصل شوید.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h4>🚀 سیستم فارکس کامل و پیشرفته</h4>
        <p>💰 Smart Money | 🤖 AI Analysis | 📊 Real-time Data | 🎯 Professional Trading</p>
        <p><small>ساخته شده با ❤️ برای معامله‌گران حرفه‌ای</small></p>
    </div>
""", unsafe_allow_html=True)

