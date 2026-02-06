"""
SET50 Futures Calendar Spread Analysis
Analyzes calendar spreads for SET50 futures and sends reports to Telegram
Supports both 5-minute (main chart) and 1-minute (intraday chart) timeframes
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
# Telegram API (use environment variables for security)
API_TOKEN = os.getenv("TELEGRAM_API_TOKEN", "8194117315:AAER8upMXj0RjSiQ8DHKDTMP1sdEZP3kkhw")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7311904934")

# Interest rate and dividend yield
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.017"))
DIVIDEND_YIELD = float(os.getenv("DIVIDEND_YIELD", "0.0373"))

# Timezone
TZ = 'Asia/Bangkok'

# Output directory
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# TELEGRAM FUNCTIONS
# =============================================================================
def send_message(message: str) -> bool:
    """Send a text message to Telegram"""
    if not API_TOKEN or not CHAT_ID:
        print("Warning: Telegram credentials not configured")
        return False

    api_url = f'https://api.telegram.org/bot{API_TOKEN}/sendMessage'
    try:
        response = requests.post(api_url, json={'chat_id': CHAT_ID, 'text': message})
        print(f"Message sent: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending message: {e}")
        return False


def send_photo(image_path: str) -> bool:
    """Send a photo to Telegram"""
    if not API_TOKEN or not CHAT_ID:
        print("Warning: Telegram credentials not configured")
        return False

    api_url = f'https://api.telegram.org/bot{API_TOKEN}/sendPhoto'
    try:
        with open(image_path, 'rb') as photo:
            response = requests.post(api_url, files={'photo': photo}, data={'chat_id': CHAT_ID})
        print(f"Photo sent: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error sending photo: {e}")
        return False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def make_spread_name(near_symbol: str, far_symbol: str) -> str:
    """
    Build a compact spread name like 'S50G26H26'
    from symbols like 'S50G2026' and 'S50H2026'.
    Format: <product><near_month><near_YY><far_month><far_YY>
    """
    product = near_symbol[:3]
    near_month = near_symbol[3]
    near_yy = near_symbol[-2:]
    far_month = far_symbol[3]
    far_yy = far_symbol[-2:]
    return f"{product}{near_month}{near_yy}{far_month}{far_yy}"


def get_trading_days(df: pd.DataFrame, n_days: int = 5) -> pd.DataFrame | None:
    """
    Filter dataframe to last N trading days.
    Returns None if fewer than n_days trading days are available.
    """
    df = df.copy()
    df['_date'] = pd.to_datetime(df['Timestamp']).dt.date
    unique_dates = sorted(df['_date'].unique())

    if len(unique_dates) < n_days:
        return None

    last_n_dates = unique_dates[-n_days:]
    filtered = df[df['_date'].isin(last_n_dates)].copy()
    filtered.drop(columns=['_date'], inplace=True)
    return filtered


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load price data for a given symbol and timeframe"""
    from price_loaders.tradingview import load_asset_price
    return load_asset_price(symbol, 100000, timeframe, None)


def load_set50(timeframe: str) -> pd.DataFrame:
    """Load SET50 index data"""
    df = load_data('SET50', timeframe)
    return df[['time', 'close']].rename(columns={'time': 'Timestamp', 'close': 'SET50_Close'})


def load_futures(symbol: str, prefix: str, timeframe: str) -> pd.DataFrame:
    """Load futures contract data"""
    df = load_data(f'TFEX:{symbol}', timeframe)
    return df[['time', 'open', 'high', 'low', 'close']].rename(columns={
        'time': 'Timestamp',
        'open': f'{prefix}_Open',
        'high': f'{prefix}_High',
        'low': f'{prefix}_Low',
        'close': f'{prefix}_Close'
    })


# =============================================================================
# SPREAD ANALYSIS
# =============================================================================
class CalendarSpreadAnalyzer:
    """Analyzes calendar spreads between two futures contracts"""

    def __init__(self, near_symbol: str, far_symbol: str,
                 near_expiry: str, far_expiry: str,
                 near_label: str, far_label: str):
        self.near_symbol = near_symbol
        self.far_symbol = far_symbol
        self.near_expiry = near_expiry
        self.far_expiry = far_expiry
        self.near_label = near_label
        self.far_label = far_label
        self.spread_name = make_spread_name(near_symbol, far_symbol)

        # Will be populated by load_and_calculate()
        self.df_5min = None
        self.df_1min = None
        self.near_prefix = None
        self.far_prefix = None
        self.spread_prefix = None

    def _calculate_spread_metrics(self, df: pd.DataFrame,
                                  near_prefix: str, far_prefix: str,
                                  spread_prefix: str) -> pd.DataFrame:
        """Calculate all spread metrics for a given dataframe"""

        # Spread OHLC
        for col in ['Open', 'High', 'Low', 'Close']:
            df[f'{spread_prefix}_{col}'] = df[f'{far_prefix}_{col}'] - df[f'{near_prefix}_{col}']

        # All possible spread combinations
        df[f'{spread_prefix}_Diff_High_High'] = df[f'{far_prefix}_High'] - df[f'{near_prefix}_High']
        df[f'{spread_prefix}_Diff_High_Low'] = df[f'{far_prefix}_Low'] - df[f'{near_prefix}_High']
        df[f'{spread_prefix}_Diff_Low_High'] = df[f'{far_prefix}_High'] - df[f'{near_prefix}_Low']
        df[f'{spread_prefix}_Diff_Low_Low'] = df[f'{far_prefix}_Low'] - df[f'{near_prefix}_Low']

        # Maximum difference
        df[f'{spread_prefix}_Max_Diff'] = df[[
            f'{spread_prefix}_Diff_High_High',
            f'{spread_prefix}_Diff_High_Low',
            f'{spread_prefix}_Diff_Low_High',
            f'{spread_prefix}_Diff_Low_Low'
        ]].max(axis=1)

        # Time to expiry
        near_expiry_dt = pd.to_datetime(self.near_expiry).tz_localize(TZ)
        far_expiry_dt = pd.to_datetime(self.far_expiry).tz_localize(TZ)

        df[f'{near_prefix}_Time_to_expiry'] = (
            (near_expiry_dt - df['Timestamp']).dt.total_seconds() / (24 * 3600)
        ).astype(int)

        df[f'{far_prefix}_Time_to_expiry'] = (
            (far_expiry_dt - df['Timestamp']).dt.total_seconds() / (24 * 3600)
        ).astype(int)

        # Theoretical pricing (Cost-of-Carry Model)
        df[f'{near_prefix}_Expected_Futures_Price'] = (
            df['SET50_Close'] *
            np.exp((RISK_FREE_RATE - DIVIDEND_YIELD) * (df[f'{near_prefix}_Time_to_expiry'] / 365))
        )

        df[f'{far_prefix}_Expected_Futures_Price'] = (
            df['SET50_Close'] *
            np.exp((RISK_FREE_RATE - DIVIDEND_YIELD) * (df[f'{far_prefix}_Time_to_expiry'] / 365))
        )

        # Basis
        df[f'{near_prefix}_Basis'] = df[f'{near_prefix}_Expected_Futures_Price'] - df['SET50_Close']
        df[f'{far_prefix}_Basis'] = df[f'{far_prefix}_Expected_Futures_Price'] - df['SET50_Close']

        # Theoretical and actual spread basis
        df[f'{spread_prefix}_Theoretical_Basis'] = df[f'{far_prefix}_Basis'] - df[f'{near_prefix}_Basis']
        df[f'{spread_prefix}_Actual_Basis'] = df[f'{spread_prefix}_Close']
        df[f'{spread_prefix}_Mispricing'] = (
            df[f'{spread_prefix}_Actual_Basis'] - df[f'{spread_prefix}_Theoretical_Basis']
        )

        return df

    def load_and_calculate(self) -> None:
        """Load data for both timeframes and calculate spread metrics"""
        near_prefix = self.near_symbol
        far_prefix = self.far_symbol
        spread_prefix = self.spread_name

        self.near_prefix = near_prefix
        self.far_prefix = far_prefix
        self.spread_prefix = spread_prefix

        for timeframe, attr_name in [('5', 'df_5min'), ('1', 'df_1min')]:
            print(f"  Loading {timeframe}-min data...")
            try:
                set50 = load_set50(timeframe)
                near = load_futures(self.near_symbol, near_prefix, timeframe)
                far = load_futures(self.far_symbol, far_prefix, timeframe)

                df = near.merge(far, on='Timestamp', how='left')
                df = set50.merge(df, on='Timestamp', how='inner')
                df = self._calculate_spread_metrics(df, near_prefix, far_prefix, spread_prefix)

                setattr(self, attr_name, df)
                print(f"  {timeframe}-min data loaded: {len(df)} rows")
            except Exception as e:
                print(f"  Warning: Failed to load {timeframe}-min data: {e}")
                setattr(self, attr_name, None)

    def find_extremes(self, df: pd.DataFrame = None) -> dict:
        """Find max and min spread values"""
        if df is None:
            df = self.df_5min
        spread_prefix = self.spread_prefix

        # Find max
        max_idx = df[f'{spread_prefix}_Max_Diff'].idxmax()
        max_row = df.loc[max_idx]
        max_value = np.round(max_row[f'{spread_prefix}_Max_Diff'], 3)
        max_timestamp = max_row['Timestamp']

        diff_cols = ['Diff_High_High', 'Diff_High_Low', 'Diff_Low_High', 'Diff_Low_Low']
        type_labels = ['H-H', 'H-L', 'L-H', 'L-L']
        max_type = type_labels[0]
        for col, label in zip(diff_cols, type_labels):
            if max_row[f'{spread_prefix}_{col}'] == max_row[f'{spread_prefix}_Max_Diff']:
                max_type = label
                break

        # Find min
        min_idx = df[f'{spread_prefix}_Max_Diff'].idxmin()
        min_row = df.loc[min_idx]
        min_value = np.round(min_row[f'{spread_prefix}_Max_Diff'], 3)
        min_timestamp = min_row['Timestamp']

        min_type = type_labels[0]
        for col, label in zip(diff_cols, type_labels):
            if min_row[f'{spread_prefix}_{col}'] == min_row[f'{spread_prefix}_Max_Diff']:
                min_type = label
                break

        return {
            'max_value': max_value,
            'max_timestamp': max_timestamp,
            'max_type': max_type,
            'min_value': min_value,
            'min_timestamp': min_timestamp,
            'min_type': min_type
        }

    def _draw_chart(self, df: pd.DataFrame, title: str, img_path: str,
                    date_fmt: str = '%Y-%m-%d', n_labels: int = 10,
                    linewidth: float = 1.5, figsize: tuple = (16, 10),
                    show_day_boundaries: bool = False) -> str:
        """Shared chart drawing logic for both 5-min and 1-min charts"""
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)

        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        x_index = np.arange(len(df))

        # --- Subplot 1: Prices + Max Diff ---
        ax1.plot(x_index, df['SET50_Close'], label='SET50 Index',
                 color='#2E86AB', linewidth=linewidth)
        ax1.plot(x_index, df[f'{self.near_prefix}_Close'],
                 label=f'{self.near_prefix} ({self.near_label})',
                 color='#A23B72', linewidth=linewidth)
        ax1.plot(x_index, df[f'{self.far_prefix}_Close'],
                 label=f'{self.far_prefix} ({self.far_label})',
                 color='#F18F01', linewidth=linewidth)

        ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_title('Futures Prices & Maximum Spread Difference', fontsize=13, pad=10)

        ax2 = ax1.twinx()
        ax2.plot(x_index, df[f'{self.spread_prefix}_Max_Diff'], label='Max Diff',
                 color='#C73E1D', linewidth=linewidth, linestyle='--', alpha=0.7)
        ax2.set_ylabel('Max Difference', fontsize=12, fontweight='bold', color='#C73E1D')
        ax2.tick_params(axis='y', labelcolor='#C73E1D')
        ax2.legend(loc='upper right', fontsize=10)

        # --- Subplot 2: Basis Analysis ---
        ax3.plot(x_index, df[f'{self.spread_prefix}_Theoretical_Basis'],
                 label='Theoretical Basis', color='#06A77D', linewidth=linewidth)
        ax3.plot(x_index, df[f'{self.spread_prefix}_Actual_Basis'],
                 label='Actual Basis', color='#D62246', linewidth=linewidth)
        ax3.plot(x_index, df[f'{self.spread_prefix}_Mispricing'],
                 label='Mispricing', color='#F77F00', linewidth=linewidth, linestyle='--')

        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax3.set_xlabel('Trading Period', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Basis (points)', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_title('Calendar Spread Basis Analysis', fontsize=13, pad=10)

        # Annotations for latest values
        latest_idx = len(df) - 1
        latest_theoretical = df[f'{self.spread_prefix}_Theoretical_Basis'].iloc[-1]
        latest_actual = df[f'{self.spread_prefix}_Actual_Basis'].iloc[-1]
        latest_mispricing = df[f'{self.spread_prefix}_Mispricing'].iloc[-1]

        ax3.annotate(f'Theoretical\n{latest_theoretical:.2f}',
                     xy=(latest_idx, latest_theoretical), xytext=(10, 10),
                     textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#06A77D',
                               alpha=0.7, edgecolor='none'),
                     color='white', fontsize=9, fontweight='bold', ha='left')
        ax3.annotate(f'Actual\n{latest_actual:.2f}',
                     xy=(latest_idx, latest_actual), xytext=(10, -10),
                     textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#D62246',
                               alpha=0.7, edgecolor='none'),
                     color='white', fontsize=9, fontweight='bold', ha='left')
        ax3.annotate(f'Mispricing\n{latest_mispricing:.2f}',
                     xy=(latest_idx, latest_mispricing), xytext=(10, 0),
                     textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#F77F00',
                               alpha=0.7, edgecolor='none'),
                     color='white', fontsize=9, fontweight='bold', ha='left')

        # Day boundary lines (for intraday charts)
        if show_day_boundaries:
            dates = df['Timestamp'].dt.date
            for i in range(1, len(df)):
                if dates.iloc[i] != dates.iloc[i - 1]:
                    for ax in [ax1, ax3]:
                        ax.axvline(x=i, color='gray', linestyle='-',
                                   linewidth=0.8, alpha=0.5)

        # X-axis labels
        n_labels = min(n_labels, len(df))
        indices = np.linspace(0, len(df) - 1, n_labels, dtype=int)
        date_labels = [df['Timestamp'].iloc[i].strftime(date_fmt) for i in indices]

        for ax in [ax1, ax3]:
            ax.set_xlim(0, len(df) - 1)
            ax.set_xticks(indices)
            ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=8)

        plt.tight_layout()
        fig.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return img_path

    def create_chart_5min(self, filter_days: int = 40) -> str | None:
        """Create the 5-minute timeframe chart"""
        if self.df_5min is None or len(self.df_5min) == 0:
            print(f"  [5-min] Skipped: no data available for {self.spread_name}")
            return None

        df = self.df_5min.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)

        # Filter to last N calendar days
        date_range_days = (df['Timestamp'].max() - df['Timestamp'].min()).days
        if date_range_days > filter_days:
            cutoff_date = df['Timestamp'].max() - pd.Timedelta(days=filter_days)
            df = df[df['Timestamp'] >= cutoff_date].reset_index(drop=True)

        title = f'SET50 Calendar Spread Analysis ({self.spread_name}, 5-min)'
        img_path = os.path.join(OUTPUT_DIR, f'{self.spread_name}_5min_chart.png')

        return self._draw_chart(
            df=df, title=title, img_path=img_path,
            date_fmt='%Y-%m-%d', n_labels=10, linewidth=1.5,
            figsize=(16, 10), show_day_boundaries=False
        )

    def create_chart_1min(self, n_trading_days: int = 5) -> str | None:
        """
        Create the 1-minute intraday chart for the last N trading days.
        Returns None if fewer than n_trading_days are available.
        """
        if self.df_1min is None or len(self.df_1min) == 0:
            print(f"  [1-min] Skipped: no data available for {self.spread_name}")
            return None

        df_filtered = get_trading_days(self.df_1min, n_trading_days)
        if df_filtered is None:
            print(f"  [1-min] Skipped: fewer than {n_trading_days} trading days "
                  f"available for {self.spread_name}")
            return None

        df_filtered = df_filtered.sort_values('Timestamp').reset_index(drop=True)

        title = (f'SET50 Calendar Spread – Last {n_trading_days} Trading Days '
                 f'({self.spread_name}, 1-min)')
        img_path = os.path.join(OUTPUT_DIR, f'{self.spread_name}_1min_chart.png')

        return self._draw_chart(
            df=df_filtered, title=title, img_path=img_path,
            date_fmt='%m-%d %H:%M', n_labels=15, linewidth=1.0,
            figsize=(18, 10), show_day_boundaries=True
        )

    def run_analysis(self, send_telegram: bool = True) -> dict:
        """Run complete analysis and optionally send to Telegram"""
        print(f"\n{'='*60}")
        print(f"Analyzing {self.spread_name}")
        print(f"{'='*60}")

        # Load and calculate both timeframes
        self.load_and_calculate()

        # Find extremes (from 5-min data)
        extremes = None
        if self.df_5min is not None:
            extremes = self.find_extremes(self.df_5min)
            print(f"Max: {extremes['max_value']} at {extremes['max_timestamp']} "
                  f"(Type: {extremes['max_type']})")
            print(f"Min: {extremes['min_value']} at {extremes['min_timestamp']} "
                  f"(Type: {extremes['min_type']})")

        # Create charts
        chart_5min_path = self.create_chart_5min()
        if chart_5min_path:
            print(f"5-min chart saved: {chart_5min_path}")

        chart_1min_path = self.create_chart_1min(n_trading_days=5)
        if chart_1min_path:
            print(f"1-min chart saved: {chart_1min_path}")

        # Send to Telegram
        if send_telegram and extremes:
            msg = (f"\n{self.spread_name} Spread Analysis\n"
                   f"Max: {extremes['max_value']} at {extremes['max_timestamp']} "
                   f"(Type: {extremes['max_type']})\n"
                   f"Min: {extremes['min_value']} at {extremes['min_timestamp']} "
                   f"(Type: {extremes['min_type']})")
            send_message(msg)
            if chart_5min_path:
                send_photo(chart_5min_path)
            if chart_1min_path:
                send_photo(chart_1min_path)

        return {
            'spread_name': self.spread_name,
            'extremes': extremes,
            'chart_5min_path': chart_5min_path,
            'chart_1min_path': chart_1min_path
        }


# =============================================================================
# SPREAD CONFIGURATIONS
# =============================================================================
SPREAD_CONFIGS = [
    {
        'near_symbol': 'S50M2026',
        'far_symbol': 'S50U2026',
        'near_expiry': '2026-06-29',
        'far_expiry': '2026-09-29',
        'near_label': 'Jun',
        'far_label': 'Sep'
    }
]


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function"""
    print("=" * 60)
    print("SET50 Futures Calendar Spread Analysis")
    print(f"Started at: {dt.datetime.now()}")
    print("=" * 60)

    results = []

    for config in SPREAD_CONFIGS:
        try:
            analyzer = CalendarSpreadAnalyzer(
                near_symbol=config['near_symbol'],
                far_symbol=config['far_symbol'],
                near_expiry=config['near_expiry'],
                far_expiry=config['far_expiry'],
                near_label=config['near_label'],
                far_label=config['far_label']
            )
            result = analyzer.run_analysis(send_telegram=True)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {config['near_symbol']}-{config['far_symbol']}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Finished at: {dt.datetime.now()}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
