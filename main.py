"""
SET50 Futures Calendar Spread Analysis
- Per-spread Telegram bot token
- Stop running after "last run time" (16:00 Asia/Bangkok)
- Run both timeframes:
    * 5 (5-min) -> last 30 days
    * 1 (1-min) -> last 7 days
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# =============================================================================
# GLOBAL CONFIG
# =============================================================================
TZ = "Asia/Bangkok"

RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.017"))
DIVIDEND_YIELD = float(os.getenv("DIVIDEND_YIELD", "0.0373"))

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHAT_ID_DEFAULT = os.getenv("TELEGRAM_CHAT_ID", "7311904934")  # you can override per spread if needed

TIMEFRAME_SETTINGS = [
    {"timeframe": "5", "label": "5m", "chart_days": 30},
    {"timeframe": "1", "label": "1m", "chart_days": 7},
]

# =============================================================================
# TELEGRAM
# =============================================================================
def send_message(api_token: str, chat_id: str, message: str) -> bool:
    if not api_token or not chat_id:
        print("Warning: Telegram credentials not configured")
        return False

    api_url = f"https://api.telegram.org/bot{api_token}/sendMessage"
    try:
        r = requests.post(
            api_url,
            json={"chat_id": chat_id, "text": message},
            timeout=20,
        )
        print(f"Message sent: {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        print(f"Error sending message: {e}")
        return False


def send_photo(api_token: str, chat_id: str, image_path: str) -> bool:
    if not api_token or not chat_id:
        print("Warning: Telegram credentials not configured")
        return False

    api_url = f"https://api.telegram.org/bot{api_token}/sendPhoto"
    try:
        with open(image_path, "rb") as photo:
            r = requests.post(
                api_url,
                files={"photo": photo},
                data={"chat_id": chat_id},
                timeout=60,
            )
        print(f"Photo sent: {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        print(f"Error sending photo: {e}")
        return False


# =============================================================================
# HELPERS
# =============================================================================
def make_spread_name(near_symbol: str, far_symbol: str) -> str:
    """
    'S50G2026' + 'S50H2026' -> 'S50G26H26'
    """
    product = near_symbol[:3]
    near_month = near_symbol[3]
    near_yy = near_symbol[-2:]
    far_month = far_symbol[3]
    far_yy = far_symbol[-2:]
    return f"{product}{near_month}{near_yy}{far_month}{far_yy}"


def _to_bkk_timestamp(ts: pd.Series) -> pd.Series:
    """
    Robust conversion to tz-aware Asia/Bangkok timestamps.
    Handles:
      - datetime64 (naive or aware)
      - unix seconds / ms integers
      - strings
    """
    s = ts.copy()

    # If already datetime:
    if np.issubdtype(s.dtype, np.datetime64):
        dt = pd.to_datetime(s, errors="coerce")
        if getattr(dt.dt, "tz", None) is None:
            # assume UTC if naive (common for many APIs)
            dt = dt.dt.tz_localize("UTC")
        return dt.dt.tz_convert(TZ)

    # If numeric epoch:
    if pd.api.types.is_numeric_dtype(s):
        # guess unit by magnitude
        # ms epoch ~ 1e12+, seconds epoch ~ 1e9+
        val = pd.to_numeric(s, errors="coerce")
        unit = "ms" if val.dropna().median() > 1e11 else "s"
        dt = pd.to_datetime(val, unit=unit, utc=True, errors="coerce")
        return dt.dt.tz_convert(TZ)

    # Else string
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return dt.dt.tz_convert(TZ)


def parse_last_run_dt(date_str: str, hour_min: str = "16:00") -> pd.Timestamp:
    # e.g. "2026-02-26" + "16:00" in Asia/Bangkok
    return pd.Timestamp(f"{date_str} {hour_min}").tz_localize(TZ)


def now_bkk() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TZ)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(symbol: str, timeframe: str) -> pd.DataFrame:
    from price_loaders.tradingview import load_asset_price
    df = load_asset_price(symbol, 100000, timeframe, None)
    df = df.copy()
    df["time"] = _to_bkk_timestamp(df["time"])
    return df


def load_set50(timeframe: str) -> pd.DataFrame:
    df = load_data("SET50", timeframe)
    return df[["time", "close"]].rename(columns={"time": "Timestamp", "close": "SET50_Close"})


def load_futures(symbol: str, timeframe: str) -> pd.DataFrame:
    # symbol like 'S50G2026'
    df = load_data(f"TFEX:{symbol}", timeframe)
    prefix = symbol  # columns => S50G2026_Open, ...
    return df[["time", "open", "high", "low", "close"]].rename(
        columns={
            "time": "Timestamp",
            "open": f"{prefix}_Open",
            "high": f"{prefix}_High",
            "low": f"{prefix}_Low",
            "close": f"{prefix}_Close",
        }
    )


# =============================================================================
# SPREAD ANALYZER
# =============================================================================
class CalendarSpreadAnalyzer:
    def __init__(self, near_symbol: str, far_symbol: str, near_expiry: str, far_expiry: str, near_label: str, far_label: str):
        self.near_symbol = near_symbol
        self.far_symbol = far_symbol
        self.near_expiry = near_expiry
        self.far_expiry = far_expiry
        self.near_label = near_label
        self.far_label = far_label

        self.spread_name = make_spread_name(near_symbol, far_symbol)
        self.near_prefix = near_symbol
        self.far_prefix = far_symbol
        self.spread_prefix = self.spread_name

    def load_and_calculate(self, timeframe: str) -> pd.DataFrame:
        set50 = load_set50(timeframe)
        near = load_futures(self.near_symbol, timeframe)
        far = load_futures(self.far_symbol, timeframe)

        # align timestamps
        df = near.merge(far, on="Timestamp", how="inner")

        # Spread OHLC = far - near
        for col in ["Open", "High", "Low", "Close"]:
            df[f"{self.spread_prefix}_{col}"] = df[f"{self.far_prefix}_{col}"] - df[f"{self.near_prefix}_{col}"]

        # Spread max-diff components
        df[f"{self.spread_prefix}_Diff_High_High"] = df[f"{self.far_prefix}_High"] - df[f"{self.near_prefix}_High"]
        df[f"{self.spread_prefix}_Diff_High_Low"] = df[f"{self.far_prefix}_Low"] - df[f"{self.near_prefix}_High"]
        df[f"{self.spread_prefix}_Diff_Low_High"] = df[f"{self.far_prefix}_High"] - df[f"{self.near_prefix}_Low"]
        df[f"{self.spread_prefix}_Diff_Low_Low"] = df[f"{self.far_prefix}_Low"] - df[f"{self.near_prefix}_Low"]

        df[f"{self.spread_prefix}_Max_Diff"] = df[
            [
                f"{self.spread_prefix}_Diff_High_High",
                f"{self.spread_prefix}_Diff_High_Low",
                f"{self.spread_prefix}_Diff_Low_High",
                f"{self.spread_prefix}_Diff_Low_Low",
            ]
        ].max(axis=1)

        # Merge with SET50
        df = set50.merge(df, on="Timestamp", how="inner").sort_values("Timestamp").reset_index(drop=True)

        # Time-to-expiry (days)
        near_expiry_dt = pd.to_datetime(self.near_expiry).tz_localize(TZ)
        far_expiry_dt = pd.to_datetime(self.far_expiry).tz_localize(TZ)

        df[f"{self.near_prefix}_Time_to_expiry"] = ((near_expiry_dt - df["Timestamp"]).dt.total_seconds() / (24 * 3600)).astype(int)
        df[f"{self.far_prefix}_Time_to_expiry"] = ((far_expiry_dt - df["Timestamp"]).dt.total_seconds() / (24 * 3600)).astype(int)

        # Cost-of-carry expected futures
        df[f"{self.near_prefix}_Expected_Futures_Price"] = df["SET50_Close"] * np.exp(
            (RISK_FREE_RATE - DIVIDEND_YIELD) * (df[f"{self.near_prefix}_Time_to_expiry"] / 365)
        )
        df[f"{self.far_prefix}_Expected_Futures_Price"] = df["SET50_Close"] * np.exp(
            (RISK_FREE_RATE - DIVIDEND_YIELD) * (df[f"{self.far_prefix}_Time_to_expiry"] / 365)
        )

        # Basis
        df[f"{self.near_prefix}_Basis"] = df[f"{self.near_prefix}_Expected_Futures_Price"] - df["SET50_Close"]
        df[f"{self.far_prefix}_Basis"] = df[f"{self.far_prefix}_Expected_Futures_Price"] - df["SET50_Close"]

        df[f"{self.spread_prefix}_Theoretical_Basis"] = df[f"{self.far_prefix}_Basis"] - df[f"{self.near_prefix}_Basis"]
        df[f"{self.spread_prefix}_Actual_Basis"] = df[f"{self.spread_prefix}_Close"]
        df[f"{self.spread_prefix}_Mispricing"] = df[f"{self.spread_prefix}_Actual_Basis"] - df[f"{self.spread_prefix}_Theoretical_Basis"]

        self.df = df
        return df

    def _extremes_on_df(self, df: pd.DataFrame) -> dict:
        sp = self.spread_prefix
        if df.empty:
            return {"max_value": np.nan, "max_timestamp": None, "max_type": None,
                    "min_value": np.nan, "min_timestamp": None, "min_type": None}

        max_idx = df[f"{sp}_Max_Diff"].idxmax()
        min_idx = df[f"{sp}_Max_Diff"].idxmin()

        max_row = df.loc[max_idx]
        min_row = df.loc[min_idx]

        def _pick_type(row) -> str:
            if row[f"{sp}_Diff_High_High"] == row[f"{sp}_Max_Diff"]:
                return "H-H"
            if row[f"{sp}_Diff_High_Low"] == row[f"{sp}_Max_Diff"]:
                return "H-L"
            if row[f"{sp}_Diff_Low_High"] == row[f"{sp}_Max_Diff"]:
                return "L-H"
            return "L-L"

        return {
            "max_value": float(np.round(max_row[f"{sp}_Max_Diff"], 3)),
            "max_timestamp": max_row["Timestamp"],
            "max_type": _pick_type(max_row),
            "min_value": float(np.round(min_row[f"{sp}_Max_Diff"], 3)),
            "min_timestamp": min_row["Timestamp"],
            "min_type": _pick_type(min_row),
        }

    def _filter_last_days(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        if df.empty:
            return df
        max_ts = df["Timestamp"].max()
        cutoff = max_ts - pd.Timedelta(days=days)
        return df[df["Timestamp"] >= cutoff].copy()

    def create_chart(self, df: pd.DataFrame, timeframe_label: str, chart_days: int) -> str:
        sp = self.spread_prefix

        df = df.sort_values("Timestamp").reset_index(drop=True)
        df_plot = self._filter_last_days(df, chart_days)

        if len(df_plot) < 2:
            # still save a tiny chart with message
            fig = plt.figure(figsize=(12, 4))
            plt.title(f"{self.spread_name} ({timeframe_label}) - Not enough data")
            img_path = os.path.join(OUTPUT_DIR, f"{self.spread_name}_{timeframe_label}_chart.png")
            fig.savefig(img_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return img_path

        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle(
            f"SET50 Futures Calendar Spread Analysis ({self.spread_name}) [{timeframe_label}]",
            fontsize=16,
            fontweight="bold",
        )

        x = np.arange(len(df_plot))

        # Prices
        ax1.plot(x, df_plot["SET50_Close"], label="SET50 Index", linewidth=1.5)
        ax1.plot(x, df_plot[f"{self.near_prefix}_Close"], label=f"{self.near_prefix} ({self.near_label})", linewidth=1.5)
        ax1.plot(x, df_plot[f"{self.far_prefix}_Close"], label=f"{self.far_prefix} ({self.far_label})", linewidth=1.5)
        ax1.set_ylabel("Price", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.set_title("Futures Prices & Maximum Spread Difference", fontsize=13, pad=10)

        # Max diff on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(x, df_plot[f"{sp}_Max_Diff"], label="Max Diff", linewidth=1.5, linestyle="--", alpha=0.7)
        ax2.set_ylabel("Max Difference", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=10)

        # Basis
        ax3.plot(x, df_plot[f"{sp}_Theoretical_Basis"], label="Theoretical Basis", linewidth=1.5)
        ax3.plot(x, df_plot[f"{sp}_Actual_Basis"], label="Actual Basis", linewidth=1.5)
        ax3.plot(x, df_plot[f"{sp}_Mispricing"], label="Mispricing", linewidth=1.5, linestyle="--")
        ax3.axhline(y=0, linestyle="-", linewidth=0.8, alpha=0.6)
        ax3.set_xlabel("Trading Period", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Basis (points)", fontsize=12, fontweight="bold")
        ax3.legend(loc="best", fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle="--")
        ax3.set_title("Calendar Spread Basis Analysis", fontsize=13, pad=10)

        # X labels
        n_labels = min(10, len(df_plot))
        idxs = np.linspace(0, len(df_plot) - 1, n_labels, dtype=int)
        labels = [df_plot["Timestamp"].iloc[i].strftime("%Y-%m-%d") for i in idxs]
        for ax in [ax1, ax3]:
            ax.set_xlim(0, len(df_plot) - 1)
            ax.set_xticks(idxs)
            ax.set_xticklabels(labels, rotation=45, ha="right")

        plt.tight_layout()
        img_path = os.path.join(OUTPUT_DIR, f"{self.spread_name}_{timeframe_label}_chart.png")
        fig.savefig(img_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        return img_path

    def run_analysis(self, timeframe: str, timeframe_label: str, chart_days: int, telegram_token: str, telegram_chat_id: str, telegram_title: str) -> dict:
        print(f"\n{'='*70}")
        print(f"Analyzing {self.spread_name} | TF={timeframe_label} | last {chart_days}d")
        print(f"{'='*70}")

        df = self.load_and_calculate(timeframe=timeframe)

        # extremes computed on the same window as chart (more relevant)
        df_window = self._filter_last_days(df, chart_days)
        extremes = self._extremes_on_df(df_window)

        chart_path = self.create_chart(df=df, timeframe_label=timeframe_label, chart_days=chart_days)

        # latest values
        latest = df.iloc[-1]
        sp = self.spread_prefix
        msg = (
            f"{telegram_title}\n"
            f"Timeframe: {timeframe_label} | Window: last {chart_days} days\n\n"
            f"MaxDiff: {extremes['max_value']} at {extremes['max_timestamp']} (Type: {extremes['max_type']})\n"
            f"MinDiff: {extremes['min_value']} at {extremes['min_timestamp']} (Type: {extremes['min_type']})\n\n"
            f"Latest:\n"
            f"  Theoretical Basis: {latest[f'{sp}_Theoretical_Basis']:.2f}\n"
            f"  Actual Basis:      {latest[f'{sp}_Actual_Basis']:.2f}\n"
            f"  Mispricing:        {latest[f'{sp}_Mispricing']:.2f}\n"
            f"  Timestamp:         {latest['Timestamp']}"
        )

        ok1 = send_message(telegram_token, telegram_chat_id, msg)
        ok2 = send_photo(telegram_token, telegram_chat_id, chart_path)

        return {
            "spread_name": self.spread_name,
            "timeframe": timeframe,
            "timeframe_label": timeframe_label,
            "chart_days": chart_days,
            "extremes": extremes,
            "chart_path": chart_path,
            "telegram_message_sent": ok1,
            "telegram_photo_sent": ok2,
        }


# =============================================================================
# SPREAD CONFIGS (per-spread token via ENV)
# =============================================================================
SPREAD_CONFIGS = [
    {
        "near_symbol": "S50G2026",
        "far_symbol": "S50H2026",
        "near_expiry": "2026-02-26",
        "far_expiry": "2026-03-30",
        "near_label": "Feb",
        "far_label": "Mar",
        "spread_id": "S50G26H26",
        "telegram_title": "S50G26H26: Spread Trading Analysis",
        "telegram_token_env": "TELEGRAM_TOKEN_S50G26H26",
        "telegram_chat_id": CHAT_ID_DEFAULT,
        "last_run_date": "2026-02-26",
    },
    {
        "near_symbol": "S50H2026",
        "far_symbol": "S50M2026",
        "near_expiry": "2026-03-30",
        "far_expiry": "2026-06-29",
        "near_label": "Mar",
        "far_label": "Jun",
        "spread_id": "S50H26M26",
        "telegram_title": "S50H26M26: Spread Trading Analysis",
        "telegram_token_env": "TELEGRAM_TOKEN_S50H26M26",
        "telegram_chat_id": CHAT_ID_DEFAULT,
        "last_run_date": "2026-03-30",
    },
    {
        "near_symbol": "S50M2026",
        "far_symbol": "S50U2026",
        "near_expiry": "2026-06-29",
        "far_expiry": "2026-09-29",
        "near_label": "Jun",
        "far_label": "Sep",
        "spread_id": "S50M26U26",
        "telegram_title": "S50M26U26: Spread Trading Analysis",
        "telegram_token_env": "TELEGRAM_TOKEN_S50M26U26",
        "telegram_chat_id": CHAT_ID_DEFAULT,
        "last_run_date": "2026-06-29",
    },
    {
        "near_symbol": "S50U2026",
        "far_symbol": "S50Z2026",
        "near_expiry": "2026-09-29",
        "far_expiry": "2026-12-29",
        "near_label": "Sep",
        "far_label": "Dec",
        "spread_id": "S50U26Z26",
        "telegram_title": "S50U26Z26: Spread Trading Analysis",
        "telegram_token_env": "TELEGRAM_TOKEN_S50U26Z26",
        "telegram_chat_id": CHAT_ID_DEFAULT,
        "last_run_date": "2026-09-29",
    },
]


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("SET50 Futures Calendar Spread Analysis (Multi-Timeframe)")
    print(f"Started at: {now_bkk()}")
    print("=" * 70)

    results = []

    for cfg in SPREAD_CONFIGS:
        spread_id = cfg["spread_id"]
        last_run_dt = parse_last_run_dt(cfg["last_run_date"], "16:00")
        current = now_bkk()

        if current > last_run_dt:
            print(f"\n[SKIP] {spread_id} expired for reporting. Last run was {last_run_dt}. Now {current}.")
            continue

        telegram_token = os.getenv(cfg["telegram_token_env"], "").strip()
        telegram_chat_id = str(cfg.get("telegram_chat_id", CHAT_ID_DEFAULT)).strip()

        if not telegram_token:
            print(f"\n[WARN] Missing token for {spread_id}. Set env var: {cfg['telegram_token_env']}")
            # continue anyway? better to skip sending but still run analysis
            # We'll skip sending by not calling Telegram if missing.
            # Here, we skip entirely to match "send reports to Telegram".
            continue

        analyzer = CalendarSpreadAnalyzer(
            near_symbol=cfg["near_symbol"],
            far_symbol=cfg["far_symbol"],
            near_expiry=cfg["near_expiry"],
            far_expiry=cfg["far_expiry"],
            near_label=cfg["near_label"],
            far_label=cfg["far_label"],
        )

        # Run both timeframes
        for tf in TIMEFRAME_SETTINGS:
            try:
                out = analyzer.run_analysis(
                    timeframe=tf["timeframe"],
                    timeframe_label=tf["label"],
                    chart_days=tf["chart_days"],
                    telegram_token=telegram_token,
                    telegram_chat_id=telegram_chat_id,
                    telegram_title=cfg["telegram_title"],
                )
                results.append(out)
            except Exception as e:
                print(f"[ERROR] {spread_id} TF={tf['label']}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print(f"Finished at: {now_bkk()}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()
