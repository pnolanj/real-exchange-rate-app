import io
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Global Macroeconomic Analytics | Real Exchange Rate Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CONSTANTS & COUNTRY DICTIONARY (44 IMF Economies + Aggregates)
# -----------------------------------------------------------------------------
IMF_SDMX_30_BASE = "https://api.imf.org/external/sdmx/3.0"
IMF_SDMX_21_BASE = "https://api.imf.org/external/sdmx/2.1"
START_PERIOD_YM = "1976-01"
START_PERIOD_SDMX3 = "1976-M01"

COUNTRIES: Dict[str, str] = {
    "GBR": "United Kingdom",
    "USA": "United States",
    "G163": "Euro Area (Member States)",
    "JPN": "Japan",
    "CAN": "Canada",
    "CHE": "Switzerland",
    "AUS": "Australia",
    "NZL": "New Zealand",
    "SWE": "Sweden",
    "NOR": "Norway",
    "DNK": "Denmark",
    "DEU": "Germany",
    "FRA": "France",
    "ITA": "Italy",
    "ESP": "Spain",
    "NLD": "Netherlands",
    "BEL": "Belgium",
    "AUT": "Austria",
    "IRL": "Ireland",
    "FIN": "Finland",
    "PRT": "Portugal",
    "GRC": "Greece",
    "POL": "Poland",
    "CZE": "Czechia",
    "HUN": "Hungary",
    "KOR": "Korea, Republic of",
    "SGP": "Singapore",
    "HKG": "Hong Kong SAR",
    "CHN": "China, P.R.: Mainland",
    "IND": "India",
    "IDN": "Indonesia",
    "THA": "Thailand",
    "MYS": "Malaysia",
    "PHL": "Philippines",
    "VNM": "Vietnam",
    "MEX": "Mexico",
    "BRA": "Brazil",
    "CHL": "Chile",
    "COL": "Colombia",
    "PER": "Peru",
    "ZAF": "South Africa",
    "TUR": "Türkiye",
    "ISR": "Israel",
    "SAU": "Saudi Arabia",
}


# -----------------------------------------------------------------------------
# INSTITUTIONAL DESIGN SYSTEM & CSS INJECTION
# -----------------------------------------------------------------------------
def inject_institutional_css() -> None:
    """Inject enterprise-grade CSS styling for institutional typography, cards, and layout density."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Global Canvas & Typography */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #0F172A;
            background-color: #F8F9FA;
        }

        .stApp {
            background-color: #F8F9FA;
        }

        /* Tight Enterprise Container Density */
        .block-container {
            padding-top: 1.25rem !important;
            padding-bottom: 2.5rem !important;
            padding-left: 2.25rem !important;
            padding-right: 2.25rem !important;
            max-width: 1600px;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 1.5rem !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }

        .sidebar-section-label {
            font-size: 0.70rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: #475569;
            margin-top: 0.75rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 0.35rem;
        }

        /* Executive Top Banner */
        .exec-banner {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #0A192F;
            border-radius: 4px;
            padding: 1.15rem 1.5rem;
            margin-bottom: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }

        .exec-title-group {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }

        .exec-eyebrow {
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            color: #475569;
        }

        .exec-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0A192F;
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.25;
        }

        .exec-badge-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .exec-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 500;
            color: #1E293B;
            background-color: #F1F5F9;
            border: 1px solid #CBD5E1;
            border-radius: 3px;
            padding: 0.3rem 0.65rem;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }

        .exec-badge-accent {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            color: #2563EB;
            background-color: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-radius: 3px;
            padding: 0.3rem 0.65rem;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }

        /* Structured KPI Summary Cards */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 1.25rem;
        }

        @media (max-width: 1100px) {
            .kpi-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 640px) {
            .kpi-grid {
                grid-template-columns: 1fr;
            }
        }

        .kpi-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 4px;
            padding: 1rem 1.25rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .kpi-label {
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #475569;
            margin-bottom: 0.45rem;
        }

        .kpi-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #0A192F;
            font-variant-numeric: tabular-nums;
            letter-spacing: -0.02em;
            line-height: 1.2;
            margin-bottom: 0.4rem;
        }

        .kpi-sub {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: #64748B;
            font-variant-numeric: tabular-nums;
            border-top: 1px solid #F1F5F9;
            padding-top: 0.4rem;
        }

        /* Institutional Panels & Containers */
        .inst-panel {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 4px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }

        .panel-header {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #1E293B;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 0.6rem;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Streamlit Tabs Overrides */
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            font-size: 0.80rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            color: #475569 !important;
            padding: 0.65rem 1.25rem !important;
            background-color: transparent !important;
        }

        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            color: #0A192F !important;
            border-bottom: 2px solid #2563EB !important;
        }

        /* Tabular Numerals for Metrics & Tables */
        div[data-testid="stMetricValue"],
        div[data-testid="stDataFrame"] {
            font-variant-numeric: tabular-nums !important;
        }

        /* Secondary Summary Bar under Chart */
        .stat-strip {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.75rem;
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 4px;
            padding: 0.85rem 1.25rem;
            margin-top: 0.5rem;
            font-variant-numeric: tabular-nums;
        }

        .stat-item-label {
            font-size: 0.65rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748B;
        }

        .stat-item-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.92rem;
            font-weight: 600;
            color: #0F172A;
            margin-top: 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# DATE NORMALIZATION UTILITIES
# -----------------------------------------------------------------------------
def normalize_ym(raw_date: str) -> Optional[str]:
    """
    Convert SDMX monthly period representations ('1976-M01', '1976-01',
    '1976-01-01', '1976M01') into canonical 'YYYY-MM' strings.
    Returns None if the string does not represent a valid monthly date.
    """
    if raw_date is None:
        return None
    s = str(raw_date).strip().upper()
    if not s:
        return None

    # Handle SDMX 3.0 monthly format: YYYY-M01 or YYYYM01
    s = s.replace("-M", "-")
    if len(s) == 7 and "M" in s:
        s = s.replace("M", "-")

    # Keep first 7 characters if YYYY-MM-DD
    if len(s) >= 7 and s[4] == "-":
        ym = s[:7]
        parts = ym.split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            year = int(parts[0])
            month = int(parts[1])
            if 1900 <= year <= 2100 and 1 <= month <= 12:
                return f"{year:04d}-{month:02d}"
    return None


def build_usa_fx_series(start_ym: str = START_PERIOD_YM) -> pd.Series:
    """
    Generate a deterministic monthly series of 1.0 for USA (USD per 1 USD)
    from start_ym through the current month.
    """
    end_ym = datetime.utcnow().strftime("%Y-%m")
    date_index = pd.date_range(start=f"{start_ym}-01", end=f"{end_ym}-01", freq="MS")
    ym_index = [d.strftime("%Y-%m") for d in date_index]
    return pd.Series(
        data=1.0, index=pd.Index(ym_index, name="Date"), name="FX_USD", dtype=float
    )


# -----------------------------------------------------------------------------
# RESILIENT IMF SDMX 3.0 PARSERS (CSV, JSON, XML)
# -----------------------------------------------------------------------------
def _parse_sdmx_csv(text: str) -> Optional[pd.Series]:
    """Parse SDMX-CSV response into a monthly pd.Series indexed by 'YYYY-MM'."""
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return None

    if df.empty:
        return None

    cols_upper = {c.upper(): c for c in df.columns}
    time_col = cols_upper.get("TIME_PERIOD") or cols_upper.get("TIME")
    val_col = cols_upper.get("OBS_VALUE") or cols_upper.get("VALUE")

    if not time_col or not val_col:
        return None

    sub = df[[time_col, val_col]].copy()
    sub["Date"] = sub[time_col].apply(normalize_ym)
    sub["Value"] = pd.to_numeric(sub[val_col], errors="coerce")
    sub = sub.dropna(subset=["Date", "Value"])
    sub = sub[sub["Date"] >= START_PERIOD_YM]

    if sub.empty:
        return None

    sub = sub.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return pd.Series(
        sub["Value"].values,
        index=pd.Index(sub["Date"].values, name="Date"),
        dtype=float,
    )


def _parse_sdmx_json(payload: dict) -> Optional[pd.Series]:
    """
    Parse SDMX-JSON (both 2.0.0 and 1.0 schemas) into a monthly pd.Series
    indexed by 'YYYY-MM'.
    """
    try:
        root = payload.get("data", payload)
        datasets = root.get("dataSets", [])
        if not datasets:
            return None

        series_dict = datasets[0].get("series", {})
        if not series_dict:
            return None

        # Locate observation time dimension values across SDMX-JSON 2.0 and 1.0
        structures = root.get("structures") or []
        structure = root.get("structure") or (structures[0] if structures else {})
        dimensions = structure.get("dimensions", {})
        obs_dims = dimensions.get("observation", [])

        time_values: List[str] = []
        for dim in obs_dims:
            if dim.get("id", "").upper() in ("TIME_PERIOD", "TIME") or "values" in dim:
                time_values = [
                    v.get("id") or v.get("name") for v in dim.get("values", [])
                ]
                if time_values:
                    break

        if not time_values:
            return None

        records: Dict[str, float] = {}
        for _, s_obj in series_dict.items():
            observations = s_obj.get("observations", {})
            for obs_idx_str, obs_val_list in observations.items():
                idx = int(obs_idx_str)
                if 0 <= idx < len(time_values) and obs_val_list:
                    raw_val = obs_val_list[0]
                    if raw_val is not None:
                        ym = normalize_ym(time_values[idx])
                        try:
                            f_val = float(raw_val)
                        except (TypeError, ValueError):
                            continue
                        if ym and ym >= START_PERIOD_YM and np.isfinite(f_val):
                            records[ym] = f_val

        if not records:
            return None

        s = pd.Series(records, dtype=float)
        s.index.name = "Date"
        return s.sort_index()
    except Exception:
        return None


def _parse_sdmx_xml(text: str) -> Optional[pd.Series]:
    """Parse SDMX-ML (StructureSpecific or Generic XML) into a monthly pd.Series."""
    try:
        root = ET.fromstring(text)
    except Exception:
        return None

    records: Dict[str, float] = {}
    for elem in root.iter():
        tag_ordered = elem.tag.split("}")[-1].lower()
        if tag_ordered == "obs":
            raw_t = elem.attrib.get("TIME_PERIOD") or elem.attrib.get("time_period")
            raw_v = elem.attrib.get("OBS_VALUE") or elem.attrib.get("obs_value")

            if raw_t is None or raw_v is None:
                for child in elem:
                    c_tag = child.tag.split("}")[-1].lower()
                    if c_tag == "obsdimension":
                        raw_t = child.attrib.get("value")
                    elif c_tag == "obsvalue":
                        raw_v = child.attrib.get("value")

            ym = normalize_ym(raw_t)
            if ym and ym >= START_PERIOD_YM and raw_v is not None:
                try:
                    val = float(raw_v)
                    if np.isfinite(val):
                        records[ym] = val
                except ValueError:
                    continue

    if not records:
        return None

    s = pd.Series(records, dtype=float)
    s.index.name = "Date"
    return s.sort_index()


# -----------------------------------------------------------------------------
# CACHED IMF SDMX 3.0 API CLIENT
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_imf_series(iso3: str, indicator_type: str) -> Tuple[pd.Series, str]:
    """
    Fetch monthly CPI or FX series from the IMF SDMX 3.0 API.

    Parameters
    ----------
    iso3 : str
        3-letter ISO country code (or IMF aggregate code such as 'G163').
    indicator_type : str
        Either 'CPI' (for {ISO3}.CPI._T.IX.M) or 'FX' (for {ISO3}.XDC_USD.PA_RT.M).

    Returns
    -------
    Tuple[pd.Series, str]
        - Monthly pandas Series indexed by 'YYYY-MM'
        - Diagnostic source URL / status string
    """
    iso3_clean = iso3.strip().upper()

    # CRITICAL USA EDGE CASE: USA FX per 1 USD is strictly 1.0 for all months
    if indicator_type == "FX" and iso3_clean == "USA":
        return (
            build_usa_fx_series(START_PERIOD_YM),
            "Deterministic Invariant: USA.XDC_USD.PA_RT.M = 1.0000",
        )

    if indicator_type == "CPI":
        series_key = f"{iso3_clean}.CPI._T.IX.M"
        candidate_dataflows = ["CPI", "IFS"]
    elif indicator_type == "FX":
        series_key = f"{iso3_clean}.XDC_USD.PA_RT.M"
        candidate_dataflows = ["ER", "IFS"]
    else:
        raise ValueError(f"Unsupported indicator_type: {indicator_type}")

    # Build prioritized endpoint candidates across IMF SDMX 3.0 and 2.1 surfaces
    candidate_urls: List[str] = []
    for df_id in candidate_dataflows:
        candidate_urls.extend(
            [
                f"{IMF_SDMX_30_BASE}/data/dataflow/IMF.STA/{df_id}/~/{series_key}?c[TIME_PERIOD]=ge:{START_PERIOD_SDMX3}",
                f"{IMF_SDMX_30_BASE}/data/dataflow/IMF.STA/{df_id}/+/{series_key}?startPeriod={START_PERIOD_YM}",
                f"{IMF_SDMX_30_BASE}/data/IMF.STA,{df_id},~/{series_key}?startPeriod={START_PERIOD_YM}",
                f"{IMF_SDMX_21_BASE}/data/IMF.STA,{df_id}/{series_key}?startPeriod={START_PERIOD_YM}",
            ]
        )

    accept_profiles = [
        ("csv", "text/csv, application/vnd.sdmx.data+csv;version=2.0.0;q=0.9"),
        (
            "json",
            "application/vnd.sdmx.data+json;version=2.0.0, application/json;q=0.9",
        ),
        (
            "xml",
            "application/vnd.sdmx.structurespecificdata+xml;version=2.1, application/xml;q=0.9",
        ),
    ]

    session = requests.Session()
    base_headers = {
        "User-Agent": "IMF-RER-MacroAnalyzer/2.0 (Institutional Edition; Python 3.12)",
        "Accept-Encoding": "gzip, deflate",
    }

    diagnostics: List[str] = []

    for url in candidate_urls:
        for fmt, accept_header in accept_profiles:
            headers = {**base_headers, "Accept": accept_header}
            try:
                resp = session.get(url, headers=headers, timeout=20)
            except requests.RequestException as exc:
                diagnostics.append(
                    f"{url} [{fmt}] -> Network error: {exc.__class__.__name__}"
                )
                continue

            if resp.status_code != 200 or not resp.text.strip():
                diagnostics.append(f"{url} [{fmt}] -> HTTP {resp.status_code}")
                continue

            content_type = resp.headers.get("Content-Type", "").lower()
            parsed: Optional[pd.Series] = None

            if fmt == "csv" or "csv" in content_type:
                parsed = _parse_sdmx_csv(resp.text)
            if parsed is None and (fmt == "json" or "json" in content_type):
                try:
                    parsed = _parse_sdmx_json(resp.json())
                except Exception:
                    parsed = None
            if parsed is None and (
                fmt == "xml"
                or "xml" in content_type
                or resp.text.lstrip().startswith("<")
            ):
                parsed = _parse_sdmx_xml(resp.text)

            if parsed is not None and not parsed.empty:
                parsed.name = f"{iso3_clean}_{indicator_type}"
                return parsed, url

    error_summary = (
        "; ".join(diagnostics[:4])
        if diagnostics
        else "No response from IMF SDMX endpoints."
    )
    raise RuntimeError(
        f"Could not retrieve valid monthly observations for `{series_key}` from IMF SDMX 3.0 API. "
        f"Diagnostics: {error_summary}"
    )


# -----------------------------------------------------------------------------
# CORE ECONOMIC & MATHEMATICAL ENGINE
# -----------------------------------------------------------------------------
def assemble_base_dataframe(
    iso1: str,
    iso2: str,
    fx1: pd.Series,
    fx2: pd.Series,
    cpi1: pd.Series,
    cpi2: pd.Series,
) -> pd.DataFrame:
    """
    Merge FX1(t), FX2(t), P1(t), and P2(t) on monthly dates ('YYYY-MM') from
    1976-01 onward, handle the USA edge case, compute S(t) and RER(t), and drop
    any months with missing, NaN, or non-positive values.
    """
    iso1_clean = iso1.strip().upper()
    iso2_clean = iso2.strip().upper()

    # Determine unified monthly index across CPI series
    cpi_union_idx = cpi1.index.union(cpi2.index)

    # Enforce strict USA FX = 1.0 aligned to all candidate monthly dates
    if iso1_clean == "USA":
        fx1 = pd.Series(
            1.0,
            index=cpi_union_idx.union(fx2.index),
            name="FX1_per_USD",
            dtype=float,
        )
    if iso2_clean == "USA":
        fx2 = pd.Series(
            1.0,
            index=cpi_union_idx.union(fx1.index),
            name="FX2_per_USD",
            dtype=float,
        )

    df = pd.DataFrame(
        {
            "FX1_per_USD": fx1,
            "FX2_per_USD": fx2,
            "CPI_1": cpi1,
            "CPI_2": cpi2,
        }
    )
    df.index.name = "Date"
    df = df.reset_index()

    # Filter from 1976-01 onward
    df = df[df["Date"] >= START_PERIOD_YM].copy()

    # Coerce numeric types
    for col in ["FX1_per_USD", "FX2_per_USD", "CPI_1", "CPI_2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop missing or non-positive observations prior to computing S(t)
    df = df.dropna(subset=["FX1_per_USD", "FX2_per_USD", "CPI_1", "CPI_2"])
    df = df[
        (df["FX1_per_USD"] > 0)
        & (df["FX2_per_USD"] > 0)
        & (df["CPI_1"] > 0)
        & (df["CPI_2"] > 0)
    ].copy()

    # S(t) = FX2(t) / FX1(t) [Units of Country 2 currency per 1 unit of Country 1 currency]
    df["Nominal_S_Ccy2_per_Ccy1"] = df["FX2_per_USD"] / df["FX1_per_USD"]

    # Drop any invalid or non-positive S(t), P1(t), P2(t)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Nominal_S_Ccy2_per_Ccy1", "CPI_1", "CPI_2"]
    )
    df = df[df["Nominal_S_Ccy2_per_Ccy1"] > 0].copy()

    # RER(t) = S(t) * (P1(t) / P2(t))
    df["RER"] = df["Nominal_S_Ccy2_per_Ccy1"] * (df["CPI_1"] / df["CPI_2"])

    df["Country_1_ISO"] = iso1_clean
    df["Country_2_ISO"] = iso2_clean

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def filter_and_normalize_sample(
    df_valid: pd.DataFrame, start_month: str, end_month: str
) -> pd.DataFrame:
    """
    1. Filter valid dataframe to [start_month, end_month] inclusive FIRST.
    2. Compute RER* = arithmetic mean of RER(t) over [start_month, end_month].
    3. Compute Real_Index(t) = 100 * (RER(t) / RER*).
    """
    sample = df_valid[
        (df_valid["Date"] >= start_month) & (df_valid["Date"] <= end_month)
    ].copy()

    if sample.empty:
        return sample

    rer_star = float(sample["RER"].mean())
    sample["RER_Star_Sample_Mean"] = rer_star
    sample["Real_Index"] = 100.0 * (sample["RER"] / rer_star)

    # Order columns strictly per export specification
    ordered_cols = [
        "Date",
        "Country_1_ISO",
        "Country_2_ISO",
        "FX1_per_USD",
        "FX2_per_USD",
        "Nominal_S_Ccy2_per_Ccy1",
        "CPI_1",
        "CPI_2",
        "RER",
        "RER_Star_Sample_Mean",
        "Real_Index",
    ]
    return sample[ordered_cols].reset_index(drop=True)


# -----------------------------------------------------------------------------
# INSTITUTIONAL PLOTLY CHARTING ENGINE
# -----------------------------------------------------------------------------
def build_exchange_rate_chart(
    sample_df: pd.DataFrame,
    iso1: str,
    iso2: str,
    name1: str,
    name2: str,
    display_mode: str,
    start_month: str,
    end_month: str,
) -> go.Figure:
    """
    Build institutional Plotly figure for 'Real', 'Nominal', or 'Both' modes.
    - Real Index line: Deep Navy (#0F172A, width 2.5)
    - Nominal S(t) line: Muted Slate/Steel (#64748B, width 2.0)
    - Period Mean reference line: Dashed Neutral Silver (#94A3B8)
    - Horizontal-only subtle grid lines (#F1F5F9); no vertical grid lines.
    """
    plot_dates = pd.to_datetime(sample_df["Date"] + "-01")
    real_y_title = "Real Exchange Rate Index (Period Mean = 100.00)"
    nom_y_title = f"Nominal Rate S(t) ({iso2} per 1 {iso1})"

    common_hoverlabel = dict(
        bgcolor="#0A192F",
        bordercolor="#1E293B",
        font=dict(
            family="JetBrains Mono, Inter, monospace",
            size=12,
            color="#F8FAFC",
        ),
    )

    common_xaxis = dict(
        title=dict(text="", font=dict(size=11, color="#475569")),
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="#CBD5E1",
        linewidth=1,
        ticks="outside",
        tickcolor="#CBD5E1",
        tickfont=dict(
            family="Inter, sans-serif",
            size=11,
            color="#475569",
        ),
        hoverformat="%Y-%m",
    )

    if display_mode == "Real":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=sample_df["Real_Index"],
                mode="lines",
                name=f"Real Exchange Rate Index ({iso1}/{iso2})",
                line=dict(color="#0F172A", width=2.5),
                hovertemplate="Real Index (Mean=100): <b>%{y:.4f}</b><extra></extra>",
            )
        )
        fig.add_hline(
            y=100.0,
            line_dash="dash",
            line_color="#94A3B8",
            line_width=1.5,
            annotation_text="Sample Mean Benchmark (100.0000)",
            annotation_position="bottom right",
            annotation_font=dict(
                family="JetBrains Mono, monospace",
                size=11,
                color="#475569",
            ),
        )
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>Bilateral Real Exchange Rate Index</b>  "
                    f"<span style='color:#64748B;font-size:13px;'>"
                    f"{name1} ({iso1}) vs. {name2} ({iso2})  |  {start_month} — {end_month}"
                    f"</span>"
                ),
                font=dict(family="Inter, sans-serif", size=15, color="#0A192F"),
                x=0.0,
                xanchor="left",
            ),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            hovermode="x unified",
            hoverlabel=common_hoverlabel,
            xaxis=common_xaxis,
            yaxis=dict(
                title=dict(
                    text=real_y_title,
                    font=dict(family="Inter, sans-serif", size=12, color="#0F172A"),
                ),
                showgrid=True,
                gridcolor="#F1F5F9",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor="#E2E8F0",
                tickfont=dict(
                    family="JetBrains Mono, monospace",
                    size=11,
                    color="#0F172A",
                ),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                font=dict(family="Inter, sans-serif", size=12, color="#1E293B"),
            ),
            height=500,
            margin=dict(l=65, r=35, t=75, b=45),
        )

    elif display_mode == "Nominal":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=sample_df["Nominal_S_Ccy2_per_Ccy1"],
                mode="lines",
                name=f"Nominal Spot Cross-Rate S(t): {iso2} per 1 {iso1}",
                line=dict(color="#64748B", width=2.0),
                hovertemplate=(
                    f"Nominal S(t): <b>%{{y:.4f}}</b> {iso2} per 1 {iso1}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>Bilateral Nominal Exchange Rate S(t)</b>  "
                    f"<span style='color:#64748B;font-size:13px;'>"
                    f"Units of {iso2} Currency per 1 Unit of {iso1} Currency  |  {start_month} — {end_month}"
                    f"</span>"
                ),
                font=dict(family="Inter, sans-serif", size=15, color="#0A192F"),
                x=0.0,
                xanchor="left",
            ),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            hovermode="x unified",
            hoverlabel=common_hoverlabel,
            xaxis=common_xaxis,
            yaxis=dict(
                title=dict(
                    text=nom_y_title,
                    font=dict(family="Inter, sans-serif", size=12, color="#475569"),
                ),
                showgrid=True,
                gridcolor="#F1F5F9",
                gridwidth=1,
                zeroline=False,
                showline=True,
                linecolor="#E2E8F0",
                tickfont=dict(
                    family="JetBrains Mono, monospace",
                    size=11,
                    color="#475569",
                ),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                font=dict(family="Inter, sans-serif", size=12, color="#1E293B"),
            ),
            height=500,
            margin=dict(l=65, r=35, t=75, b=45),
        )

    else:  # "Both" — Dual-Axis Institutional Configuration
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=sample_df["Real_Index"],
                mode="lines",
                name="Real Exchange Rate Index (Left Axis, Mean = 100)",
                line=dict(color="#0F172A", width=2.5),
                hovertemplate="Real Index (Mean=100): <b>%{y:.4f}</b><extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=sample_df["Nominal_S_Ccy2_per_Ccy1"],
                mode="lines",
                name=f"Nominal S(t): {iso2} per 1 {iso1} (Right Axis)",
                line=dict(color="#64748B", width=2.0),
                hovertemplate=(
                    f"Nominal S(t) ({iso2}/{iso1}): <b>%{{y:.4f}}</b><extra></extra>"
                ),
            ),
            secondary_y=True,
        )
        fig.add_hline(
            y=100.0,
            line_dash="dash",
            line_color="#94A3B8",
            line_width=1.5,
            secondary_y=False,
            annotation_text="Period Mean (100.00)",
            annotation_position="bottom right",
            annotation_font=dict(
                family="JetBrains Mono, monospace",
                size=10,
                color="#64748B",
            ),
        )
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>Real Exchange Rate Index vs. Nominal Cross-Rate S(t)</b>  "
                    f"<span style='color:#64748B;font-size:13px;'>"
                    f"{name1} ({iso1}) / {name2} ({iso2})  |  {start_month} — {end_month}"
                    f"</span>"
                ),
                font=dict(family="Inter, sans-serif", size=15, color="#0A192F"),
                x=0.0,
                xanchor="left",
            ),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            hovermode="x unified",
            hoverlabel=common_hoverlabel,
            xaxis=common_xaxis,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="left",
                x=0.0,
                bgcolor="rgba(255,255,255,0.9)",
                font=dict(family="Inter, sans-serif", size=12, color="#0F172A"),
            ),
            height=520,
            margin=dict(l=65, r=65, t=85, b=45),
        )
        fig.update_yaxes(
            title_text=real_y_title,
            title_font=dict(family="Inter, sans-serif", size=12, color="#0F172A"),
            tickfont=dict(
                family="JetBrains Mono, monospace", size=11, color="#0F172A"
            ),
            showgrid=True,
            gridcolor="#F1F5F9",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor="#E2E8F0",
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text=nom_y_title,
            title_font=dict(family="Inter, sans-serif", size=12, color="#64748B"),
            tickfont=dict(
                family="JetBrains Mono, monospace", size=11, color="#64748B"
            ),
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="#E2E8F0",
            secondary_y=True,
        )

    return fig


# -----------------------------------------------------------------------------
# AUDIT & VERIFICATION RENDERING ENGINE
# -----------------------------------------------------------------------------
def render_observation_verification(
    row: pd.Series, title_label: str, iso1: str, iso2: str, n_obs: int
) -> None:
    """Render institutional step-by-step arithmetic audit proof for a single monthly observation."""
    t_str = row["Date"]
    fx1 = float(row["FX1_per_USD"])
    fx2 = float(row["FX2_per_USD"])
    s_t = float(row["Nominal_S_Ccy2_per_Ccy1"])
    p1 = float(row["CPI_1"])
    p2 = float(row["CPI_2"])
    rer_t = float(row["RER"])
    rer_star = float(row["RER_Star_Sample_Mean"])
    real_idx = float(row["Real_Index"])

    st.markdown(
        f'<div class="panel-header">'
        f"<span>{title_label}</span>"
        f'<span style="font-family:\'JetBrains Mono\',monospace;color:#2563EB;">OBS PERIOD: {t_str}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"**1. Raw IMF USD Exchange Rate Series** (`{{ISO3}}.XDC_USD.PA_RT.M`)\n"
        f"$$FX_1(\\text{{{t_str}}}) = {fx1:.6f}\\text{{ ({iso1}/USD)}}, \\qquad "
        f"FX_2(\\text{{{t_str}}}) = {fx2:.6f}\\text{{ ({iso2}/USD)}}$$\n\n"
        f"**2. Bilateral Nominal Cross-Rate Construction** ($S(t)$ in units of `{iso2}` per 1 unit of `{iso1}`)\n"
        f"$$S(\\text{{{t_str}}}) = \\frac{{FX_2(\\text{{{t_str}}})}}{{FX_1(\\text{{{t_str}}})}} "
        f"= \\frac{{{fx2:.6f}}}{{{fx1:.6f}}} = {s_t:.6f}$$\n\n"
        f"**3. Monthly Consumer Price Index Inputs** (`{{ISO3}}.CPI._T.IX.M`)\n"
        f"$$P_1(\\text{{{t_str}}}) = {p1:.4f}\\text{{ ({iso1} CPI)}}, \\qquad "
        f"P_2(\\text{{{t_str}}}) = {p2:.4f}\\text{{ ({iso2} CPI)}}$$\n\n"
        f"**4. Bilateral Real Exchange Rate Level** ($RER(t)$)\n"
        f"$$RER(\\text{{{t_str}}}) = S(\\text{{{t_str}}}) \\times \\frac{{P_1(\\text{{{t_str}}})}}{{P_2(\\text{{{t_str}}})}} "
        f"= {s_t:.6f} \\times \\frac{{{p1:.4f}}}{{{p2:.4f}}} = {rer_t:.6f}$$\n\n"
        f"**5. Dynamic Sample-Window Index Normalization** ($N = {n_obs:,}$ months, $RER^* = {rer_star:.6f}$)\n"
        f"$$\\text{{Real Index}}(\\text{{{t_str}}}) = 100 \\times \\frac{{RER(\\text{{{t_str}}})}}{{RER^*}} "
        f"= 100 \\times \\frac{{{rer_t:.6f}}}{{{rer_star:.6f}}} = {real_idx:.2f}\\;({real_idx:.4f})$$"
    )


# -----------------------------------------------------------------------------
# MAIN APPLICATION WORKSPACE
# -----------------------------------------------------------------------------
def main() -> None:
    inject_institutional_css()

    # --- Top Executive Banner ---
    st.markdown(
        """
        <div class="exec-banner">
            <div class="exec-title-group">
                <span class="exec-eyebrow">Cross-Border FX &amp; Relative Price Valuation</span>
                <h1 class="exec-title">Global Macroeconomic Analytics &nbsp;|&nbsp; Real Exchange Rate Intelligence</h1>
            </div>
            <div class="exec-badge-group">
                <span class="exec-badge">Data Source: IMF SDMX 3.0 IFS</span>
                <span class="exec-badge">Base Period: 1976–Present</span>
                <span class="exec-badge-accent">Window Normalization: Mean = 100.00</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    country_options = [f"{code} - {name}" for code, name in COUNTRIES.items()]
    default_c1_idx = list(COUNTRIES.keys()).index("GBR")
    default_c2_idx = list(COUNTRIES.keys()).index("USA")

    # --- Sidebar Institutional Controls ---
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-section-label">Bilateral Currency Specification</div>',
            unsafe_allow_html=True,
        )

        c1_selection = st.selectbox(
            "Country 1 — Base Currency (ISO3_1)",
            options=country_options,
            index=default_c1_idx,
            help="Base economy: denominator of nominal cross-rate S(t) and numerator of relative price ratio P1(t)/P2(t).",
        )
        c2_selection = st.selectbox(
            "Country 2 — Quote Currency (ISO3_2)",
            options=country_options,
            index=default_c2_idx,
            help="Quote/Counter economy: numerator of nominal cross-rate S(t) and denominator of relative price ratio P1(t)/P2(t).",
        )

        with st.expander("Custom ISO3 Code Override", expanded=False):
            custom_iso1 = (
                st.text_input("Override Country 1 ISO3", value="").strip().upper()
            )
            custom_iso2 = (
                st.text_input("Override Country 2 ISO3", value="").strip().upper()
            )

        iso1 = custom_iso1 if len(custom_iso1) >= 3 else c1_selection.split(" - ")[0]
        iso2 = custom_iso2 if len(custom_iso2) >= 3 else c2_selection.split(" - ")[0]
        name1 = COUNTRIES.get(iso1, iso1)
        name2 = COUNTRIES.get(iso2, iso2)

        st.markdown(
            '<div class="sidebar-section-label">SDMX 3.0 Series Keys</div>',
            unsafe_allow_html=True,
        )
        st.code(
            f"CPI_1 : {iso1}.CPI._T.IX.M\n"
            f"CPI_2 : {iso2}.CPI._T.IX.M\n"
            f"FX_1  : {iso1}.XDC_USD.PA_RT.M\n"
            f"FX_2  : {iso2}.XDC_USD.PA_RT.M",
            language="text",
        )

        st.markdown(
            '<div class="sidebar-section-label">Cache &amp; Session Management</div>',
            unsafe_allow_html=True,
        )
        if st.button("Refresh IMF SDMX 3.0 Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # --- Data Retrieval from IMF SDMX 3.0 API ---
    with st.spinner(
        f"Synchronizing monthly CPI and FX time series for {iso1} and {iso2} via IMF SDMX 3.0 API..."
    ):
        try:
            cpi1_s, cpi1_src = fetch_imf_series(iso1, "CPI")
            cpi2_s, cpi2_src = fetch_imf_series(iso2, "CPI")
            fx1_s, fx1_src = fetch_imf_series(iso1, "FX")
            fx2_s, fx2_src = fetch_imf_series(iso2, "FX")
        except Exception as err:
            st.error(f"**IMF SDMX 3.0 Data Retrieval Failure:** {err}")
            st.info(
                "Confirm that both selected ISO3 jurisdictions publish monthly Consumer Price Index (`CPI._T.IX.M`) "
                "and domestic currency per USD (`XDC_USD.PA_RT.M`) series in the IMF International Financial Statistics database."
            )
            st.stop()

    # --- Merge & Validate Overlapping Observations ---
    df_valid = assemble_base_dataframe(iso1, iso2, fx1_s, fx2_s, cpi1_s, cpi2_s)

    if df_valid.empty:
        st.warning(
            f"No valid overlapping monthly observations found for **{iso1}** and **{iso2}** from `{START_PERIOD_YM}` onward."
        )
        st.stop()

    valid_months: List[str] = df_valid["Date"].tolist()
    earliest_month = valid_months[0]
    latest_month = valid_months[-1]

    # --- Synchronize Sample Window State Across KPI Header and Tabs ---
    slider_state_key = f"sample_window_slider_{iso1}_{iso2}"
    if slider_state_key not in st.session_state:
        st.session_state[slider_state_key] = (earliest_month, latest_month)
    else:
        curr_s, curr_e = st.session_state[slider_state_key]
        if (
            curr_s not in valid_months
            or curr_e not in valid_months
            or curr_s > curr_e
        ):
            st.session_state[slider_state_key] = (earliest_month, latest_month)

    active_start, active_end = st.session_state[slider_state_key]

    # Pre-compute normalized sample dataframe for KPI cards (re-verified when slider renders)
    sample_df = filter_and_normalize_sample(df_valid, active_start, active_end)
    if sample_df.empty:
        st.warning("Selected sample window contains no valid monthly observations.")
        st.stop()

    n_obs = len(sample_df)
    rer_star_val = float(sample_df["RER_Star_Sample_Mean"].iloc[0])
    real_idx_mean = float(sample_df["Real_Index"].mean())
    latest_s = float(sample_df["Nominal_S_Ccy2_per_Ccy1"].iloc[-1])
    latest_real_idx = float(sample_df["Real_Index"].iloc[-1])
    latest_cpi1 = float(sample_df["CPI_1"].iloc[-1])
    latest_cpi2 = float(sample_df["CPI_2"].iloc[-1])

    # --- Institutional KPI Summary Cards ---
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Base Country (ISO3_1)</div>
                <div class="kpi-value">{iso1} &mdash; {name1}</div>
                <div class="kpi-sub">Denominator of S(t) &nbsp;|&nbsp; Latest CPI_1: {latest_cpi1:.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Quote Country (ISO3_2)</div>
                <div class="kpi-value">{iso2} &mdash; {name2}</div>
                <div class="kpi-sub">Numerator of S(t) &nbsp;|&nbsp; Latest CPI_2: {latest_cpi2:.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Period Mean RER* ({active_start} to {active_end})</div>
                <div class="kpi-value">{rer_star_val:.6f}</div>
                <div class="kpi-sub">Mean Real Index: {real_idx_mean:.2f} &nbsp;|&nbsp; Latest S(t): {latest_s:.4f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Sample Months (N)</div>
                <div class="kpi-value">{n_obs:,} Months</div>
                <div class="kpi-sub">Latest Real Index ({active_end}): {latest_real_idx:.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Tabbed Institutional Architecture ---
    tab_chart, tab_data, tab_audit = st.tabs(
        [
            "Chart Analytics",
            "Time Series Data",
            "Methodology & Audit Proof",
        ]
    )

    # =========================================================================
    # TAB 1: CHART ANALYTICS
    # =========================================================================
    with tab_chart:
        st.markdown('<div class="inst-panel">', unsafe_allow_html=True)
        ctrl_col1, ctrl_col2 = st.columns([1.1, 2.4])

        with ctrl_col1:
            display_mode = st.radio(
                "Series Display Mode",
                options=["Real", "Nominal", "Both"],
                index=2,
                horizontal=True,
                help=(
                    "'Real': Normalized Real Exchange Rate Index (Period Mean = 100.00). "
                    "'Nominal': Unnormalized Nominal Cross-Rate S(t) in Currency 2 per 1 Currency 1. "
                    "'Both': Dual-axis overlay comparing Real Index (left) and Nominal S(t) (right)."
                ),
            )

        with ctrl_col2:
            selected_window = st.select_slider(
                "Active Sample Window (YYYY-MM) — Dynamically Rebases RER* & Real Index (Mean = 100.00)",
                options=valid_months,
                key=slider_state_key,
            )

        start_month, end_month = selected_window
        sample_df = filter_and_normalize_sample(df_valid, start_month, end_month)
        n_obs = len(sample_df)
        rer_star_val = float(sample_df["RER_Star_Sample_Mean"].iloc[0])
        real_idx_mean = float(sample_df["Real_Index"].mean())
        latest_s = float(sample_df["Nominal_S_Ccy2_per_Ccy1"].iloc[-1])
        latest_real_idx = float(sample_df["Real_Index"].iloc[-1])

        fig = build_exchange_rate_chart(
            sample_df=sample_df,
            iso1=iso1,
            iso2=iso2,
            name1=name1,
            name2=name2,
            display_mode=display_mode,
            start_month=start_month,
            end_month=end_month,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Distribution & Valuation Summary Strip
        idx_min = float(sample_df["Real_Index"].min())
        idx_min_dt = sample_df.loc[sample_df["Real_Index"].idxmin(), "Date"]
        idx_max = float(sample_df["Real_Index"].max())
        idx_max_dt = sample_df.loc[sample_df["Real_Index"].idxmax(), "Date"]
        idx_std = float(sample_df["Real_Index"].std(ddof=1)) if n_obs > 1 else 0.0
        deviation_pct = latest_real_idx - 100.0

        st.markdown(
            f"""
            <div class="stat-strip">
                <div>
                    <div class="stat-item-label">Sample Mean Real Index</div>
                    <div class="stat-item-val">{real_idx_mean:.2f} (Target: 100.00)</div>
                </div>
                <div>
                    <div class="stat-item-label">Latest Valuation Spread ({end_month})</div>
                    <div class="stat-item-val">{deviation_pct:+.2f}% vs. Period Mean</div>
                </div>
                <div>
                    <div class="stat-item-label">Sample Minimum Real Index</div>
                    <div class="stat-item-val">{idx_min:.2f} ({idx_min_dt})</div>
                </div>
                <div>
                    <div class="stat-item-label">Sample Maximum Real Index</div>
                    <div class="stat-item-val">{idx_max:.2f} ({idx_max_dt})</div>
                </div>
                <div>
                    <div class="stat-item-label">Real Index Volatility (Std Dev)</div>
                    <div class="stat-item-val">{idx_std:.2f} index pts</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 2: TIME SERIES DATA
    # =========================================================================
    with tab_data:
        st.markdown('<div class="inst-panel">', unsafe_allow_html=True)
        hdr_col1, hdr_col2 = st.columns([3, 1])

        with hdr_col1:
            st.markdown(
                f"**Normalized Monthly Dataset: {name1} (`{iso1}`) vs. {name2} (`{iso2}`)**  \n"
                f"<span style='color:#64748B;font-size:0.84rem;'>"
                f"Active Window: <code>{start_month}</code> to <code>{end_month}</code> "
                f"({n_obs:,} valid monthly observations) &nbsp;|&nbsp; "
                f"Sample Mean RER*: <code>{rer_star_val:.6f}</code> &nbsp;|&nbsp; "
                f"Mean Real Index: <code>{real_idx_mean:.2f}</code>"
                f"</span>",
                unsafe_allow_html=True,
            )

        csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
        file_name = f"IMF_RER_{iso1}_{iso2}_{start_month}_to_{end_month}.csv"

        with hdr_col2:
            st.download_button(
                label=f"Export Data (.CSV) — {n_obs:,} Rows",
                data=csv_bytes,
                file_name=file_name,
                mime="text/csv",
                use_container_width=True,
            )

        st.dataframe(
            sample_df,
            use_container_width=True,
            height=460,
            hide_index=True,
            column_config={
                "Date": st.column_config.TextColumn("Month (YYYY-MM)"),
                "Country_1_ISO": st.column_config.TextColumn("Base ISO3"),
                "Country_2_ISO": st.column_config.TextColumn("Quote ISO3"),
                "FX1_per_USD": st.column_config.NumberColumn(
                    f"FX1 ({iso1}/USD)", format="%.6f"
                ),
                "FX2_per_USD": st.column_config.NumberColumn(
                    f"FX2 ({iso2}/USD)", format="%.6f"
                ),
                "Nominal_S_Ccy2_per_Ccy1": st.column_config.NumberColumn(
                    f"Nominal S(t) ({iso2}/{iso1})", format="%.6f"
                ),
                "CPI_1": st.column_config.NumberColumn(
                    f"CPI_1 ({iso1})", format="%.4f"
                ),
                "CPI_2": st.column_config.NumberColumn(
                    f"CPI_2 ({iso2})", format="%.4f"
                ),
                "RER": st.column_config.NumberColumn("RER(t)", format="%.6f"),
                "RER_Star_Sample_Mean": st.column_config.NumberColumn(
                    "Sample Mean RER*", format="%.6f"
                ),
                "Real_Index": st.column_config.NumberColumn(
                    "Real Index (Mean=100)", format="%.2f"
                ),
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: METHODOLOGY & AUDIT PROOF
    # =========================================================================
    with tab_audit:
        # --- Structural Equations Panel ---
        st.markdown('<div class="inst-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-header">'
            "<span>Structural Macroeconomic Specification &amp; Index Construction</span>"
            "<span>IMF SDMX 3.0 IFS STANDARD</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        eq_col1, eq_col2 = st.columns(2)
        with eq_col1:
            st.markdown(
                "**1. Bilateral Nominal Cross-Rate $S(t)$**\n"
                "Constructed from period-average local currency units per $1\\text{ USD}$ "
                "(`{ISO3}.XDC_USD.PA_RT.M`), expressing units of **Country 2** currency per 1 unit of **Country 1** currency:\n"
                "$$S(t) = \\frac{FX_2(t)}{FX_1(t)}$$\n"
                "*USA Edge Case Invariant*: Whenever `USA` is selected as Country 1 or Country 2, "
                "$FX_{\\text{USA}}(t) \\equiv 1.000000$ across all monthly dates.\n\n"
                "**2. Bilateral Real Exchange Rate $RER(t)$**\n"
                "Deflates nominal exchange rate movements by the ratio of monthly Consumer Price Indices "
                "(`{ISO3}.CPI._T.IX.M`):\n"
                "$$RER(t) = S(t) \\times \\frac{P_1(t)}{P_2(t)}$$"
            )
        with eq_col2:
            st.markdown(
                "**3. Dynamic Sample Window Mean $RER^*$**\n"
                "The valid dataset is filtered to the active `[start_month, end_month]` window **first**, "
                "after which the arithmetic mean $RER^*$ is computed over the $N$ valid months in the window:\n"
                "$$RER^* = \\frac{1}{N} \\sum_{t=\\text{start}}^{\\text{end}} RER(t)$$\n"
                "**4. Normalized Real Exchange Rate Index**\n"
                "Each monthly observation inside the selected sample is indexed to $RER^*$ so that the "
                "sample mean of $\\text{Real Index}(t)$ is identically **`100.00`**:\n"
                "$$\\text{Real Index}(t) = 100 \\times \\frac{RER(t)}{RER^*}$$"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Step-by-Step Arithmetic Audit Proof Panel ---
        audit_col_latest, audit_col_first = st.columns(2)

        with audit_col_latest:
            st.markdown('<div class="inst-panel">', unsafe_allow_html=True)
            render_observation_verification(
                row=sample_df.iloc[-1],
                title_label="Latest Observation Arithmetic Audit Proof",
                iso1=iso1,
                iso2=iso2,
                n_obs=n_obs,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with audit_col_first:
            st.markdown('<div class="inst-panel">', unsafe_allow_html=True)
            render_observation_verification(
                row=sample_df.iloc[0],
                title_label="Initial Window Observation Arithmetic Audit Proof",
                iso1=iso1,
                iso2=iso2,
                n_obs=n_obs,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # --- API Provenance Footer ---
        st.markdown('<div class="inst-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-header">'
            "<span>IMF SDMX 3.0 Endpoint Provenance</span>"
            "<span>LIVE CACHE TTL: 3,600 SECONDS</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.code(
            f"Country 1 CPI ({iso1}) : {cpi1_src}\n"
            f"Country 2 CPI ({iso2}) : {cpi2_src}\n"
            f"Country 1 FX  ({iso1}) : {fx1_src}\n"
            f"Country 2 FX  ({iso2}) : {fx2_src}",
            language="text",
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
