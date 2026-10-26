import io
import xml.etree.ElementTree as ET
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
MIN_TIMESTAMP = pd.Timestamp("1976-01-01")

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

EXPORT_COLUMNS: List[str] = [
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


# -----------------------------------------------------------------------------
# THEME-ADAPTIVE INSTITUTIONAL CSS (LIGHT & DARK MODE SAFE)
# -----------------------------------------------------------------------------
def inject_institutional_css() -> None:
    """
    Inject theme-adaptive CSS rules bound strictly to Streamlit's native CSS
    variables (--background-color, --secondary-background-color, --text-color)
    to eliminate invisible text in both Light and Dark themes.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: var(--text-color);
            background-color: var(--background-color);
        }

        .block-container {
            padding-top: 1.15rem !important;
            padding-bottom: 2.25rem !important;
            padding-left: 2.0rem !important;
            padding-right: 2.0rem !important;
            max-width: 1600px;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: var(--secondary-background-color) !important;
            border-right: 1px solid rgba(128, 128, 128, 0.25) !important;
        }

        .sidebar-section-label {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-color) !important;
            opacity: 0.78;
            margin-top: 0.85rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid rgba(128, 128, 128, 0.25);
            padding-bottom: 0.35rem;
        }

        /* Native st.metric & Custom Metric Cards */
        [data-testid="stMetric"], .metric-card {
            background-color: var(--secondary-background-color) !important;
            border: 1px solid rgba(128, 128, 128, 0.25) !important;
            border-radius: 8px !important;
            padding: 14px 18px !important;
            font-variant-numeric: tabular-nums !important;
        }

        [data-testid="stMetricLabel"] p, .metric-card-label {
            color: var(--text-color) !important;
            opacity: 0.8 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            margin-bottom: 0.25rem !important;
        }

        [data-testid="stMetricValue"] div, .metric-card-value {
            color: var(--text-color) !important;
            font-weight: 700 !important;
            font-size: 1.35rem !important;
            font-variant-numeric: tabular-nums !important;
        }

        [data-testid="stMetricDelta"] div, .metric-card-sub {
            color: var(--text-color) !important;
            opacity: 0.75 !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.75rem !important;
            font-variant-numeric: tabular-nums !important;
        }

        /* Executive Top Banner */
        .exec-banner {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-left: 4px solid #2563EB;
            border-radius: 8px;
            padding: 1.1rem 1.4rem;
            margin-bottom: 1.0rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
        }

        .exec-title-group {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }

        .exec-eyebrow {
            font-size: 0.70rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            color: var(--text-color) !important;
            opacity: 0.72;
        }

        .exec-title {
            font-size: 1.32rem;
            font-weight: 700;
            color: var(--text-color) !important;
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
            color: var(--text-color) !important;
            background-color: var(--background-color);
            border: 1px solid rgba(128, 128, 128, 0.3);
            border-radius: 4px;
            padding: 0.3rem 0.65rem;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }

        .exec-badge-accent {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            color: #3B82F6 !important;
            background-color: rgba(59, 130, 246, 0.14);
            border: 1px solid rgba(59, 130, 246, 0.40);
            border-radius: 4px;
            padding: 0.3rem 0.65rem;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }

        /* Institutional Panels */
        .inst-panel {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 8px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1rem;
        }

        .panel-header {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--text-color) !important;
            border-bottom: 1px solid rgba(128, 128, 128, 0.25);
            padding-bottom: 0.6rem;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Distribution Summary Strip */
        .stat-strip {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.75rem;
            margin-top: 0.5rem;
        }

        @media (max-width: 1100px) {
            .stat-strip {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        /* Streamlit Tabs Overrides */
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            font-size: 0.80rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            color: var(--text-color) !important;
            opacity: 0.72 !important;
            padding: 0.65rem 1.25rem !important;
            background-color: transparent !important;
        }

        div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--text-color) !important;
            opacity: 1.0 !important;
            border-bottom: 2px solid #2563EB !important;
        }

        div[data-testid="stDataFrame"] {
            font-variant-numeric: tabular-nums !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 1. RESILIENT SDMX 3.0 PARSERS -> TWO-COLUMN (Date, Value) FORMAT
# -----------------------------------------------------------------------------
def _standardize_two_column_df(
    raw_dates: pd.Series, raw_values: pd.Series
) -> Optional[pd.DataFrame]:
    """
    Normalize raw SDMX date strings and observation values into a clean
    two-column DataFrame ('Date', 'Value') where:
      - 'Date' is pd.Timestamp normalized to first-of-month (YYYY-MM-01)
      - 'Value' is strictly positive float64 (> 0.0)
    Sorted ascending by 'Date' with duplicates removed.
    """
    if raw_dates.empty or raw_values.empty:
        return None

    cleaned_tokens = (
        raw_dates.astype(str)
        .str.strip()
        .str.upper()
        .str.replace("-M", "-", regex=False)
        .str.replace("M", "-", regex=False)
        .str.slice(0, 7)
    )

    dt_series = pd.to_datetime(cleaned_tokens, format="%Y-%m", errors="coerce")
    val_series = pd.to_numeric(raw_values, errors="coerce")

    valid_mask = (
        dt_series.notna()
        & (dt_series >= MIN_TIMESTAMP)
        & val_series.notna()
        & np.isfinite(val_series)
        & (val_series > 0.0)
    )

    if not valid_mask.any():
        return None

    out = pd.DataFrame(
        {
            "Date": dt_series[valid_mask].dt.to_period("M").dt.to_timestamp(),
            "Value": val_series[valid_mask].astype(float),
        }
    )
    out = (
        out.sort_values("Date", ascending=True)
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )
    return out if not out.empty else None


def _parse_sdmx_csv(text: str) -> Optional[pd.DataFrame]:
    """Parse SDMX-CSV response into a clean two-column (Date, Value) DataFrame."""
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

    return _standardize_two_column_df(df[time_col], df[val_col])


def _parse_sdmx_json(payload: dict) -> Optional[pd.DataFrame]:
    """Parse SDMX-JSON (2.0.0 and 1.0 schemas) into a two-column (Date, Value) DataFrame."""
    try:
        root = payload.get("data", payload)
        datasets = root.get("dataSets", [])
        if not datasets:
            return None

        series_dict = datasets[0].get("series", {})
        if not series_dict:
            return None

        structures = root.get("structures") or []
        structure = root.get("structure") or (structures[0] if structures else {})
        obs_dims = structure.get("dimensions", {}).get("observation", [])

        time_values: List[str] = []
        for dim in obs_dims:
            if dim.get("id", "").upper() in ("TIME_PERIOD", "TIME") or "values" in dim:
                time_values = [
                    str(v.get("id") or v.get("name") or "")
                    for v in dim.get("values", [])
                ]
                if time_values:
                    break

        if not time_values:
            return None

        dates_list: List[str] = []
        vals_list: List[object] = []

        for _, s_obj in series_dict.items():
            observations = s_obj.get("observations", {})
            for obs_idx_str, obs_val_list in observations.items():
                idx = int(obs_idx_str)
                if 0 <= idx < len(time_values) and obs_val_list:
                    dates_list.append(time_values[idx])
                    vals_list.append(obs_val_list[0])

        if not dates_list:
            return None

        return _standardize_two_column_df(
            pd.Series(dates_list, dtype="object"),
            pd.Series(vals_list, dtype="object"),
        )
    except Exception:
        return None


def _parse_sdmx_xml(text: str) -> Optional[pd.DataFrame]:
    """Parse SDMX-ML (StructureSpecific or Generic XML) into a two-column (Date, Value) DataFrame."""
    try:
        root = ET.fromstring(text)
    except Exception:
        return None

    dates_list: List[str] = []
    vals_list: List[object] = []

    for elem in root.iter():
        if elem.tag.split("}")[-1].lower() == "obs":
            raw_t = elem.attrib.get("TIME_PERIOD") or elem.attrib.get("time_period")
            raw_v = elem.attrib.get("OBS_VALUE") or elem.attrib.get("obs_value")

            if raw_t is None or raw_v is None:
                for child in elem:
                    c_tag = child.tag.split("}")[-1].lower()
                    if c_tag == "obsdimension":
                        raw_t = child.attrib.get("value")
                    elif c_tag == "obsvalue":
                        raw_v = child.attrib.get("value")

            if raw_t is not None and raw_v is not None:
                dates_list.append(str(raw_t))
                vals_list.append(raw_v)

    if not dates_list:
        return None

    return _standardize_two_column_df(
        pd.Series(dates_list, dtype="object"),
        pd.Series(vals_list, dtype="object"),
    )


# -----------------------------------------------------------------------------
# DECOUPLED NETWORK CLIENT (CACHES ONLY RAW HTTP CALLS BY SCALAR PRIMITIVES)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_imf_series(iso_code: str, series_type: str) -> Tuple[pd.DataFrame, str]:
    """
    Retrieve monthly CPI or FX observations from the IMF SDMX 3.0 API.
    Cache key relies strictly on scalar primitives: (iso_code, series_type).

    Parameters
    ----------
    iso_code : str
        3-letter ISO country code (or IMF aggregate code such as 'G163').
    series_type : str
        'CPI' -> {ISO3}.CPI._T.IX.M
        'FX'  -> {ISO3}.XDC_USD.PA_RT.M

    Returns
    -------
    Tuple[pd.DataFrame, str]
        - Two-column DataFrame with columns ['Date', 'Value']
        - Provenance URL string
    """
    iso_clean = str(iso_code).strip().upper()
    stype_clean = str(series_type).strip().upper()

    if stype_clean == "CPI":
        series_key = f"{iso_clean}.CPI._T.IX.M"
        candidate_dataflows = ["CPI", "IFS"]
    elif stype_clean == "FX":
        series_key = f"{iso_clean}.XDC_USD.PA_RT.M"
        candidate_dataflows = ["ER", "IFS"]
    else:
        raise ValueError(f"Unsupported series_type: {series_type}")

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
        "User-Agent": "IMF-RER-MacroAnalyzer/5.0 (Institutional Full-Stack; Python 3.12)",
        "Accept-Encoding": "gzip, deflate",
    }

    diagnostics: List[str] = []

    for url in candidate_urls:
        for fmt, accept_header in accept_profiles:
            headers = {**base_headers, "Accept": accept_header}
            try:
                resp = session.get(url, headers=headers, timeout=15)
            except requests.RequestException as exc:
                diagnostics.append(
                    f"{url} [{fmt}] -> Network error: {exc.__class__.__name__}"
                )
                continue

            if resp.status_code != 200 or not resp.text.strip():
                diagnostics.append(f"{url} [{fmt}] -> HTTP {resp.status_code}")
                continue

            content_type = resp.headers.get("Content-Type", "").lower()
            parsed_df: Optional[pd.DataFrame] = None

            if fmt == "csv" or "csv" in content_type:
                parsed_df = _parse_sdmx_csv(resp.text)
            if parsed_df is None and (fmt == "json" or "json" in content_type):
                try:
                    parsed_df = _parse_sdmx_json(resp.json())
                except Exception:
                    parsed_df = None
            if parsed_df is None and (
                fmt == "xml"
                or "xml" in content_type
                or resp.text.lstrip().startswith("<")
            ):
                parsed_df = _parse_sdmx_xml(resp.text)

            if parsed_df is not None and not parsed_df.empty:
                return parsed_df, url

    error_summary = (
        "; ".join(diagnostics[:4])
        if diagnostics
        else "No response from IMF SDMX endpoints."
    )
    raise RuntimeError(
        f"No valid monthly observations returned for `{series_key}`. Diagnostics: {error_summary}"
    )


# -----------------------------------------------------------------------------
# 2. DETERMINISTIC INNER-JOIN MERGING & USA / SAME-COUNTRY SAFEGUARDS
# -----------------------------------------------------------------------------
def build_master_dataframe(
    iso1: str, iso2: str
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Merge Country 1 FX, Country 2 FX, Country 1 CPI, and Country 2 CPI into a single
    master DataFrame (`df_master`) using a strict inner join on monthly dates.
    - USA Edge Case: Does not call the IMF FX endpoint for USA; injects a synthetic
      series where FX = 1.0 for all available monthly dates.
    - Same-Country Edge Case: Short-circuits redundant fetches when iso1 == iso2.
    - Drops any rows where any of the 4 inputs are missing, null, or <= 0, and sorts
      strictly by date ascending.
    """
    iso1_clean = iso1.strip().upper()
    iso2_clean = iso2.strip().upper()

    empty_df = pd.DataFrame()
    empty_prov = {"cpi1": "N/A", "cpi2": "N/A", "fx1": "N/A", "fx2": "N/A"}

    # --- Same-Country Edge Case (iso1 == iso2) ---
    if iso1_clean == iso2_clean:
        try:
            cpi1_df, cpi1_src = fetch_imf_series(iso1_clean, "CPI")
        except Exception:
            return empty_df, empty_prov

        if iso1_clean == "USA":
            fx1_df = pd.DataFrame(
                {"Date": cpi1_df["Date"].copy(), "Value": 1.0}
            )
            fx1_src = "Synthetic In-Memory Numeraire: USA.XDC_USD.PA_RT.M = 1.0000"
        else:
            try:
                fx1_df, fx1_src = fetch_imf_series(iso1_clean, "FX")
            except Exception:
                return empty_df, empty_prov

        merged = pd.merge(
            cpi1_df.rename(columns={"Value": "CPI_1"}),
            fx1_df.rename(columns={"Value": "FX1_per_USD"}),
            on="Date",
            how="inner",
        )
        if merged.empty:
            return empty_df, empty_prov

        merged["CPI_2"] = merged["CPI_1"]
        merged["FX2_per_USD"] = merged["FX1_per_USD"]
        cpi2_src = f"Identical Base/Quote Short-Circuit ({cpi1_src})"
        fx2_src = f"Identical Base/Quote Short-Circuit ({fx1_src})"

    # --- Distinct Country Pair ---
    else:
        try:
            cpi1_df, cpi1_src = fetch_imf_series(iso1_clean, "CPI")
            cpi2_df, cpi2_src = fetch_imf_series(iso2_clean, "CPI")
        except Exception:
            return empty_df, empty_prov

        cpi_merged = pd.merge(
            cpi1_df.rename(columns={"Value": "CPI_1"}),
            cpi2_df.rename(columns={"Value": "CPI_2"}),
            on="Date",
            how="inner",
        )
        if cpi_merged.empty:
            return empty_df, empty_prov

        if iso1_clean == "USA":
            try:
                fx2_df, fx2_src = fetch_imf_series(iso2_clean, "FX")
            except Exception:
                return empty_df, empty_prov
            fx1_df = pd.DataFrame(
                {"Date": cpi_merged["Date"].copy(), "Value": 1.0}
            )
            fx1_src = "Synthetic In-Memory Numeraire: USA.XDC_USD.PA_RT.M = 1.0000"
        elif iso2_clean == "USA":
            try:
                fx1_df, fx1_src = fetch_imf_series(iso1_clean, "FX")
            except Exception:
                return empty_df, empty_prov
            fx2_df = pd.DataFrame(
                {"Date": cpi_merged["Date"].copy(), "Value": 1.0}
            )
            fx2_src = "Synthetic In-Memory Numeraire: USA.XDC_USD.PA_RT.M = 1.0000"
        else:
            try:
                fx1_df, fx1_src = fetch_imf_series(iso1_clean, "FX")
                fx2_df, fx2_src = fetch_imf_series(iso2_clean, "FX")
            except Exception:
                return empty_df, empty_prov

        fx_merged = pd.merge(
            fx1_df.rename(columns={"Value": "FX1_per_USD"}),
            fx2_df.rename(columns={"Value": "FX2_per_USD"}),
            on="Date",
            how="inner",
        )
        if fx_merged.empty:
            return empty_df, empty_prov

        merged = pd.merge(cpi_merged, fx_merged, on="Date", how="inner")

    if merged.empty:
        return empty_df, empty_prov

    # Drop any rows where any of the 4 inputs are missing, null, or <= 0
    req_cols = ["FX1_per_USD", "FX2_per_USD", "CPI_1", "CPI_2"]
    for col in req_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.dropna(subset=req_cols)
    positive_mask = (
        (merged["FX1_per_USD"] > 0.0)
        & (merged["FX2_per_USD"] > 0.0)
        & (merged["CPI_1"] > 0.0)
        & (merged["CPI_2"] > 0.0)
    )
    merged = merged.loc[positive_mask].copy()
    if merged.empty:
        return empty_df, empty_prov

    # Sort strictly by date ascending
    merged = (
        merged.sort_values("Date", ascending=True)
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    # Preserve pd.Timestamp (first of month) for plotting and YYYY-MM string label for slicing/export
    merged["Date_TS"] = merged["Date"]
    merged["Date"] = merged["Date_TS"].dt.strftime("%Y-%m")

    # Vectorized Bilateral Nominal Cross-Rate S(t) and Real Exchange Rate RER(t)
    merged["Nominal_S_Ccy2_per_Ccy1"] = merged["FX2_per_USD"] / merged["FX1_per_USD"]
    merged["RER"] = merged["Nominal_S_Ccy2_per_Ccy1"] * (
        merged["CPI_1"] / merged["CPI_2"]
    )

    finite_mask = (
        np.isfinite(merged["Nominal_S_Ccy2_per_Ccy1"])
        & (merged["Nominal_S_Ccy2_per_Ccy1"] > 0.0)
        & np.isfinite(merged["RER"])
        & (merged["RER"] > 0.0)
    )
    merged = merged.loc[finite_mask].reset_index(drop=True)
    if merged.empty:
        return empty_df, empty_prov

    merged["Country_1_ISO"] = iso1_clean
    merged["Country_2_ISO"] = iso2_clean

    provenance = {
        "cpi1": cpi1_src,
        "cpi2": cpi2_src,
        "fx1": fx1_src,
        "fx2": fx2_src,
    }
    return merged, provenance


# -----------------------------------------------------------------------------
# CACHED CSV EXPORT HELPER
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Encode the filtered sample DataFrame to UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------------------------------------------------------
# UNIFIED DUAL-AXIS PLOTLY VISUALIZATION ENGINE
# -----------------------------------------------------------------------------
def build_exchange_rate_chart(
    df_sample: pd.DataFrame,
    iso1: str,
    iso2: str,
    name1: str,
    name2: str,
    display_mode: str,
    start_date: str,
    end_date: str,
) -> go.Figure:
    """
    Build a crash-proof Plotly figure using make_subplots(specs=[[{"secondary_y": True}]])
    across all three display modes ('Real', 'Nominal', 'Both').
    Uses explicit reference traces rather than add_hline to prevent subplot axis
    reference errors during mode transitions.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    plot_dates = df_sample["Date_TS"]

    real_axis_title = "Real Exchange Rate Index (Period Avg = 100)"
    nominal_axis_title = (
        f"Nominal S(t) [Units of Ccy 2 ({iso2}) per 1 Unit of Ccy 1 ({iso1})]"
    )

    # Theme-safe high-contrast colors legible on both dark (#0E1117) and light (#FFFFFF) canvases
    deep_blue = "#2563EB"
    amber_gold = "#F59E0B"
    axis_neutral = "#94A3B8"
    grid_color = "rgba(148, 163, 184, 0.20)"
    axis_line_color = "rgba(148, 163, 184, 0.40)"

    common_hoverlabel = dict(
        bgcolor="#0F172A",
        bordercolor="#334155",
        font=dict(
            family="JetBrains Mono, Inter, monospace",
            size=12,
            color="#F8FAFC",
        ),
    )

    common_xaxis = dict(
        title=dict(text="", font=dict(size=11, color=axis_neutral)),
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=axis_line_color,
        linewidth=1,
        ticks="outside",
        tickcolor=axis_line_color,
        tickfont=dict(
            family="Inter, sans-serif",
            size=11,
            color=axis_neutral,
        ),
        hoverformat="%Y-%m",
    )

    if display_mode == "Real":
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=df_sample["Real_Index"],
                mode="lines",
                name=f"Real Exchange Rate Index ({iso1}/{iso2})",
                line=dict(color=deep_blue, width=2.5),
                hovertemplate="Real Index (Avg=100): <b>%{y:.4f}</b><extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=[plot_dates.iloc[0], plot_dates.iloc[-1]],
                y=[100.0, 100.0],
                mode="lines",
                name="Period Average Benchmark (100.00)",
                line=dict(color=axis_neutral, width=1.5, dash="dash"),
                hoverinfo="skip",
            ),
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text=real_axis_title,
            title_font=dict(family="Inter, sans-serif", size=12, color=deep_blue),
            tickfont=dict(
                family="JetBrains Mono, monospace", size=11, color=deep_blue
            ),
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=axis_line_color,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            secondary_y=True,
        )
        chart_title = (
            f"<b>Bilateral Real Exchange Rate Index</b>  "
            f"<span style='color:{axis_neutral};font-size:13px;'>"
            f"{name1} ({iso1}) vs. {name2} ({iso2})  |  {start_date} — {end_date}"
            f"</span>"
        )

    elif display_mode == "Nominal":
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=df_sample["Nominal_S_Ccy2_per_Ccy1"],
                mode="lines",
                name=f"Nominal S(t) [{iso2} per 1 {iso1}]",
                line=dict(color=amber_gold, width=2.3),
                hovertemplate=(
                    f"Nominal S(t): <b>%{{y:.4f}}</b> {iso2} per 1 {iso1}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text=nominal_axis_title,
            title_font=dict(family="Inter, sans-serif", size=12, color=amber_gold),
            tickfont=dict(
                family="JetBrains Mono, monospace", size=11, color=amber_gold
            ),
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=axis_line_color,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            secondary_y=True,
        )
        chart_title = (
            f"<b>Bilateral Nominal Exchange Rate S(t)</b>  "
            f"<span style='color:{axis_neutral};font-size:13px;'>"
            f"Units of {iso2} Currency per 1 Unit of {iso1} Currency  |  {start_date} — {end_date}"
            f"</span>"
        )

    else:  # "Both"
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=df_sample["Real_Index"],
                mode="lines",
                name="Real Exchange Rate Index (Left Axis, Period Avg = 100)",
                line=dict(color=deep_blue, width=2.5),
                hovertemplate="Real Index (Avg=100): <b>%{y:.4f}</b><extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=[plot_dates.iloc[0], plot_dates.iloc[-1]],
                y=[100.0, 100.0],
                mode="lines",
                name="Period Avg (100.00)",
                line=dict(color=axis_neutral, width=1.4, dash="dash"),
                hoverinfo="skip",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=df_sample["Nominal_S_Ccy2_per_Ccy1"],
                mode="lines",
                name=f"Nominal S(t): {iso2} per 1 {iso1} (Right Axis)",
                line=dict(color=amber_gold, width=2.1),
                hovertemplate=(
                    f"Nominal S(t) ({iso2}/{iso1}): <b>%{{y:.4f}}</b><extra></extra>"
                ),
            ),
            secondary_y=True,
        )
        fig.update_yaxes(
            title_text=real_axis_title,
            title_font=dict(family="Inter, sans-serif", size=12, color=deep_blue),
            tickfont=dict(
                family="JetBrains Mono, monospace", size=11, color=deep_blue
            ),
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor=axis_line_color,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text=nominal_axis_title,
            title_font=dict(family="Inter, sans-serif", size=12, color=amber_gold),
            tickfont=dict(
                family="JetBrains Mono, monospace", size=11, color=amber_gold
            ),
            showticklabels=True,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor=axis_line_color,
            secondary_y=True,
        )
        chart_title = (
            f"<b>Real Exchange Rate Index vs. Nominal Cross-Rate S(t)</b>  "
            f"<span style='color:{axis_neutral};font-size:13px;'>"
            f"{name1} ({iso1}) / {name2} ({iso2})  |  {start_date} — {end_date}"
            f"</span>"
        )

    fig.update_layout(
        font=dict(family="Inter, sans-serif", color=axis_neutral),
        title=dict(
            text=chart_title,
            font=dict(family="Inter, sans-serif", size=15, color=axis_neutral),
            x=0.0,
            xanchor="left",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=common_hoverlabel,
        xaxis=common_xaxis,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12, color=axis_neutral),
        ),
        height=500,
        margin=dict(l=65, r=65, t=85, b=40),
    )

    return fig


# -----------------------------------------------------------------------------
# AUDIT & VERIFICATION RENDERING ENGINE
# -----------------------------------------------------------------------------
def render_observation_verification(
    row: pd.Series, title_label: str, iso1: str, iso2: str, n_obs: int
) -> None:
    """Render step-by-step arithmetic audit proof for a single monthly observation."""
    t_str = str(row["Date"])
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
        f'<span style="font-family:\'JetBrains Mono\',monospace;color:#3B82F6;">OBS PERIOD: {t_str}</span>'
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
# MAIN APPLICATION WORKSPACE (STRICT TOP-DOWN EXECUTION ORDER)
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
            key="select_country_1",
            help="Base economy: denominator of nominal cross-rate S(t) and numerator of relative price ratio P1(t)/P2(t).",
        )
        c2_selection = st.selectbox(
            "Country 2 — Quote Currency (ISO3_2)",
            options=country_options,
            index=default_c2_idx,
            key="select_country_2",
            help="Quote/Counter economy: numerator of nominal cross-rate S(t) and denominator of relative price ratio P1(t)/P2(t).",
        )

        with st.expander("Custom ISO3 Code Override", expanded=False):
            custom_iso1 = (
                st.text_input("Override Country 1 ISO3", value="", key="cust_iso1")
                .strip()
                .upper()
            )
            custom_iso2 = (
                st.text_input("Override Country 2 ISO3", value="", key="cust_iso2")
                .strip()
                .upper()
            )

        country1 = (
            custom_iso1 if len(custom_iso1) >= 3 else c1_selection.split(" - ")[0]
        )
        country2 = (
            custom_iso2 if len(custom_iso2) >= 3 else c2_selection.split(" - ")[0]
        )
        name1 = COUNTRIES.get(country1, country1)
        name2 = COUNTRIES.get(country2, country2)

        st.markdown(
            '<div class="sidebar-section-label">SDMX 3.0 Series Keys</div>',
            unsafe_allow_html=True,
        )
        st.code(
            f"CPI_1 : {country1}.CPI._T.IX.M\n"
            f"CPI_2 : {country2}.CPI._T.IX.M\n"
            f"FX_1  : {country1}.XDC_USD.PA_RT.M\n"
            f"FX_2  : {country2}.XDC_USD.PA_RT.M",
            language="text",
        )

        st.markdown(
            '<div class="sidebar-section-label">Cache &amp; Session Management</div>',
            unsafe_allow_html=True,
        )
        if st.button("Refresh IMF SDMX 3.0 Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("last_loaded_pair", None)
            st.rerun()

    # --- Same-Country Edge Case Callout ---
    if country1 == country2:
        st.info(
            "Identical base and quote currencies selected: Nominal S(t) = 1.0000, Real Index = 100.00."
        )

    # --- Data Ingestion & Deterministic Inner-Join Merge ---
    current_pair_id = f"{country1}_{country2}"
    if st.session_state.get("last_loaded_pair") != current_pair_id:
        with st.spinner("Querying IMF SDMX gateway..."):
            df_master, provenance = build_master_dataframe(country1, country2)
        st.session_state["last_loaded_pair"] = current_pair_id
    else:
        df_master, provenance = build_master_dataframe(country1, country2)

    if df_master.empty:
        st.error("Insufficient overlapping historical data for this pair.")
        st.stop()

    valid_months: List[str] = df_master["Date"].tolist()
    if len(valid_months) < 2:
        st.error("Insufficient overlapping historical data for this pair.")
        st.stop()

    # --- Top-Level Interactive Control Bar (Guarantees Deterministic Slicing) ---
    st.markdown('<div class="inst-panel">', unsafe_allow_html=True)
    ctrl_col1, ctrl_col2 = st.columns([1.15, 2.35])

    with ctrl_col1:
        display_mode = st.radio(
            "Series Display Mode",
            options=["Real", "Nominal", "Both"],
            index=2,
            horizontal=True,
            key="radio_display_mode",
            help=(
                "'Real': Normalized Real Exchange Rate Index (Period Avg = 100). "
                "'Nominal': Unnormalized Nominal Cross-Rate S(t) in Currency 2 per 1 Currency 1. "
                "'Both': Dual-axis overlay comparing Real Index (left) and Nominal S(t) (right)."
            ),
        )

    with ctrl_col2:
        start_date, end_date = st.select_slider(
            "Active Sample Window (YYYY-MM) — Dynamically Rebases RER* & Real Index (Mean = 100.00)",
            options=valid_months,
            value=(valid_months[0], valid_months[-1]),
            key=f"slider_{country1}_{country2}",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Vectorized Economic Math & Dynamic Sample Normalization ---
    df_sample = df_master[
        (df_master["Date"] >= start_date) & (df_master["Date"] <= end_date)
    ].copy()

    if len(df_sample) < 2:
        st.warning("Insufficient observations in selected window. Select at least 2 months.")
        st.stop()

    rer_star_val = float(df_sample["RER"].mean())
    if not np.isfinite(rer_star_val) or rer_star_val <= 0.0:
        st.warning("Insufficient observations in selected window.")
        st.stop()

    df_sample["RER_Star_Sample_Mean"] = rer_star_val
    df_sample["Real_Index"] = 100.0 * (df_sample["RER"] / rer_star_val)
    df_sample = df_sample.reset_index(drop=True)

    # --- Compute Active Window KPIs ---
    n_obs = len(df_sample)
    real_idx_mean = float(df_sample["Real_Index"].mean())
    latest_s = float(df_sample["Nominal_S_Ccy2_per_Ccy1"].iloc[-1])
    latest_real_idx = float(df_sample["Real_Index"].iloc[-1])
    latest_cpi1 = float(df_sample["CPI_1"].iloc[-1])
    latest_cpi2 = float(df_sample["CPI_2"].iloc[-1])

    # --- KPI Summary Strip (Reflects Active Slider Window Immediately) ---
    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    k_col1.metric(
        label="Base Country (ISO3_1)",
        value=f"{country1} — {name1}",
        delta=f"Denominator of S(t) | CPI_1: {latest_cpi1:.2f}",
        delta_color="off",
    )
    k_col2.metric(
        label="Quote Country (ISO3_2)",
        value=f"{country2} — {name2}",
        delta=f"Numerator of S(t) | CPI_2: {latest_cpi2:.2f}",
        delta_color="off",
    )
    k_col3.metric(
        label=f"Period Mean RER* ({start_date} to {end_date})",
        value=f"{rer_star_val:.6f}",
        delta=f"Mean Index: {real_idx_mean:.2f} | S(t): {latest_s:.4f}",
        delta_color="off",
    )
    k_col4.metric(
        label="Total Sample Months (N)",
        value=f"{n_obs:,} Months",
        delta=f"Latest Real Index ({end_date}): {latest_real_idx:.2f}",
        delta_color="off",
    )

    # --- Tabbed Institutional Workspace ---
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
        fig = build_exchange_rate_chart(
            df_sample=df_sample,
            iso1=country1,
            iso2=country2,
            name1=name1,
            name2=name2,
            display_mode=display_mode,
            start_date=start_date,
            end_date=end_date,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"plotly_{country1}_{country2}_{display_mode}_{start_date}_{end_date}",
            config={"displayModeBar": False, "responsive": True},
        )

        # Distribution & Valuation Summary Strip
        idx_min = float(df_sample["Real_Index"].min())
        idx_min_dt = str(df_sample.loc[df_sample["Real_Index"].idxmin(), "Date"])
        idx_max = float(df_sample["Real_Index"].max())
        idx_max_dt = str(df_sample.loc[df_sample["Real_Index"].idxmax(), "Date"])
        idx_std = float(df_sample["Real_Index"].std(ddof=1)) if n_obs > 1 else 0.0
        deviation_pct = latest_real_idx - 100.0

        st.markdown(
            f"""
            <div class="stat-strip">
                <div class="metric-card">
                    <div class="metric-card-label">Sample Mean Real Index</div>
                    <div class="metric-card-value">{real_idx_mean:.2f}</div>
                    <div class="metric-card-sub">Target Benchmark: 100.00</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-label">Valuation Spread ({end_date})</div>
                    <div class="metric-card-value">{deviation_pct:+.2f}%</div>
                    <div class="metric-card-sub">vs. Window Mean (100.00)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-label">Window Minimum Index</div>
                    <div class="metric-card-value">{idx_min:.2f}</div>
                    <div class="metric-card-sub">Observed: {idx_min_dt}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-label">Window Maximum Index</div>
                    <div class="metric-card-value">{idx_max:.2f}</div>
                    <div class="metric-card-sub">Observed: {idx_max_dt}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-card-label">Index Volatility (Std Dev)</div>
                    <div class="metric-card-value">{idx_std:.2f} pts</div>
                    <div class="metric-card-sub">Sample N = {n_obs:,} months</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 2: TIME SERIES DATA
    # =========================================================================
    with tab_data:
        df_export = df_sample[EXPORT_COLUMNS].copy()

        hdr_col1, hdr_col2 = st.columns([3, 1])
        with hdr_col1:
            st.markdown(
                f"**Normalized Monthly Dataset: {name1} (`{country1}`) vs. {name2} (`{country2}`)**  \n"
                f"Active Window: `{start_date}` to `{end_date}` "
                f"(`{n_obs:,}` valid monthly observations) | "
                f"Sample Mean $RER^*$: `{rer_star_val:.6f}` | "
                f"Mean Real Index: `{real_idx_mean:.2f}`"
            )

        csv_bytes = convert_df_to_csv(df_export)
        file_name = f"IMF_RER_{country1}_{country2}_{start_date}_to_{end_date}.csv"

        with hdr_col2:
            st.download_button(
                label=f"Export Data (.CSV) — {n_obs:,} Rows",
                data=csv_bytes,
                file_name=file_name,
                mime="text/csv",
                use_container_width=True,
            )

        st.dataframe(
            df_export,
            use_container_width=True,
            height=460,
            hide_index=True,
            column_config={
                "Date": st.column_config.TextColumn("Month (YYYY-MM)"),
                "Country_1_ISO": st.column_config.TextColumn("Base ISO3"),
                "Country_2_ISO": st.column_config.TextColumn("Quote ISO3"),
                "FX1_per_USD": st.column_config.NumberColumn(
                    f"FX1 ({country1}/USD)", format="%.6f"
                ),
                "FX2_per_USD": st.column_config.NumberColumn(
                    f"FX2 ({country2}/USD)", format="%.6f"
                ),
                "Nominal_S_Ccy2_per_Ccy1": st.column_config.NumberColumn(
                    f"Nominal S(t) ({country2}/{country1})", format="%.6f"
                ),
                "CPI_1": st.column_config.NumberColumn(
                    f"CPI_1 ({country1})", format="%.4f"
                ),
                "CPI_2": st.column_config.NumberColumn(
                    f"CPI_2 ({country2})", format="%.4f"
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

    # =========================================================================
    # TAB 3: METHODOLOGY & AUDIT PROOF
    # =========================================================================
    with tab_audit:
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
                "*USA & Same-Country Invariants*: Whenever `USA` is selected, $FX_{\\text{USA}}(t) \\equiv 1.000000$ "
                "is synthesized in memory without network calls. Whenever $\\text{Country}_1 == \\text{Country}_2$, "
                "$S(t) = 1.0000$, $RER(t) = 1.0000$, and $\\text{Real Index}(t) = 100.00$.\n\n"
                "**2. Bilateral Real Exchange Rate $RER(t)$**\n"
                "Deflates nominal exchange rate movements by the ratio of monthly Consumer Price Indices "
                "(`{ISO3}.CPI._T.IX.M`):\n"
                "$$RER(t) = S(t) \\times \\frac{P_1(t)}{P_2(t)}$$"
            )
        with eq_col2:
            st.markdown(
                "**3. Dynamic Sample Window Mean $RER^*$**\n"
                "The valid inner-join dataset `df_master` is filtered to `[start_date, end_date]` **first**, "
                "after which the arithmetic mean $RER^*$ is computed strictly over the $N$ valid months in the window:\n"
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
                row=df_sample.iloc[-1],
                title_label="Latest Observation Arithmetic Audit Proof",
                iso1=country1,
                iso2=country2,
                n_obs=n_obs,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with audit_col_first:
            st.markdown('<div class="inst-panel">', unsafe_allow_html=True)
            render_observation_verification(
                row=df_sample.iloc[0],
                title_label="Initial Window Observation Arithmetic Audit Proof",
                iso1=country1,
                iso2=country2,
                n_obs=n_obs,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # --- API Provenance Footer ---
        st.markdown('<div class="inst-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-header">'
            "<span>IMF SDMX 3.0 Endpoint Provenance</span>"
            "<span>CACHE TTL: 86,400 SECONDS (24 HOURS)</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.code(
            f"Country 1 CPI ({country1}) : {provenance['cpi1']}\n"
            f"Country 2 CPI ({country2}) : {provenance['cpi2']}\n"
            f"Country 1 FX  ({country1}) : {provenance['fx1']}\n"
            f"Country 2 FX  ({country2}) : {provenance['fx2']}",
            language="text",
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
