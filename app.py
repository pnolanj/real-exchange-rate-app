import bisect
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
# THEME-ADAPTIVE INSTITUTIONAL CSS (LIGHT & DARK MODE COMPATIBLE)
# -----------------------------------------------------------------------------
def inject_institutional_css() -> None:
    """
    Inject theme-aware CSS rules utilizing Streamlit's native CSS custom properties
    (--background-color, --secondary-background-color, --text-color) to guarantee
    high visual contrast in both Light Mode and Dark Mode.
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
            color: var(--text-color);
            opacity: 0.75;
            margin-top: 0.85rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid rgba(128, 128, 128, 0.25);
            padding-bottom: 0.35rem;
        }

        /* Card & Metric Containers (Strict Theme Discipline) */
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
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            margin-bottom: 0.25rem !important;
        }

        [data-testid="stMetricValue"] div, .metric-card-value {
            color: var(--text-color) !important;
            font-weight: 700 !important;
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
            padding: 1.15rem 1.5rem;
            margin-bottom: 1.25rem;
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
            color: var(--text-color);
            opacity: 0.7;
        }

        .exec-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text-color);
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
            color: var(--text-color);
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
            color: #2563EB;
            background-color: rgba(37, 99, 235, 0.12);
            border: 1px solid rgba(37, 99, 235, 0.35);
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
            padding: 1.15rem 1.35rem;
            margin-bottom: 1rem;
        }

        .panel-header {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--text-color);
            border-bottom: 1px solid rgba(128, 128, 128, 0.25);
            padding-bottom: 0.6rem;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Secondary Summary Strip under Chart */
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
            opacity: 0.7 !important;
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
# VECTORIZED DATE NORMALIZATION & RESILIENT SDMX PARSERS
# -----------------------------------------------------------------------------
def normalize_ym_scalar(raw_date: object) -> Optional[str]:
    """Normalize a single SDMX monthly period token to canonical 'YYYY-MM'."""
    if raw_date is None:
        return None
    s = str(raw_date).strip().upper().replace("-M", "-")
    if len(s) == 7 and "M" in s:
        s = s.replace("M", "-")
    if len(s) >= 7 and s[4] == "-":
        ym = s[:7]
        if ym[:4].isdigit() and ym[5:7].isdigit():
            yr, mo = int(ym[:4]), int(ym[5:7])
            if 1900 <= yr <= 2100 and 1 <= mo <= 12:
                return f"{yr:04d}-{mo:02d}"
    return None


def _clean_positive_series(s: Optional[pd.Series]) -> pd.Series:
    """
    Standardize a monthly Series to a sorted, unique 'YYYY-MM' index,
    dropping any missing, NaN, infinite, or non-positive (<= 0) observations.
    """
    if s is None or s.empty:
        return pd.Series(dtype=float, index=pd.Index([], name="Date"))

    vals = pd.to_numeric(s, errors="coerce")
    idx_str = pd.Index(s.index.astype(str), name="Date")

    valid_mask = (
        idx_str.str.match(r"^\d{4}-(0[1-9]|1[0-2])$", na=False)
        & (idx_str >= START_PERIOD_YM)
        & vals.notna()
        & np.isfinite(vals)
        & (vals > 0.0)
    )

    if not np.any(valid_mask):
        return pd.Series(dtype=float, index=pd.Index([], name="Date"))

    cleaned = pd.Series(
        vals[valid_mask].to_numpy(dtype=float),
        index=idx_str[valid_mask],
        dtype=float,
    )
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()
    cleaned.index.name = "Date"
    return cleaned


def _parse_sdmx_csv(text: str) -> Optional[pd.Series]:
    """Vectorized parser for SDMX-CSV payloads returning a monthly pd.Series indexed by 'YYYY-MM'."""
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

    dates = (
        df[time_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("-M", "-", regex=False)
        .str.replace("M", "-", regex=False)
        .str.slice(0, 7)
    )
    vals = pd.to_numeric(df[val_col], errors="coerce")

    raw_series = pd.Series(vals.to_numpy(), index=pd.Index(dates.to_numpy(), name="Date"))
    cleaned = _clean_positive_series(raw_series)
    return cleaned if not cleaned.empty else None


def _parse_sdmx_json(payload: dict) -> Optional[pd.Series]:
    """Parse SDMX-JSON (2.0.0 and 1.0 schemas) into a monthly pd.Series indexed by 'YYYY-MM'."""
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
                        ym = normalize_ym_scalar(time_values[idx])
                        try:
                            f_val = float(raw_val)
                        except (TypeError, ValueError):
                            continue
                        if ym and ym >= START_PERIOD_YM and np.isfinite(f_val) and f_val > 0.0:
                            records[ym] = f_val

        if not records:
            return None

        s = pd.Series(records, dtype=float)
        s.index.name = "Date"
        cleaned = _clean_positive_series(s)
        return cleaned if not cleaned.empty else None
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

            ym = normalize_ym_scalar(raw_t)
            if ym and ym >= START_PERIOD_YM and raw_v is not None:
                try:
                    val = float(raw_v)
                    if np.isfinite(val) and val > 0.0:
                        records[ym] = val
                except ValueError:
                    continue

    if not records:
        return None

    s = pd.Series(records, dtype=float)
    s.index.name = "Date"
    cleaned = _clean_positive_series(s)
    return cleaned if not cleaned.empty else None


# -----------------------------------------------------------------------------
# DECOUPLED NETWORK LAYER (PRIMITIVE SCALAR CACHE KEY, TTL = 86400s)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_imf_series(iso_code: str, series_type: str) -> Tuple[pd.Series, str]:
    """
    Dedicated raw HTTP client for the IMF SDMX 3.0 API.
    Cache key relies strictly on primitive scalar arguments: (iso_code, series_type).

    Parameters
    ----------
    iso_code : str
        ISO3 country code (e.g., 'GBR', 'USA', 'G163').
    series_type : str
        'CPI' -> {ISO3}.CPI._T.IX.M
        'FX'  -> {ISO3}.XDC_USD.PA_RT.M
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
        "User-Agent": "IMF-RER-MacroAnalyzer/4.0 (Production-Hardened; Python 3.12)",
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
                parsed.name = f"{iso_clean}_{stype_clean}"
                return parsed, url

    error_summary = (
        "; ".join(diagnostics[:4])
        if diagnostics
        else "No response from IMF SDMX endpoints."
    )
    raise RuntimeError(
        f"No valid observations returned for `{series_key}`. Diagnostics: {error_summary}"
    )


# -----------------------------------------------------------------------------
# STRICT INNER-JOIN DATA PIPELINE & SAME-COUNTRY / USA SAFEGUARDS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_merged_pair_dataset(
    iso1: str, iso2: str
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Construct the base bilateral DataFrame on the strict inner intersection of valid
    monthly dates ('YYYY-MM').
    - Handles Same-Country edge case (iso1 == iso2) without redundant fetches.
    - Handles USA numeraire edge case (USA.XDC_USD.PA_RT.M = 1.0) in memory.
    - Returns an empty DataFrame if no overlapping valid observations exist.
    """
    iso1_clean = iso1.strip().upper()
    iso2_clean = iso2.strip().upper()

    empty_df = pd.DataFrame()
    empty_prov = {"cpi1": "N/A", "cpi2": "N/A", "fx1": "N/A", "fx2": "N/A"}

    # --- CASE 1: Same-Country Edge Case (Country_1 == Country_2) ---
    if iso1_clean == iso2_clean:
        try:
            raw_cpi1, cpi1_src = fetch_imf_series(iso1_clean, "CPI")
        except Exception:
            return empty_df, empty_prov

        cpi1_s = _clean_positive_series(raw_cpi1)
        if cpi1_s.empty:
            return empty_df, empty_prov

        if iso1_clean == "USA":
            fx1_s = pd.Series(1.0, index=cpi1_s.index, name="USA_FX", dtype=float)
            fx1_src = "Synthetic In-Memory Numeraire: USA.XDC_USD.PA_RT.M = 1.0000"
        else:
            try:
                raw_fx1, fx1_src = fetch_imf_series(iso1_clean, "FX")
                fx1_s = _clean_positive_series(raw_fx1)
            except Exception:
                fx1_s = pd.Series(dtype=float, index=pd.Index([], name="Date"))
                fx1_src = "Unavailable"

            if fx1_s.empty:
                return empty_df, empty_prov

        # Strict inner intersection of valid dates
        common_idx = cpi1_s.index.intersection(fx1_s.index).sort_values()
        if len(common_idx) == 0:
            return empty_df, empty_prov

        cpi_aligned = cpi1_s.reindex(common_idx)
        fx_aligned = fx1_s.reindex(common_idx)

        df = pd.DataFrame(
            {
                "FX1_per_USD": fx_aligned,
                "FX2_per_USD": fx_aligned.copy(),
                "CPI_1": cpi_aligned,
                "CPI_2": cpi_aligned.copy(),
                "Nominal_S_Ccy2_per_Ccy1": 1.0,
                "RER": 1.0,
                "Country_1_ISO": iso1_clean,
                "Country_2_ISO": iso2_clean,
            },
            index=common_idx,
        )
        df.index.name = "Date"
        provenance = {
            "cpi1": cpi1_src,
            "cpi2": f"Identical Base/Quote Short-Circuit ({cpi1_src})",
            "fx1": fx1_src,
            "fx2": f"Identical Base/Quote Short-Circuit ({fx1_src})",
        }
        return df, provenance

    # --- CASE 2: Distinct Countries (Strict Inner-Join Pipeline) ---
    try:
        raw_cpi1, cpi1_src = fetch_imf_series(iso1_clean, "CPI")
        raw_cpi2, cpi2_src = fetch_imf_series(iso2_clean, "CPI")
    except Exception:
        return empty_df, empty_prov

    cpi1_s = _clean_positive_series(raw_cpi1)
    cpi2_s = _clean_positive_series(raw_cpi2)

    if cpi1_s.empty or cpi2_s.empty:
        return empty_df, empty_prov

    cpi_common_idx = cpi1_s.index.intersection(cpi2_s.index)
    if len(cpi_common_idx) == 0:
        return empty_df, empty_prov

    if iso1_clean == "USA":
        try:
            raw_fx2, fx2_src = fetch_imf_series(iso2_clean, "FX")
        except Exception:
            return empty_df, empty_prov
        fx2_s = _clean_positive_series(raw_fx2)
        common_idx = cpi_common_idx.intersection(fx2_s.index).sort_values()
        if len(common_idx) == 0:
            return empty_df, empty_prov
        fx1_s = pd.Series(1.0, index=common_idx, name="USA_FX", dtype=float)
        fx1_src = "Synthetic In-Memory Numeraire: USA.XDC_USD.PA_RT.M = 1.0000"
    elif iso2_clean == "USA":
        try:
            raw_fx1, fx1_src = fetch_imf_series(iso1_clean, "FX")
        except Exception:
            return empty_df, empty_prov
        fx1_s = _clean_positive_series(raw_fx1)
        common_idx = cpi_common_idx.intersection(fx1_s.index).sort_values()
        if len(common_idx) == 0:
            return empty_df, empty_prov
        fx2_s = pd.Series(1.0, index=common_idx, name="USA_FX", dtype=float)
        fx2_src = "Synthetic In-Memory Numeraire: USA.XDC_USD.PA_RT.M = 1.0000"
    else:
        try:
            raw_fx1, fx1_src = fetch_imf_series(iso1_clean, "FX")
            raw_fx2, fx2_src = fetch_imf_series(iso2_clean, "FX")
        except Exception:
            return empty_df, empty_prov
        fx1_s = _clean_positive_series(raw_fx1)
        fx2_s = _clean_positive_series(raw_fx2)
        common_idx = (
            cpi_common_idx.intersection(fx1_s.index)
            .intersection(fx2_s.index)
            .sort_values()
        )
        if len(common_idx) == 0:
            return empty_df, empty_prov

    # Reindex all 4 series strictly to the inner intersection of valid monthly dates
    df = pd.DataFrame(
        {
            "FX1_per_USD": fx1_s.reindex(common_idx),
            "FX2_per_USD": fx2_s.reindex(common_idx),
            "CPI_1": cpi1_s.reindex(common_idx),
            "CPI_2": cpi2_s.reindex(common_idx),
        },
        index=common_idx,
    )
    df.index.name = "Date"

    # Drop any residual missing or non-positive values
    df = df.dropna(subset=["FX1_per_USD", "FX2_per_USD", "CPI_1", "CPI_2"])
    valid_mask = (
        (df["FX1_per_USD"] > 0.0)
        & (df["FX2_per_USD"] > 0.0)
        & (df["CPI_1"] > 0.0)
        & (df["CPI_2"] > 0.0)
    )
    df = df.loc[valid_mask].copy()
    if df.empty:
        return empty_df, empty_prov

    # Vectorized bilateral nominal cross-rate S(t) and Real Exchange Rate RER(t)
    df["Nominal_S_Ccy2_per_Ccy1"] = df["FX2_per_USD"] / df["FX1_per_USD"]
    df["RER"] = df["Nominal_S_Ccy2_per_Ccy1"] * (df["CPI_1"] / df["CPI_2"])

    finite_mask = (
        np.isfinite(df["Nominal_S_Ccy2_per_Ccy1"])
        & (df["Nominal_S_Ccy2_per_Ccy1"] > 0.0)
        & np.isfinite(df["RER"])
        & (df["RER"] > 0.0)
    )
    df = df.loc[finite_mask].copy()
    if df.empty:
        return empty_df, empty_prov

    df["Country_1_ISO"] = iso1_clean
    df["Country_2_ISO"] = iso2_clean

    provenance = {
        "cpi1": cpi1_src,
        "cpi2": cpi2_src,
        "fx1": fx1_src,
        "fx2": fx2_src,
    }
    return df, provenance


# -----------------------------------------------------------------------------
# DYNAMIC SLIDER SESSION-STATE GUARD & IN-MEMORY SLICING
# -----------------------------------------------------------------------------
def _snap_to_valid_month(target_ym: str, valid_months: List[str], prefer_upper: bool) -> str:
    """
    Clamp and snap an arbitrary 'YYYY-MM' string to a guaranteed member
    of the sorted `valid_months` list using binary search.
    """
    if target_ym <= valid_months[0]:
        return valid_months[0]
    if target_ym >= valid_months[-1]:
        return valid_months[-1]

    idx = bisect.bisect_left(valid_months, target_ym)
    if idx < len(valid_months) and valid_months[idx] == target_ym:
        return valid_months[idx]

    if prefer_upper:
        return valid_months[min(idx, len(valid_months) - 1)]
    return valid_months[max(0, idx - 1)]


def synchronize_slider_session_state(
    valid_months: List[str], current_pair_key: str
) -> Tuple[str, str]:
    """
    Ensure `st.session_state.selected_dates` is always a valid 2-tuple of
    (start_month, end_month) strictly contained within `valid_months`.
    - Resets to (valid_min, valid_max) on first load or when the user switches country pairs
      if existing bounds are invalid.
    - Clamps out-of-bounds session-state values into [valid_min, valid_max] before rendering
      the slider to eliminate StreamlitAPIException crashes.
    """
    valid_min = valid_months[0]
    valid_max = valid_months[-1]

    prev_pair = st.session_state.get("slider_bound_pair_key")

    if "selected_dates" not in st.session_state:
        st.session_state.selected_dates = (valid_min, valid_max)
        st.session_state["slider_bound_pair_key"] = current_pair_key
        return valid_min, valid_max

    raw_val = st.session_state.selected_dates
    if not isinstance(raw_val, (tuple, list)) or len(raw_val) != 2:
        st.session_state.selected_dates = (valid_min, valid_max)
        st.session_state["slider_bound_pair_key"] = current_pair_key
        return valid_min, valid_max

    raw_start, raw_end = str(raw_val[0]), str(raw_val[1])

    # Clamp and snap both bounds into the active valid_months list
    clamped_start = _snap_to_valid_month(raw_start, valid_months, prefer_upper=True)
    clamped_end = _snap_to_valid_month(raw_end, valid_months, prefer_upper=False)

    # If the country pair changed and previous bounds were full-range or collapsed after clamping,
    # ensure a valid non-empty span (at least 2 observations)
    start_pos = bisect.bisect_left(valid_months, clamped_start)
    end_pos = bisect.bisect_left(valid_months, clamped_end)

    if start_pos >= end_pos:
        clamped_start, clamped_end = valid_min, valid_max
    elif prev_pair != current_pair_key and (
        raw_start < valid_min or raw_end > valid_max
    ):
        # Keep the clamped intersection if it spans >= 2 months, otherwise reset to full overlap
        if (end_pos - start_pos) < 1:
            clamped_start, clamped_end = valid_min, valid_max

    st.session_state.selected_dates = (clamped_start, clamped_end)
    st.session_state["slider_bound_pair_key"] = current_pair_key
    return clamped_start, clamped_end


def slice_and_normalize_window(
    df_base: pd.DataFrame, start_month: str, end_month: str
) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Slice the date-indexed base DataFrame in memory via df.loc[start_month:end_month]
    and dynamically normalize Real_Index(t) = 100 * (RER(t) / RER*).
    Protects against empty slices, single-observation windows, and ZeroDivisionError.
    """
    if df_base.empty or start_month > end_month:
        return None, None

    sliced = df_base.loc[start_month:end_month].copy()
    if len(sliced) <= 1:
        return None, None

    rer_star = float(sliced["RER"].mean())
    if not np.isfinite(rer_star) or rer_star <= 0.0:
        return None, None

    sliced["RER_Star_Sample_Mean"] = rer_star
    sliced["Real_Index"] = 100.0 * (sliced["RER"] / rer_star)

    out = sliced.reset_index()
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
    return out[ordered_cols], rer_star


# -----------------------------------------------------------------------------
# HIGH-EFFICIENCY CACHED CSV SERIALIZATION
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Serialize the filtered sample DataFrame to UTF-8 CSV bytes with caching."""
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------------------------------------------------------
# UNIFIED PLOTLY SUBPLOT ENGINE (ZERO AXIS MISMATCH BUGS)
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
    Unified Plotly engine built strictly on make_subplots(specs=[[{"secondary_y": True}]])
    across all three display modes ('Real', 'Nominal', 'Both').
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    plot_dates = pd.to_datetime(sample_df["Date"] + "-01")

    real_axis_title = "Real Exchange Rate Index (Period Avg = 100)"
    nominal_axis_title = (
        f"Nominal S(t) [Units of Ccy 2 ({iso2}) per 1 Unit of Ccy 1 ({iso1})]"
    )

    # Institutional high-contrast palette
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
                y=sample_df["Real_Index"],
                mode="lines",
                name=f"Real Exchange Rate Index ({iso1}/{iso2})",
                line=dict(color=deep_blue, width=2.5),
                hovertemplate="Real Index (Avg=100): <b>%{y:.4f}</b><extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_hline(
            y=100.0,
            line_dash="dash",
            line_color=axis_neutral,
            line_width=1.5,
            secondary_y=False,
            annotation_text="Period Average (100.00)",
            annotation_position="bottom right",
            annotation_font=dict(
                family="JetBrains Mono, monospace",
                size=11,
                color=axis_neutral,
            ),
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
            f"{name1} ({iso1}) vs. {name2} ({iso2})  |  {start_month} — {end_month}"
            f"</span>"
        )

    elif display_mode == "Nominal":
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=sample_df["Nominal_S_Ccy2_per_Ccy1"],
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
            f"Units of {iso2} Currency per 1 Unit of {iso1} Currency  |  {start_month} — {end_month}"
            f"</span>"
        )

    else:  # "Both"
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=sample_df["Real_Index"],
                mode="lines",
                name="Real Exchange Rate Index (Left Axis, Period Avg = 100)",
                line=dict(color=deep_blue, width=2.5),
                hovertemplate="Real Index (Avg=100): <b>%{y:.4f}</b><extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=sample_df["Nominal_S_Ccy2_per_Ccy1"],
                mode="lines",
                name=f"Nominal S(t): {iso2} per 1 {iso1} (Right Axis)",
                line=dict(color=amber_gold, width=2.1),
                hovertemplate=(
                    f"Nominal S(t) ({iso2}/{iso1}): <b>%{{y:.4f}}</b><extra></extra>"
                ),
            ),
            secondary_y=True,
        )
        fig.add_hline(
            y=100.0,
            line_dash="dash",
            line_color=axis_neutral,
            line_width=1.4,
            secondary_y=False,
            annotation_text="Period Avg (100.00)",
            annotation_position="bottom right",
            annotation_font=dict(
                family="JetBrains Mono, monospace",
                size=10,
                color=axis_neutral,
            ),
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
            f"{name1} ({iso1}) / {name2} ({iso2})  |  {start_month} — {end_month}"
            f"</span>"
        )

    fig.update_layout(
        title=dict(
            text=chart_title,
            font=dict(family="Inter, sans-serif", size=15),
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
            st.session_state.pop("active_pair_key", None)
            st.session_state.pop("active_df_base", None)
            st.session_state.pop("active_provenance", None)
            st.session_state.pop("selected_dates", None)
            st.session_state.pop("slider_bound_pair_key", None)
            st.rerun()

    # --- Same-Country Edge Case Callout ---
    if iso1 == iso2:
        st.info(
            "Identical base and quote currencies selected: Nominal S(t) = 1.0000, Real Index = 100.00."
        )

    # --- Decoupled Network Fetch: Spinner Triggers ONLY on Country Pair Change ---
    current_pair_key = f"{iso1}_{iso2}"
    if (
        st.session_state.get("active_pair_key") != current_pair_key
        or "active_df_base" not in st.session_state
    ):
        with st.spinner("Querying IMF SDMX gateway..."):
            df_base, provenance = get_merged_pair_dataset(iso1, iso2)

        st.session_state["active_pair_key"] = current_pair_key
        st.session_state["active_df_base"] = df_base
        st.session_state["active_provenance"] = provenance
    else:
        df_base = st.session_state["active_df_base"]
        provenance = st.session_state["active_provenance"]

    # --- Strict Inner-Join Empty Intersection Guard ---
    if df_base.empty:
        st.error("No overlapping historical data available for this country pair.")
        st.stop()

    if len(df_base) <= 1:
        st.warning("Insufficient observations in selected window.")
        st.stop()

    valid_months: List[str] = df_base.index.tolist()

    # --- Dynamic Slider Session-State Guard (Clamp to [valid_min, valid_max]) ---
    synchronize_slider_session_state(valid_months, current_pair_key)

    # Top container placeholder so KPI cards always reflect the active slider slice
    kpi_container = st.container()

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
        ctrl_col1, ctrl_col2 = st.columns([1.1, 2.4])

        with ctrl_col1:
            display_mode = st.radio(
                "Series Display Mode",
                options=["Real", "Nominal", "Both"],
                index=2,
                horizontal=True,
                help=(
                    "'Real': Normalized Real Exchange Rate Index (Period Avg = 100). "
                    "'Nominal': Unnormalized Nominal Cross-Rate S(t) in Currency 2 per 1 Currency 1. "
                    "'Both': Dual-axis overlay comparing Real Index (left) and Nominal S(t) (right)."
                ),
            )

        with ctrl_col2:
            selected_window = st.select_slider(
                "Active Sample Window (YYYY-MM) — Dynamically Rebases RER* & Real Index (Mean = 100.00)",
                options=valid_months,
                key="selected_dates",
            )

        start_month, end_month = selected_window
        sample_df, rer_star_val = slice_and_normalize_window(
            df_base, start_month, end_month
        )

        # ZeroDivisionError & Empty/Single-Row Slice Protection
        if sample_df is None or rer_star_val is None:
            st.warning("Insufficient observations in selected window.")
            st.stop()

        n_obs = len(sample_df)
        real_idx_mean = float(sample_df["Real_Index"].mean())
        latest_s = float(sample_df["Nominal_S_Ccy2_per_Ccy1"].iloc[-1])
        latest_real_idx = float(sample_df["Real_Index"].iloc[-1])
        latest_cpi1 = float(sample_df["CPI_1"].iloc[-1])
        latest_cpi2 = float(sample_df["CPI_2"].iloc[-1])

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
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

        # Distribution & Valuation Summary Strip (.metric-card theme-adaptive containers)
        idx_min = float(sample_df["Real_Index"].min())
        idx_min_dt = sample_df.loc[sample_df["Real_Index"].idxmin(), "Date"]
        idx_max = float(sample_df["Real_Index"].max())
        idx_max_dt = sample_df.loc[sample_df["Real_Index"].idxmax(), "Date"]
        idx_std = float(sample_df["Real_Index"].std(ddof=1)) if n_obs > 1 else 0.0
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
                    <div class="metric-card-label">Valuation Spread ({end_month})</div>
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

    # Populate Top KPI Summary Cards using native st.metric (styled by [data-testid="stMetric"])
    with kpi_container:
        k_col1, k_col2, k_col3, k_col4 = st.columns(4)
        k_col1.metric(
            label="Base Country (ISO3_1)",
            value=f"{iso1} — {name1}",
            delta=f"Denominator of S(t) | CPI_1: {latest_cpi1:.2f}",
            delta_color="off",
        )
        k_col2.metric(
            label="Quote Country (ISO3_2)",
            value=f"{iso2} — {name2}",
            delta=f"Numerator of S(t) | CPI_2: {latest_cpi2:.2f}",
            delta_color="off",
        )
        k_col3.metric(
            label=f"Period Mean RER* ({start_month} to {end_month})",
            value=f"{rer_star_val:.6f}",
            delta=f"Mean Index: {real_idx_mean:.2f} | S(t): {latest_s:.4f}",
            delta_color="off",
        )
        k_col4.metric(
            label="Total Sample Months (N)",
            value=f"{n_obs:,} Months",
            delta=f"Latest Real Index ({end_month}): {latest_real_idx:.2f}",
            delta_color="off",
        )

    # =========================================================================
    # TAB 2: TIME SERIES DATA
    # =========================================================================
    with tab_data:
        hdr_col1, hdr_col2 = st.columns([3, 1])

        with hdr_col1:
            st.markdown(
                f"**Normalized Monthly Dataset: {name1} (`{iso1}`) vs. {name2} (`{iso2}`)**  \n"
                f"Active Window: `{start_month}` to `{end_month}` "
                f"(`{n_obs:,}` valid monthly observations) | "
                f"Sample Mean $RER^*$: `{rer_star_val:.6f}` | "
                f"Mean Real Index: `{real_idx_mean:.2f}`"
            )

        csv_bytes = convert_df_to_csv(sample_df)
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
                "is synthesized in memory. Whenever $\\text{Country}_1 == \\text{Country}_2$, "
                "$S(t) = 1.0000$, $RER(t) = 1.0000$, and $\\text{Real Index}(t) = 100.00$.\n\n"
                "**2. Bilateral Real Exchange Rate $RER(t)$**\n"
                "Deflates nominal exchange rate movements by the ratio of monthly Consumer Price Indices "
                "(`{ISO3}.CPI._T.IX.M`):\n"
                "$$RER(t) = S(t) \\times \\frac{P_1(t)}{P_2(t)}$$"
            )
        with eq_col2:
            st.markdown(
                "**3. Dynamic Sample Window Mean $RER^*$**\n"
                "The valid inner-join dataset is sliced in memory via `df.loc[start_month:end_month]` **first**, "
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
            "<span>CACHE TTL: 86,400 SECONDS (24 HOURS)</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.code(
            f"Country 1 CPI ({iso1}) : {provenance['cpi1']}\n"
            f"Country 2 CPI ({iso2}) : {provenance['cpi2']}\n"
            f"Country 1 FX  ({iso1}) : {provenance['fx1']}\n"
            f"Country 2 FX  ({iso2}) : {provenance['fx2']}",
            language="text",
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
