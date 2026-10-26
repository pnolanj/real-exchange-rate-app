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
    page_title="IMF Bilateral Real Exchange Rate (RER) Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CONSTANTS & COUNTRY DICTIONARY (42 IMF Economies + Aggregates)
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
    return pd.Series(data=1.0, index=pd.Index(ym_index, name="Date"), name="FX_USD", dtype=float)


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
    return pd.Series(sub["Value"].values, index=pd.Index(sub["Date"].values, name="Date"), dtype=float)


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
                time_values = [v.get("id") or v.get("name") for v in dim.get("values", [])]
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
        tag_Ordered = elem.tag.split("}")[-1].lower()
        if tag_Ordered == "obs":
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
        return build_usa_fx_series(START_PERIOD_YM), "Deterministic Rule: USA.XDC_USD.PA_RT.M = 1.0"

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
        ("json", "application/vnd.sdmx.data+json;version=2.0.0, application/json;q=0.9"),
        ("xml", "application/vnd.sdmx.structurespecificdata+xml;version=2.1, application/xml;q=0.9"),
    ]

    session = requests.Session()
    base_headers = {
        "User-Agent": "IMF-RER-MacroAnalyzer/1.0 (Streamlit; Python 3.12)",
        "Accept-Encoding": "gzip, deflate",
    }

    diagnostics: List[str] = []

    for url in candidate_urls:
        for fmt, accept_header in accept_profiles:
            headers = {**base_headers, "Accept": accept_header}
            try:
                resp = session.get(url, headers=headers, timeout=20)
            except requests.RequestException as exc:
                diagnostics.append(f"{url} [{fmt}] -> Network error: {exc.__class__.__name__}")
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
            if parsed is None and (fmt == "xml" or "xml" in content_type or resp.text.lstrip().startswith("<")):
                parsed = _parse_sdmx_xml(resp.text)

            if parsed is not None and not parsed.empty:
                parsed.name = f"{iso3_clean}_{indicator_type}"
                return parsed, url

    error_summary = "; ".join(diagnostics[:4]) if diagnostics else "No response from IMF SDMX endpoints."
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
        fx1 = pd.Series(1.0, index=cpi_union_idx.union(fx2.index), name="FX1_per_USD", dtype=float)
    if iso2_clean == "USA":
        fx2 = pd.Series(1.0, index=cpi_union_idx.union(fx1.index), name="FX2_per_USD", dtype=float)

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
# INTERACTIVE PLOTLY VISUALIZATION
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
    """Build interactive Plotly figure for 'Real', 'Nominal', or 'Both' modes."""
    plot_dates = pd.to_datetime(sample_df["Date"] + "-01")
    real_y_title = "Real Exchange Rate Index (Period Avg = 100)"
    nom_y_title = f"Nominal Rate S(t): {iso2} per 1 {iso1}"

    if display_mode == "Real":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=sample_df["Real_Index"],
                mode="lines",
                name=f"Real Index ({iso1}/{iso2})",
                line=dict(color="#1f77b4", width=2.5),
                hovertemplate="<b>%{x|%Y-%m}</b><br>Real Index: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_hline(
            y=100.0,
            line_dash="dash",
            line_color="#666666",
            line_width=1.5,
            annotation_text="Period Average (100.0)",
            annotation_position="bottom right",
        )
        fig.update_layout(
            title=f"Bilateral Real Exchange Rate Index: {name1} ({iso1}) vs. {name2} ({iso2}) [{start_month} to {end_month}]",
            xaxis_title="Month",
            yaxis_title=real_y_title,
            hovermode="x unified",
            template="plotly_white",
            height=520,
            margin=dict(l=60, r=40, t=70, b=50),
        )

    elif display_mode == "Nominal":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=sample_df["Nominal_S_Ccy2_per_Ccy1"],
                mode="lines",
                name=f"Nominal S(t) ({iso2} per 1 {iso1})",
                line=dict(color="#d95f02", width=2.5),
                hovertemplate=(
                    f"<b>%{{x|%Y-%m}}</b><br>Nominal S(t): %{{y:.4f}} {iso2} per 1 {iso1}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=f"Bilateral Nominal Exchange Rate S(t): Units of {iso2} Currency per 1 Unit of {iso1} Currency [{start_month} to {end_month}]",
            xaxis_title="Month",
            yaxis_title=f"Units of {iso2} Currency per 1 Unit of {iso1} Currency",
            hovermode="x unified",
            template="plotly_white",
            height=520,
            margin=dict(l=60, r=40, t=70, b=50),
        )

    else:  # "Both"
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=sample_df["Real_Index"],
                mode="lines",
                name="Real Exchange Rate Index (Left Axis)",
                line=dict(color="#1f77b4", width=2.5),
                hovertemplate="Real Index (Avg=100): %{y:.2f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=sample_df["Nominal_S_Ccy2_per_Ccy1"],
                mode="lines",
                name=f"Nominal S(t): {iso2} per 1 {iso1} (Right Axis)",
                line=dict(color="#d95f02", width=2.2),
                hovertemplate=f"Nominal S(t): %{{y:.4f}} {iso2}/{iso1}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.add_hline(
            y=100.0,
            line_dash="dash",
            line_color="#1f77b4",
            line_width=1.2,
            opacity=0.65,
            secondary_y=False,
        )
        fig.update_layout(
            title=f"Real Exchange Rate Index vs. Unnormalized Nominal Rate S(t): {name1} ({iso1}) / {name2} ({iso2}) [{start_month} to {end_month}]",
            xaxis_title="Month",
            hovermode="x unified",
            template="plotly_white",
            height=540,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            margin=dict(l=60, r=60, t=90, b=50),
        )
        fig.update_yaxes(
            title_text=real_y_title,
            title_font=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4"),
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text=nom_y_title,
            title_font=dict(color="#d95f02"),
            tickfont=dict(color="#d95f02"),
            showgrid=False,
            secondary_y=True,
        )

    return fig


# -----------------------------------------------------------------------------
# MANUAL VERIFICATION HELPER
# -----------------------------------------------------------------------------
def render_observation_verification(row: pd.Series, label: str, iso1: str, iso2: str) -> None:
    """Render exact step-by-step formula verification for a single observation row."""
    t_str = row["Date"]
    fx1 = row["FX1_per_USD"]
    fx2 = row["FX2_per_USD"]
    s_t = row["Nominal_S_Ccy2_per_Ccy1"]
    p1 = row["CPI_1"]
    p2 = row["CPI_2"]
    rer_t = row["RER"]
    rer_star = row["RER_Star_Sample_Mean"]
    real_idx = row["Real_Index"]

    st.markdown(f"#### {label}: `{t_str}`")
    st.markdown(
        f"- **Step 1 — Raw USD Exchange Rates** (`{{ISO3}}.XDC_USD.PA_RT.M`):\n"
        f"  $$FX_1({t_str}) = {fx1:.6f} \\text{{ ({iso1} per 1 USD)}}, \\quad "
        f"FX_2({t_str}) = {fx2:.6f} \\text{{ ({iso2} per 1 USD)}}$$\n"
        f"- **Step 2 — Bilateral Nominal Exchange Rate** ($S(t)$ in units of `{iso2}` per 1 `{iso1}`):\n"
        f"  $$S({t_str}) = \\frac{{FX_2({t_str})}}{{FX_1({t_str})}} = \\frac{{{fx2:.6f}}}{{{fx1:.6f}}} = {s_t:.6f}$$\n"
        f"- **Step 3 — Monthly Consumer Price Indices** (`{{ISO3}}.CPI._T.IX.M`):\n"
        f"  $$P_1({t_str}) = {p1:.4f} \\text{{ ({iso1} CPI)}}, \\quad "
        f"P_2({t_str}) = {p2:.4f} \\text{{ ({iso2} CPI)}}$$\n"
        f"- **Step 4 — Bilateral Real Exchange Rate** ($RER(t)$):\n"
        f"  $$RER({t_str}) = S({t_str}) \\times \\frac{{P_1({t_str})}}{{P_2({t_str})}} "
        f"= {s_t:.6f} \\times \\frac{{{p1:.4f}}}{{{p2:.4f}}} = {rer_t:.6f}$$\n"
        f"- **Step 5 — Normalized Real Exchange Rate Index** (using sample mean $RER^* = {rer_star:.6f}$):\n"
        f"  $$\\text{{Real Index}}({t_str}) = 100 \\times \\frac{{RER({t_str})}}{{RER^*}} "
        f"= 100 \\times \\frac{{{rer_t:.6f}}}{{{rer_star:.6f}}} = {real_idx:.2f}$$"
    )


# -----------------------------------------------------------------------------
# MAIN APPLICATION UI
# -----------------------------------------------------------------------------
def main() -> None:
    st.title("📈 Bilateral Real Exchange Rate (RER) & Index Analyzer")
    st.caption(
        "Live monthly data via IMF SDMX 3.0 API (`{ISO3}.CPI._T.IX.M` & `{ISO3}.XDC_USD.PA_RT.M`) | "
        "Dynamic sample-window normalization ($\\text{Mean Real Index} = 100.00$)"
    )

    country_options = [f"{code} - {name}" for code, name in COUNTRIES.items()]
    default_c1_idx = list(COUNTRIES.keys()).index("GBR")
    default_c2_idx = list(COUNTRIES.keys()).index("USA")

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("Model Parameters")

        c1_selection = st.selectbox(
            "Country 1 (Base Currency — ISO3_1)",
            options=country_options,
            index=default_c1_idx,
            help="Base country (Denominator of S(t), Numerator of Price Ratio P1/P2).",
        )
        c2_selection = st.selectbox(
            "Country 2 (Quote Currency — ISO3_2)",
            options=country_options,
            index=default_c2_idx,
            help="Quote/Counter country (Numerator of S(t), Denominator of Price Ratio P1/P2).",
        )

        with st.expander("Custom ISO3 Code Override (Optional)"):
            custom_iso1 = st.text_input("Custom ISO3 for Country 1", value="").strip().upper()
            custom_iso2 = st.text_input("Custom ISO3 for Country 2", value="").strip().upper()

        iso1 = custom_iso1 if len(custom_iso1) >= 3 else c1_selection.split(" - ")[0]
        iso2 = custom_iso2 if len(custom_iso2) >= 3 else c2_selection.split(" - ")[0]
        name1 = COUNTRIES.get(iso1, iso1)
        name2 = COUNTRIES.get(iso2, iso2)

        st.divider()
        display_mode = st.radio(
            "Chart Display Mode",
            options=["Real", "Nominal", "Both"],
            index=2,
            help=(
                "'Real': Normalized Real Index (Period Avg = 100). "
                "'Nominal': Actual S(t) currency units (Ccy2 per 1 Ccy1). "
                "'Both': Dual-axis chart comparing Real Index and Nominal S(t)."
            ),
        )

        st.divider()
        if st.button("🔄 Clear API Cache & Reload", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # --- Data Retrieval from IMF SDMX 3.0 API ---
    with st.spinner(f"Fetching monthly CPI and FX series for {iso1} and {iso2} from IMF SDMX 3.0 API..."):
        try:
            cpi1_s, cpi1_src = fetch_imf_series(iso1, "CPI")
            cpi2_s, cpi2_src = fetch_imf_series(iso2, "CPI")
            fx1_s, fx1_src = fetch_imf_series(iso1, "FX")
            fx2_s, fx2_src = fetch_imf_series(iso2, "FX")
        except Exception as err:
            st.error(f"**IMF SDMX 3.0 Data Retrieval Error:** {err}")
            st.info(
                "Verify that both ISO3 codes report monthly Consumer Price Index (`CPI._T.IX.M`) "
                "and domestic currency per USD (`XDC_USD.PA_RT.M`) series in the IMF database."
            )
            st.stop()

    # --- Merge & Validate Overlapping Observations ---
    df_valid = assemble_base_dataframe(iso1, iso2, fx1_s, fx2_s, cpi1_s, cpi2_s)

    if df_valid.empty:
        st.warning(
            f"No overlapping valid monthly observations found for **{iso1}** and **{iso2}** starting from `{START_PERIOD_YM}`."
        )
        st.stop()

    valid_months: List[str] = df_valid["Date"].tolist()
    earliest_month = valid_months[0]
    latest_month = valid_months[-1]

    # --- Date Range Selector (Bounded by Valid Overlapping Months) ---
    st.subheader("1. Sample Window Selection")
    col_start, col_end, col_info = st.columns([1, 1, 2])

    with col_start:
        start_month = st.selectbox(
            "Start Month (YYYY-MM)",
            options=valid_months,
            index=0,
        )
    with col_end:
        valid_end_options = [m for m in valid_months if m >= start_month]
        end_month = st.selectbox(
            "End Month (YYYY-MM)",
            options=valid_end_options,
            index=len(valid_end_options) - 1,
        )
    with col_info:
        st.info(
            f"**Full Valid Overlap Available:** `{earliest_month}` to `{latest_month}` "
            f"({len(valid_months):,} valid months).  \n"
            f"Changing **Start Month** or **End Month** dynamically recalculates $RER^*$ and "
            f"rebases $\\text{{Real Index}}(t)$ so the selected window mean is strictly `100.00`."
        )

    # --- Filter FIRST, then Calculate RER* and Real Index(t) ---
    sample_df = filter_and_normalize_sample(df_valid, start_month, end_month)

    if sample_df.empty:
        st.warning("Selected date window contains no valid observations.")
        st.stop()

    # --- KPI Summary Metrics ---
    n_obs = len(sample_df)
    rer_star_val = float(sample_df["RER_Star_Sample_Mean"].iloc[0])
    real_idx_mean = float(sample_df["Real_Index"].mean())
    latest_s = float(sample_df["Nominal_S_Ccy2_per_Ccy1"].iloc[-1])
    latest_real_idx = float(sample_df["Real_Index"].iloc[-1])

    st.subheader("2. Sample Window Summary Metrics")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Selected Sample Period", f"{start_month} → {end_month}")
    k2.metric("Total Valid Months (N)", f"{n_obs:,}")
    k3.metric("Period Average RER*", f"{rer_star_val:.6f}")
    k4.metric("Mean of Real Index(t)", f"{real_idx_mean:.2f}")
    k5.metric(
        f"Latest S({end_month}) [{iso2}/{iso1}]",
        f"{latest_s:.4f}",
        delta=f"Real Index: {latest_real_idx:.2f}",
        delta_color="off",
    )

    # --- Interactive Plotly Visualization ---
    st.subheader("3. Interactive Exchange Rate Chart")
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

    # --- Manual Verification Expander ---
    with st.expander("Manual Verification Check (First & Latest Observation)", expanded=False):
        st.markdown(
            "This verification panel plugs exact values from the selected sample window into "
            "each structural equation to audit cross-rate construction and index normalization."
        )
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            render_observation_verification(
                sample_df.iloc[0], "First Observation in Selected Sample", iso1, iso2
            )
        with v_col2:
            render_observation_verification(
                sample_df.iloc[-1], "Latest Observation in Selected Sample", iso1, iso2
            )

        st.divider()
        st.caption(
            f"**Data Provenance:** `{iso1}` CPI: `{cpi1_src}` | `{iso2}` CPI: `{cpi2_src}` | "
            f"`{iso1}` FX: `{fx1_src}` | `{iso2}` FX: `{fx2_src}`"
        )

    # --- CSV Export & Data Preview ---
    st.subheader("4. Filtered Dataset Preview & CSV Export")
    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    file_name = f"IMF_RER_{iso1}_{iso2}_{start_month}_to_{end_month}.csv"

    dl_col, _ = st.columns([1, 3])
    with dl_col:
        st.download_button(
            label=f"📥 Download Filtered CSV ({n_obs:,} rows)",
            data=csv_bytes,
            file_name=file_name,
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(
        sample_df.style.format(
            {
                "FX1_per_USD": "{:.6f}",
                "FX2_per_USD": "{:.6f}",
                "Nominal_S_Ccy2_per_Ccy1": "{:.6f}",
                "CPI_1": "{:.4f}",
                "CPI_2": "{:.4f}",
                "RER": "{:.6f}",
                "RER_Star_Sample_Mean": "{:.6f}",
                "Real_Index": "{:.2f}",
            }
        ),
        use_container_width=True,
        height=380,
    )


if __name__ == "__main__":
    main()
