# streamlit_app.py
# -*- coding: utf-8 -*-
import os, json, pickle, logging, glob, zipfile, pathlib
import numpy as np, pandas as pd, geopandas as gpd
import streamlit as st, pydeck as pdk
from shapely.ops import unary_union
from app_utils import ensure_bundle  # downloads & extracts when BUNDLE_URL is provided

# Quiet SHAP noise
logging.getLogger("shap").setLevel(logging.ERROR)

# ---------- BUNDLE LOCATION (robust) ----------
def _extract_local_zip_if_present() -> str | None:
    """
    If a bundle ZIP (streamlit_bundle_full_*.zip) is present in the repo working dir,
    extract it under ./data/bundle_local and return that absolute path. Otherwise None.
    """
    candidates = (
        glob.glob("streamlit_bundle_full_*.zip") +
        glob.glob("/mount/src/*/streamlit_bundle_full_*.zip") +
        glob.glob("/app/*/streamlit_bundle_full_*.zip")
    )
    if not candidates:
        return None
    zip_path = candidates[0]
    target = pathlib.Path("./data/bundle_local").resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target)
    return str(target)

# Priority: explicit env (local/dev) → secrets/env BUNDLE_URL → local zip fallback
BUNDLE_URL = (os.environ.get("BUNDLE_URL", "") or st.secrets.get("BUNDLE_URL", "")).strip()

if os.environ.get("MVP_BASE"):
    BASE = os.environ["MVP_BASE"]
elif BUNDLE_URL:
    BASE = ensure_bundle(BUNDLE_URL)  # downloads & caches under ./data/bundle
else:
    # Last-chance fallback: a zip committed/uploaded next to the app (not recommended, but handy)
    local = _extract_local_zip_if_present()
    if local:
        BASE = local
    else:
        st.set_page_config(page_title="Mohammadpur Spatial Risk", layout="wide")
        st.error(
            "Missing bundle. Provide one of the following:\n\n"
            "1) **Secrets**: add\n"
            '```toml\nBUNDLE_URL = "https://.../streamlit_bundle_full_YYYYMMDDTHHMMSSZ.zip"\n```\n'
            "2) **Advanced settings → Environment variables**: set `BUNDLE_URL`\n"
            "3) **Env var** `MVP_BASE` pointing to an unzipped bundle directory\n"
            "4) Place a local ZIP named `streamlit_bundle_full_*.zip` in the repo root\n"
        )
        st.stop()

# ---------- PATHS ----------
MODEL_CAL = os.path.join(BASE, "models", "lgbm_isotonic.pkl")
MODEL_RAW = os.path.join(BASE, "models", "lgbm_raw.pkl")  # optional
META_JSON = os.path.join(BASE, "models", "features.json")
FEATS_PARQ= os.path.join(BASE, "features", "training.parquet")
GRID_GJ   = os.path.join(BASE, "h3_grid.geojson")
EVAL_SUM  = os.path.join(BASE, "metrics", "eval_summary.csv")
HRK_SW    = os.path.join(BASE, "metrics", "hrk_sweep.csv")
SECT_CSV  = os.path.join(BASE, "ops", "patrol_sectors.csv")
TOPK_CSV  = os.path.join(BASE, "ops", "topk_hexes.csv")
SECT_GJ   = os.path.join(BASE, "ops", "patrol_sectors.geojson")

# ---------- PAGE ----------
st.set_page_config(page_title="Mohammadpur Spatial Risk", layout="wide")
st.markdown(
    '''<div style="padding:10px;border-left:6px solid #ffcc00;background:#fff9e6">
<b>Policy</b>: <i>Spatial risk only.</i> <b>No person-level profiling.</b> Decision support, not a warrant.
</div>''',
    unsafe_allow_html=True
)
st.sidebar.caption(f"Bundle base: {BASE}")

# ---------- HELPERS ----------
def union_all_safe(geom_series):
    try:
        return geom_series.union_all()
    except Exception:
        return unary_union(list(geom_series))

@st.cache_data(show_spinner=False)
def load_grid(path):
    g = gpd.read_file(path).to_crs(4326)
    g["h3"] = g["h3"].astype(str)
    return g

@st.cache_resource(show_spinner=False)
def load_models():
    with open(MODEL_CAL, "rb") as f:
        clf_cal = pickle.load(f)
    clf_raw = None
    if os.path.exists(MODEL_RAW):
        try:
            with open(MODEL_RAW, "rb") as f:
                clf_raw = pickle.load(f)
        except Exception:
            pass
    with open(META_JSON, "r") as f:
        meta = json.load(f)
    return clf_cal, clf_raw, meta

@st.cache_data(show_spinner=False)
def parquet_available_columns():
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(FEATS_PARQ).schema.names)
    except Exception:
        return list(pd.read_parquet(FEATS_PARQ, engine="pyarrow").columns)

@st.cache_data(show_spinner=False)
def months_from_ops_or_parquet():
    for p in (SECT_CSV, TOPK_CSV):
        if os.path.exists(p):
            d = pd.read_csv(p, parse_dates=["date"])
            return sorted(d["date"].dt.to_period("M").astype(str).unique())
    d = pd.read_parquet(FEATS_PARQ, columns=["date"], engine="pyarrow")
    return sorted(pd.to_datetime(d["date"]).dt.to_period("M").astype(str).unique())

@st.cache_data(show_spinner=False)
def load_month_slice(month_str, feat_cols, extra_cols):
    p = pd.Period(month_str, "M")
    start = pd.Timestamp(p.start_time).to_datetime64()
    end   = (pd.Timestamp(p.end_time) + pd.Timedelta(seconds=1)).to_datetime64()
    avail = set(parquet_available_columns())
    cols  = [c for c in (set(feat_cols) | set(extra_cols)) if c in avail]
    if "date" not in cols: cols.append("date")
    if "h3"   not in cols: cols.append("h3")
    X = pd.read_parquet(
        FEATS_PARQ, engine="pyarrow", columns=cols,
        filters=[("date", ">=", start), ("date", "<", end)]
    )
    X["date"] = pd.to_datetime(X["date"]).dt.normalize()
    return X

def infer_dayparts(df):
    if "daypart" in df.columns:
        return sorted(df["daypart"].astype(str).unique()), "column"
    dps = [c for c in df.columns if c.startswith("daypart_")]
    return sorted([c.replace("daypart_", "") for c in dps]), "dummies"

# ---------- SANITY ----------
for pth, label in [(GRID_GJ, "grid"), (MODEL_CAL, "model"), (META_JSON, "meta"), (FEATS_PARQ, "features")]:
    if not os.path.exists(pth):
        st.error(f"Missing {label}: {pth}")
        st.stop()

grid = load_grid(GRID_GJ)
clf_cal, clf_raw, meta = load_models()
feat_cols   = meta.get("feat_cols", [])
used_nowcast= bool(meta.get("used_nowcast", False))

# ---------- SIDEBAR ----------
months = months_from_ops_or_parquet()
if not months:
    st.error("No months found.")
    st.stop()

sel_month = st.sidebar.selectbox("Month", options=months, index=len(months)-1)
TOPK      = st.sidebar.slider("Top-K % (per day, per daypart)", 1, 10, 5)
risk_mode = st.sidebar.radio("Risk mode", ["pro_only", "blend (0.6/0.4)"], index=0)

# ---------- LOAD THIS MONTH ----------
with st.spinner(f"Loading {sel_month}…"):
    maybe_dp = ["daypart", "daypart_00-06", "daypart_06-12", "daypart_12-18", "daypart_18-24"]
    extra    = ["date", "h3", "ri_norm"] + maybe_dp
    Xmon     = load_month_slice(sel_month, feat_cols, extra)

dayparts, dp_mode = infer_dayparts(Xmon)
if not dayparts:
    dayparts, dp_mode = ["12-18"], "dummies"

sel_dp   = st.sidebar.selectbox("Daypart", options=dayparts, index=min(2, len(dayparts)-1))
dates_in = sorted(Xmon["date"].unique())
if not dates_in:
    st.warning("No dates in this month.")
    st.stop()
sel_date = st.sidebar.selectbox("Date", options=[d.date() for d in dates_in])
sel_date = pd.Timestamp(sel_date)

# ---------- FILTER & ALIGN ----------
Df = Xmon[Xmon["date"] == sel_date].copy()
if dp_mode == "column":
    Df = Df[Df["daypart"].astype(str) == sel_dp]
else:
    dpcol = f"daypart_{sel_dp}"
    if dpcol in Df.columns:
        Df = Df[Df[dpcol] == 1]

for c in feat_cols:
    if c not in Df.columns:
        Df[c] = 0.0
Df = Df.drop(columns=[c for c in Df.columns if c not in feat_cols and c not in {"date", "h3", "ri_norm"}], errors="ignore")
rep  = Df[["date", "h3"] + (["ri_norm"] if "ri_norm" in Df.columns else [])].copy()
Xsel = Df[feat_cols].copy()

# ---------- PREDICT ----------
with st.spinner("Scoring…"):
    pro = clf_cal.predict_proba(Xsel)[:, 1]
ri   = rep["ri_norm"].to_numpy() if "ri_norm" in rep.columns else np.zeros_like(pro)
risk = pro if (risk_mode == "pro_only" or used_nowcast) else (0.6 * pro + 0.4 * ri)

pred = pd.DataFrame({"h3": rep["h3"].astype(str), "risk": risk})
top_per_h3 = pred.groupby("h3", as_index=False)["risk"].max().sort_values("risk", ascending=False)
k = max(1, int(len(top_per_h3) * TOPK / 100.0))
top = top_per_h3.head(k)

st.subheader(f"Top-{TOPK}% — {sel_date.date()} {sel_dp} (n={len(top)})")
dl = top.copy()
dl.insert(0, "date", sel_date.date())
dl.insert(1, "daypart", sel_dp)
st.download_button("Download Top-K hexes (CSV)", dl.to_csv(index=False).encode("utf-8"),
                   file_name=f"topk_{sel_date.date()}_{sel_dp.replace(':','-')}.csv")

# ---------- MAP ----------
use_folium = st.toggle("Use Folium map (fallback)", value=False)

try:
    joined = grid.merge(top, on="h3", how="inner")
    if joined.empty:
        st.info("No Top-K hexes for this selection.")
    else:
        # Ensure numeric and normalize safely (no .ptp())
        joined["risk"] = pd.to_numeric(joined["risk"], errors="coerce").fillna(0.0)
        rmin = float(joined["risk"].min())
        rmax = float(joined["risk"].max())
        denom = (rmax - rmin) if (rmax > rmin) else 1.0
        joined["risk_norm"] = (joined["risk"] - rmin) / denom

        # Precompute RGBA in Python (deck.gl JSON can't call functions)
        # Orange fill, alpha from 110..240 based on risk_norm
        alpha = (110 + (joined["risk_norm"] * 130)).round().astype(int).clip(0, 255)
        joined["fill_rgba"] = [[255, 136, 0, int(a)] for a in alpha]

        # Map center
        u = union_all_safe(joined.geometry)
        center = [float(u.centroid.y), float(u.centroid.x)]

        if not use_folium:
            # PyDeck (WebGL)
            layer = pdk.Layer(
                "GeoJsonLayer",
                data=joined.to_json(),
                get_fill_color="properties.fill_rgba",   # <-- read from column
                get_line_color=[0, 0, 0, 180],
                get_line_width=1,
                line_width_min_pixels=1,
                pickable=True,
            )
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=pdk.ViewState(
                        latitude=center[0], longitude=center[1], zoom=13
                    ),
                )
            )
        else:
            # Folium fallback (no WebGL)
            import folium
            m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")
            folium.GeoJson(
                joined.to_json(),
                name="TopK",
                style_function=lambda feat: {
                    "color": "#ff8800",
                    "weight": 1.5,
                    # fill opacity tied to risk_norm (0.3..1.0)
                    "fillOpacity": float(feat["properties"].get("risk_norm", 0)) * 0.7 + 0.3,
                },
                highlight_function=lambda feat: {"weight": 3, "color": "#000000"},
            ).add_to(m)
            from streamlit.components.v1 import html
            html(m._repr_html_(), height=520)

except Exception as e:
    st.warning(f"Map render failed: {e}")
    st.dataframe(top.head(20))
# ---------- VIGILANCE BY DAYPART (month summary) ----------
st.subheader("Vigilance by daypart (month summary)")

metric_choice = st.radio(
    "Show by", ["dwell minutes per day", "risk sum per day"], index=0, horizontal=True
)

def _month_filter(df, month_str):
    return df[df["date"].dt.to_period("M").astype(str) == month_str].copy()

def _safe_bar(series, label):
    if series.empty:
        st.info("No data for the selected month.")
        return
    st.bar_chart(series.sort_values(ascending=False), height=240)
    st.caption(label)

summary_df = None
note = ""

if os.path.exists(SECT_CSV):
    # Best source: patrol sectors (already merged & weighted)
    sect_all = pd.read_csv(SECT_CSV, parse_dates=["date"])
    sect_m   = _month_filter(sect_all, sel_month)

    # per (date, daypart) totals → average per daypart
    dd = (sect_m.groupby(["date","daypart"], as_index=False)[["dwell_min","risk_sum"]].sum())
    g  = dd.groupby("daypart", as_index=False).agg(
            days        = ("date","nunique"),
            dwell_total = ("dwell_min","sum"),
            risk_total  = ("risk_sum","sum"),
        )
    g["dwell_per_day"] = g["dwell_total"] / g["days"].where(g["days"]>0, 1)
    g["risk_per_day"]  = g["risk_total"]  / g["days"].where(g["days"]>0, 1)

    if metric_choice.startswith("dwell"):
        series = g.set_index("daypart")["dwell_per_day"]
        label  = "Average dwell minutes per day by daypart (this month)"
    else:
        series = g.set_index("daypart")["risk_per_day"]
        label  = "Average risk sum per day by daypart (this month)"

    pct = 100 * series / series.sum() if series.sum() else series*0
    summary_df = pd.DataFrame({"avg_per_day": series.round(2), "share_%": pct.round(1)})
    note = "Source: patrol_sectors.csv"

elif os.path.exists(TOPK_CSV):
    # Fallback: aggregate Top-K hex risk by (date, daypart)
    tk = pd.read_csv(TOPK_CSV, parse_dates=["date"])
    tkm = _month_filter(tk, sel_month)
    dd  = tkm.groupby(["date","daypart"], as_index=False)["risk"].sum()
    g   = dd.groupby("daypart", as_index=False).agg(
            days       = ("date","nunique"),
            risk_total = ("risk","sum"),
        )
    g["risk_per_day"] = g["risk_total"] / g["days"].where(g["days"]>0, 1)
    series = g.set_index("daypart")["risk_per_day"]
    pct    = 100 * series / series.sum() if series.sum() else series*0
    summary_df = pd.DataFrame({"avg_per_day": series.round(3), "share_%": pct.round(1)})
    label = "Average Top-K risk per day by daypart (this month)"
    note  = "Source: topk_hexes.csv (fallback)"

else:
    st.info("No sectors or Top-K files found; add ops/patrol_sectors.csv or ops/topk_hexes.csv to enable this summary.")

if summary_df is not None:
    st.dataframe(
        summary_df.rename(columns={"avg_per_day":"avg", "share_%":"share %"}),
        use_container_width=True,
    )
    _safe_bar(summary_df["avg_per_day"], label)
    st.caption(f"Use this to weight patrol hours across dayparts. {note}")


# ---------- SECTORS ----------
st.subheader("Patrol sectors")
SECT_OK = os.path.exists(SECT_CSV) and os.path.exists(SECT_GJ)
if SECT_OK:
    sect = pd.read_csv(SECT_CSV, parse_dates=["date"])
    s = sect[(sect["date"] == sel_date) & (sect["daypart"] == sel_dp)].sort_values("sector_rank")
    if s.empty:
        st.info("No exported sectors for this day/daypart.")
    else:
        st.dataframe(
            s[["sector_id", "n_hex", "risk_sum", "area_ha", "dwell_min", "sector_rank"]],
            use_container_width=True,
        )
        st.download_button("Download sectors (CSV)", data=open(SECT_CSV, "rb"), file_name="patrol_sectors.csv")
else:
    st.info("ops/patrol_sectors.* not found (optional).")

# ---------- WHY HERE ----------
st.subheader("Why here?")
show_shap = st.checkbox("Compute SHAP (slower)", value=False)
sel_hex = st.selectbox("Inspect hex (H3)", options=top["h3"].tolist() if len(top) else [])
if sel_hex:
    row_X = Xsel.loc[rep["h3"].astype(str) == sel_hex]
    if row_X.empty:
        st.info("No feature row for that hex.")
    else:
        if show_shap:
            try:
                import shap
                model_for_shap = None
                if os.path.exists(MODEL_RAW):
                    with open(MODEL_RAW, "rb") as f:
                        model_for_shap = pickle.load(f)
                if model_for_shap is None:
                    model_for_shap = getattr(clf_cal, "base_estimator", None) or getattr(clf_cal, "estimator", None)
                sv = shap.TreeExplainer(model_for_shap).shap_values(row_X)
                if isinstance(sv, list):
                    sv = sv[1]
                vals = sv[0]
                idx = np.argsort(np.abs(vals))[::-1][:12]
                contrib = pd.DataFrame({"feature": np.array(feat_cols)[idx], "shap": vals[idx]}).round(5)
                st.dataframe(contrib, use_container_width=True)
            except Exception as e:
                st.info(f"SHAP not available ({e}). Showing feature magnitudes.")
                st.dataframe(row_X.iloc[0].abs().sort_values(ascending=False).head(12).to_frame("value"))
        else:
            st.dataframe(row_X.iloc[0].abs().sort_values(ascending=False).head(12).to_frame("value"))

# ---------- REPORTS ----------
st.subheader("Reports")
c1, c2 = st.columns(2)
with c1:
    if os.path.exists(EVAL_SUM):
        st.download_button("Download eval summary", data=open(EVAL_SUM, "rb"), file_name="eval_summary.csv")
with c2:
    if os.path.exists(HRK_SW):
        st.download_button("Download HR@K sweep", data=open(HRK_SW, "rb"), file_name="hrk_sweep.csv")

with st.expander("Limitations & notes"):
    st.markdown("""
- Data are **sparse**; incidents may be **under-reported**; some positives are **augmented** during training.
- Predictions are **area-based**; **no person-level** inference.
- Use with **local thana logs** and professional judgment.
""")
