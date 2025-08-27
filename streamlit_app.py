# streamlit_app.py
# -*- coding: utf-8 -*-
import os, json, pickle, logging, glob, zipfile, pathlib, re
import numpy as np, pandas as pd, geopandas as gpd
import streamlit as st, pydeck as pdk
from shapely.ops import unary_union
from app_utils import ensure_bundle  # downloads & extracts when BUNDLE_URL is provided
import numpy as np

# Build friendly labels for the hex dropdown (e.g., "S1 • risk 0.873 • 618065…1039")
def build_hex_dropdown_labels(hex_list, df_with_risk, grid_gdf, sect_today_gdf=None):
    """
    hex_list: list[str] of H3 ids to show (usually the Top-K list)
    df_with_risk: DataFrame with columns ["h3","risk"] at least (e.g., `top`)
    grid_gdf: GeoDataFrame with columns ["h3","geometry"] (crs=4326)
    sect_today_gdf: optional GeoDataFrame of sectors for the selected date/daypart
    """
    import pandas as pd
    import numpy as np

    # risk & rank for the options we are showing
    d = pd.DataFrame({"h3": pd.Series(hex_list, dtype=str)})
    r = df_with_risk[["h3","risk"]].copy()
    r["h3"] = r["h3"].astype(str)
    d = d.merge(r, on="h3", how="left").sort_values("risk", ascending=False).reset_index(drop=True)
    d["rank"] = d.index + 1

    # optional: sector membership (robust)
    d["sector_rank"] = None
    if sect_today_gdf is not None and not sect_today_gdf.empty:
        sectors = sect_today_gdf.to_crs(4326)
        gidx = grid_gdf.set_index("h3")["geometry"]
        for i, row in d.iterrows():
            h = str(row["h3"])
            if h not in gidx.index:
                continue
            # representative_point() is guaranteed inside polygon
            pt = gidx.loc[h].representative_point()
            # 'intersects' will also match boundary cases
            hit = sectors[sectors.intersects(pt)]
            if not hit.empty:
                try:
                    d.at[i, "sector_rank"] = int(hit.iloc[0].get("sector_rank"))
                except Exception:
                    d.at[i, "sector_rank"] = None

    def trunc(h):
        h = str(h)
        return h if len(h) <= 10 else f"{h[:6]}…{h[-4:]}"
    def make_label(r):
        s = f"S{int(r['sector_rank'])}" if pd.notna(r["sector_rank"]) else "—"
        risk = f"{float(r['risk']):.3f}" if pd.notna(r["risk"]) else "n/a"
        return f"{s} • risk {risk} • {trunc(r['h3'])}"

    return {str(r["h3"]): make_label(r) for _, r in d.iterrows()}


def allocate_dwell_minutes(
    risks,                 # 1D list/array of sector risk_sum
    total=360,             # total minutes available for this time-of-day
    min_each=15,           # hard minimum minutes per sector
    max_share=0.60,        # at most 60% of 'total' can go to any one sector
    alpha=0.6,             # temper spiky risks (1.0=proportional; 0.5–0.7 flattens)
    mix_equal=0.25         # blend some equal split (0–1) to smooth further
):
    """Return integer minutes per sector (sum == total)."""
    risks = np.asarray(risks, dtype=float)
    n = len(risks)
    total = int(total)
    if n == 0:
        return []

    # 1) start with hard floor
    base = np.full(n, float(min_each))
    remaining = total - base.sum()

    if remaining <= 0:
        # not enough time even for floor: split the total evenly
        x = np.full(n, total / n, dtype=float)
    else:
        # 2) tempered weights + a bit of equal split
        if risks.sum() <= 0:
            w = np.ones(n) / n
        else:
            w = risks ** alpha
            w = w / w.sum()
            w = mix_equal * (np.ones(n) / n) + (1 - mix_equal) * w
            sw = w.sum()
            if sw <= 0:
                w = np.ones(n) / n
            else:
                w = w / sw

        x = base + remaining * w

        # 3) cap large allocations and re-distribute overflow safely
        cap = float(max_share * total)
        for _ in range(5):  # a few passes in case capping creates new overflow
            over = x > cap
            if not over.any():
                break
            overflow_sum = float((x[over] - cap).sum())
            x[over] = cap

            can = ~over
            if not can.any() or overflow_sum <= 1e-9:
                break

            # weights for receivers only
            w2 = w.copy()
            w2[over] = 0.0
            s2 = w2.sum()
            if s2 <= 0:
                w2 = can.astype(float)
                s2 = w2.sum()

            # add overflow back (vectorized, no shape mismatch)
            x += overflow_sum * (w2 / s2)

    # 4) make integers (largest remainder) and keep sum == total
    x = np.maximum(x, 0.0)
    flo = np.floor(x).astype(int)
    diff = int(total - flo.sum())
    if diff > 0:
        order = np.argsort(x - flo)[::-1]
        flo[order[:diff]] += 1
    elif diff < 0:
        # very rare; trim minutes from largest allocations
        order = np.argsort(flo)[::-1]
        for i in range(-diff):
            flo[order[i % n]] -= 1

    return flo.tolist()



# ---------- PAGE ----------
st.set_page_config(page_title="Mohammadpur Spatial Risk", layout="wide")

# Quiet SHAP noise
logging.getLogger("shap").setLevel(logging.ERROR)

# Keep last picked hex in session (for click-to-select)
if "sel_hex" not in st.session_state:
    st.session_state["sel_hex"] = None

# ---------- BUNDLE LOCATION (robust) ----------
def _extract_local_zip_if_present() -> str | None:
    """If a bundle ZIP exists next to the app, extract under ./data/bundle_local."""
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
    local = _extract_local_zip_if_present()
    if local:
        BASE = local
    else:
        st.error(
            "Missing bundle. Provide one of the following:\n\n"
            "1) **Secrets**: add\n"
            '```toml\nBUNDLE_URL = "https://.../streamlit_bundle_full_YYYYMMDDTHHMMSSZ.zip"\n```'
            "\n2) **Advanced settings → Environment variables**: set `BUNDLE_URL`"
            "\n3) **Env var** `MVP_BASE` pointing to an unzipped bundle directory"
            "\n4) Place a local ZIP named `streamlit_bundle_full_*.zip` in the repo root"
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

# ---------- BANNER ----------
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
    with open(MODEL_CAL,"rb") as f: clf_cal = pickle.load(f)
    clf_raw = None
    if os.path.exists(MODEL_RAW):
        try:
            with open(MODEL_RAW,"rb") as f: clf_raw = pickle.load(f)
        except Exception: pass
    with open(META_JSON,"r") as f: meta = json.load(f)
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
    cols  = [c for c in (set(feat_cols)|set(extra_cols)) if c in avail]
    if "date" not in cols: cols.append("date")
    if "h3"   not in cols: cols.append("h3")
    X = pd.read_parquet(FEATS_PARQ, engine="pyarrow", columns=cols,
                        filters=[("date", ">=", start), ("date", "<", end)])
    X["date"] = pd.to_datetime(X["date"]).dt.normalize()
    return X

# Sector lookup after grid is loaded
def find_sector_for_hex(sel_hex: str, sel_date: pd.Timestamp, sel_dp: str, grid_gdf: gpd.GeoDataFrame):
    if not (os.path.exists(SECT_GJ) and sel_hex and sel_dp):
        return None
    try:
        sg = gpd.read_file(SECT_GJ).to_crs(4326)
    except Exception:
        return None
    if "date" in sg.columns:
        try: sg["date"] = pd.to_datetime(sg["date"]).dt.normalize()
        except: pass
    mask = (sg.get("daypart") == sel_dp)
    if "date" in sg.columns:
        mask &= (sg["date"] == pd.to_datetime(sel_date).normalize())
    today_sectors = sg.loc[mask].copy()
    if today_sectors.empty:
        return None
    try:
        hex_geom = grid_gdf.loc[grid_gdf["h3"].astype(str) == sel_hex, "geometry"].iloc[0]
    except Exception:
        return None
    hit = today_sectors.loc[today_sectors.intersects(hex_geom)]
    if hit.empty:
        return None
    return hit.iloc[0]

# ---------- LOAD GRID & MODELS ----------
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

TOPK = st.sidebar.slider("Places to cover (%)", 1, 10, 5)
st.sidebar.caption("We flag the top **K%** of grid cells for each time-of-day, every day. Example: 5% ≈ the riskiest 1 in 20 cells.")

risk_choice = st.sidebar.radio(
    "Risk score used",
    ["Model only (recommended)", "Model + recent activity (60/40)"],
    index=0
)
risk_mode = "pro_only" if risk_choice.startswith("Model only") else "blend"
st.sidebar.caption("**Model only** = calibrated probability. **Model + recent** = 60% model + 40% short-term activity.")

# ---------- LOAD SELECTED MONTH ----------
with st.spinner(f"Loading {sel_month}…"):
    maybe_dp = ["daypart","daypart_00-06","daypart_06-12","daypart_12-18","daypart_18-24"]
    extra    = ["date","h3","ri_norm"] + maybe_dp
    Xmon     = load_month_slice(sel_month, feat_cols, extra)

# Daypart & date selection
if "daypart" in Xmon.columns:
    dayparts = sorted(Xmon["daypart"].astype(str).unique())
    dp_mode  = "column"
else:
    dps = [c for c in Xmon.columns if c.startswith("daypart_")]
    dayparts = sorted([c.replace("daypart_","") for c in dps]) or ["12-18"]
    dp_mode  = "dummies"

sel_dp   = st.sidebar.selectbox("Time of day", options=dayparts, index=min(2,len(dayparts)-1))
dates_in = sorted(Xmon["date"].unique())
if not dates_in:
    st.warning("No dates in this month."); st.stop()
sel_date = st.sidebar.selectbox("Date", options=[d.date() for d in dates_in])
sel_date = pd.Timestamp(sel_date)

# ---------- FILTER & ALIGN ----------
Df = Xmon[Xmon["date"]==sel_date].copy()
if dp_mode=="column":
    Df = Df[Df["daypart"].astype(str)==sel_dp]
else:
    dpcol=f"daypart_{sel_dp}"
    if dpcol in Df.columns:
        Df = Df[Df[dpcol]==1]

# add missing feature columns as zeros
for c in feat_cols:
    if c not in Df.columns:
        Df[c]=0.0
# drop extras not needed for model
Df = Df.drop(columns=[c for c in Df.columns if c not in feat_cols and c not in {"date","h3","ri_norm"}], errors="ignore")

rep  = Df[["date","h3"] + (["ri_norm"] if "ri_norm" in Df.columns else [])].copy()
Xsel = Df[feat_cols].copy()

# ---------- PREDICT ----------
with st.spinner("Scoring…"):
    pro = clf_cal.predict_proba(Xsel)[:,1]
ri  = rep["ri_norm"].to_numpy() if "ri_norm" in rep.columns else np.zeros_like(pro)
risk= pro if (risk_mode=="pro_only" or used_nowcast) else (0.6*pro + 0.4*ri)

pred = pd.DataFrame({"h3": rep["h3"].astype(str), "risk": risk})
top_per_h3 = pred.groupby("h3", as_index=False)["risk"].max().sort_values("risk", ascending=False)
k = max(1, int(len(top_per_h3) * TOPK / 100.0))
top = top_per_h3.head(k)

st.subheader(f"Top-{TOPK}% — {sel_date.date()} {sel_dp} (n={len(top)})")
dl = top.copy(); dl.insert(0,"date",sel_date.date()); dl.insert(1,"daypart",sel_dp)
st.download_button("Download Top-K hexes (CSV)", dl.to_csv(index=False).encode("utf-8"),
                   file_name=f"topk_{sel_date.date()}_{sel_dp.replace(':','-')}.csv")

# ---------- MAP ----------
use_folium = st.toggle("Use Folium map (fallback)", value=False)

try:
    joined = grid.merge(top, on="h3", how="inner")
    if joined.empty:
        st.info("No Top-K cells for this selection.")
    else:
        # numeric + normalize
        joined["risk"] = pd.to_numeric(joined["risk"], errors="coerce").fillna(0.0)
        rmin, rmax = float(joined["risk"].min()), float(joined["risk"].max())
        denom = (rmax - rmin) if (rmax > rmin) else 1.0
        joined["risk_norm"] = (joined["risk"] - rmin) / denom
        joined = joined.sort_values("risk", ascending=False).reset_index(drop=True)
        joined["rank"] = joined.index + 1
        alpha = (110 + (joined["risk_norm"] * 130)).round().astype(int).clip(0, 255)
        joined["fill_rgba"] = [[255, 136, 0, int(a)] for a in alpha]

        # Whitelist JSON-safe props
        gj = joined[["h3","risk","risk_norm","rank","fill_rgba","geometry"]].copy()

        # Center
        u = union_all_safe(gj.geometry)
        center = [float(u.centroid.y), float(u.centroid.x)]

        if not use_folium:
            # ---- PYDECK ----
            geojson_dict = json.loads(gj.to_json())
            layer = pdk.Layer(
                "GeoJsonLayer",
                data=geojson_dict,
                filled=True, stroked=True, opacity=0.7,
                get_fill_color="properties.fill_rgba",
                get_line_color=[0,0,0,180], line_width_min_pixels=1,
                pickable=True, auto_highlight=True,
            )
            tooltip = {
                "html": "<b>H3:</b> {h3}<br/>"
                        "<b>Risk score:</b> {risk}<br/>"
                        "<b>Priority:</b> {rank}",
                "style": {"backgroundColor":"rgba(0,0,0,0.7)","color":"white"},
            }
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=13),
                    tooltip=tooltip,
                )
            )

        else:
            # ---- FOLIUM (click-to-select) ----
            import folium
            from streamlit_folium import st_folium

            m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")
            folium.GeoJson(
                json.loads(gj.to_json()),
                name="TopK cells",
                style_function=lambda feat: {
                    "color": "#000000",
                    "weight": 1,
                    "fillColor": "#ff8800",
                    "fillOpacity": float(feat["properties"].get("risk_norm", 0)) * 0.7 + 0.3,
                },
                highlight_function=lambda feat: {"weight": 3, "color": "#000"},
                tooltip=folium.GeoJsonTooltip(
                    fields=["h3","risk","rank"],
                    aliases=["H3","Risk score","Priority"], localize=True, sticky=True
                ),
                popup=folium.GeoJsonPopup(
                    fields=["h3","risk","rank"],
                    aliases=["H3","Risk score","Priority"], localize=True
                ),
            ).add_to(m)

            # Optional: sectors (ensure JSON-safe fields)
            if os.path.exists(SECT_GJ):
                sect_gdf = gpd.read_file(SECT_GJ).to_crs(4326)
                mask = pd.Series([True]*len(sect_gdf))
                if "daypart" in sect_gdf.columns:
                    mask &= (sect_gdf["daypart"] == sel_dp)
                if "date" in sect_gdf.columns:
                    sect_gdf["date_str"] = pd.to_datetime(sect_gdf["date"]).dt.strftime("%Y-%m-%d")
                    mask &= (pd.to_datetime(sect_gdf["date"]).dt.normalize() == sel_date.normalize())
                sect_today = sect_gdf.loc[mask]
                if not sect_today.empty:
                    sgj = sect_today[["geometry","sector_id","sector_rank","daypart"]].copy()
                    if "date_str" in sect_today.columns:
                        sgj["date_str"] = sect_today["date_str"]
                    folium.GeoJson(
                        json.loads(sgj.to_json()),
                        name="Patrol sectors",
                        style_function=lambda f: {"color":"#0066CC","weight":2,"fillOpacity":0.05},
                        highlight_function=lambda f: {"weight":4,"color":"#003366"},
                        tooltip=folium.GeoJsonTooltip(
                            fields=[c for c in ["sector_id","sector_rank","date_str"] if c in sgj.columns],
                            aliases=["Sector","Priority","Date"][: len([c for c in ["sector_id","sector_rank","date_str"] if c in sgj.columns])],
                            localize=True, sticky=True
                        ),
                    ).add_to(m)

            out = st_folium(m, height=520, use_container_width=True)
            props = (out or {}).get("last_object_clicked", {}).get("properties", {})
            if props:
                if "h3" in props:
                    st.session_state["sel_hex"] = str(props["h3"])
                    st.rerun()
                elif "sector_id" in props and 'sect_today' in locals() and not sect_today.empty:
                    hit = sect_today.loc[sect_today["sector_id"] == str(props["sector_id"])]
                    if not hit.empty:
                        poly = hit.geometry.iloc[0]
                        gj["__c"] = gj.geometry.centroid
                        inside = gj.loc[gj["__c"].within(poly)]
                        if not inside.empty:
                            best = inside.sort_values("risk", ascending=False).iloc[0]
                            st.session_state["sel_hex"] = str(best["h3"])
                            st.rerun()

except Exception as e:
    st.warning(f"Map render failed: {e}")
    st.dataframe(top.head(20))

# ---------- VIGILANCE BY DAYPART (month summary) ----------
st.subheader("Vigilance by daypart (month summary)")
daily_budget_min = st.number_input("Total patrol minutes per day (across all times)", 60, 1440, 480, 30)
source = st.radio("Compute share using", ["Incidents (labels)", "Sector risk (ops)", "Top-K risk (fallback)"], index=1, horizontal=True)

def _month_filter(df, m):
    return df[df["date"].dt.to_period("M").astype(str) == m].copy()

def _bar(series, label):
    if series.empty:
        st.info("No data for the selected month."); return
    st.bar_chart(series.sort_values(ascending=False), height=240); st.caption(label)

summary = None
note = ""
if source.startswith("Incidents"):
    # Try to compute from labels in Xmon (slim parquet must include y or y_next7_any)
    Xlab = Xmon.copy()
    if "daypart" not in Xlab.columns:
        dp_cols = [c for c in Xlab.columns if c.startswith("daypart_")]
        if dp_cols:
            parts = []
            for c in dp_cols:
                dp = c.replace("daypart_","")
                parts.append(Xlab[Xlab[c]==1].assign(daypart=dp))
            Xlab = pd.concat(parts, ignore_index=True) if parts else Xlab.assign(daypart="unknown")
        else:
            Xlab = Xlab.assign(daypart="unknown")
    label_col = "y" if "y" in Xlab.columns else ("y_next7_any" if "y_next7_any" in Xlab.columns else None)
    if label_col is None:
        st.info("No label column ('y' or 'y_next7_any') in features; switch to Sector risk or Top-K risk.")
    else:
        dd = Xlab.groupby(["date","daypart"], as_index=False)[label_col].sum(numeric_only=True)
        month_totals = dd.groupby("daypart", as_index=False)[label_col].sum().rename(columns={label_col:"month_count"})
        total = month_totals["month_count"].sum()
        g = (month_totals.set_index("daypart")["month_count"] / total) if total else (month_totals.set_index("daypart")["month_count"]*0+0.25)
        rec = (g * daily_budget_min).round(0).astype(int)
        summary = pd.DataFrame({"avg_share_%": (g*100).round(1), "recommended_min_per_day": rec}).sort_values("avg_share_%", ascending=False)
        note = "Source: incident labels from features"
elif source.startswith("Sector"):
    if os.path.exists(SECT_CSV):
        sect_all = pd.read_csv(SECT_CSV, parse_dates=["date"])
        dd = _month_filter(sect_all, sel_month).groupby("daypart", as_index=False)["risk_sum"].sum()
        total = dd["risk_sum"].sum()
        g = (dd.set_index("daypart")["risk_sum"] / total) if total else (dd.set_index("daypart")["risk_sum"]*0+0.25)
        rec = (g * daily_budget_min).round(0).astype(int)
        summary = pd.DataFrame({"avg_share_%": (g*100).round(1), "recommended_min_per_day": rec}).sort_values("avg_share_%", ascending=False)
        note = "Source: patrol_sectors.csv"
    else:
        st.info("Missing ops/patrol_sectors.csv; switch to labels or Top-K risk.")
else:
    if os.path.exists(TOPK_CSV):
        tk = pd.read_csv(TOPK_CSV, parse_dates=["date"])
        dd = _month_filter(tk, sel_month).groupby("daypart", as_index=False)["risk"].sum()
        total = dd["risk"].sum()
        g = (dd.set_index("daypart")["risk"] / total) if total else (dd.set_index("daypart")["risk"]*0+0.25)
        rec = (g * daily_budget_min).round(0).astype(int)
        summary = pd.DataFrame({"avg_share_%": (g*100).round(1), "recommended_min_per_day": rec}).sort_values("avg_share_%", ascending=False)
        note = "Source: topk_hexes.csv"
    else:
        st.info("Missing ops/topk_hexes.csv; switch to labels or Sector risk.")
if summary is not None and not summary.empty:
    st.dataframe(summary.rename_axis("daypart"), use_container_width=True)
    _bar(summary["recommended_min_per_day"], "Recommended patrol minutes per day by time-of-day")
    st.caption(note)

# ---------- SECTORS ----------
st.subheader("Patrol sectors")

SECT_OK = os.path.exists(SECT_CSV) and os.path.exists(SECT_GJ)
if SECT_OK:
    sect = pd.read_csv(SECT_CSV, parse_dates=["date"])
    s = sect[(sect["date"] == sel_date) & (sect["daypart"] == sel_dp)].copy()
    if s.empty:
        st.info("No exported sectors for this day/time.")
    else:
        # Friendlier labels/derived fields
        s["Priority"] = s["sector_rank"].astype(int)
        s["Date"]     = pd.to_datetime(s["sector_id"].str.slice(0,8), format="%Y%m%d").dt.date
        s["Time"]     = s["sector_id"].str.extract(r"_(\d{2}-\d{2})_")[0]
        s["Label"]    = s.apply(lambda r: f"{r['Date']} {r['Time']} • S{r['Priority']}", axis=1)

        # Risk level bands relative to this day/time
        q50, q80, q95 = s["risk_sum"].quantile([0.50, 0.80, 0.95]).tolist()
        def band(v):
            if v <= q50: return "Low"
            if v <= q80: return "Medium"
            if v <= q95: return "High"
            return "Very high"
        s["Risk level"]       = s["risk_sum"].apply(band)
        s["Risk score"]       = s["risk_sum"].round(3)
        s["Risk share %"]     = (100 * s["risk_sum"] / s["risk_sum"].sum()).round(1)
        s["Cells"]            = s["n_hex"].astype(int)
        s["Area (ha)"]        = s["area_ha"].round(1)

        # ✅ Recalculate "Suggested minutes" for display using the allocator
        TOTAL_MIN = 6 * 60  # minutes per time-of-day (tune if needed)
        s = s.sort_values("Priority").reset_index(drop=True)
        s["Suggested minutes"] = allocate_dwell_minutes(
            s["risk_sum"].values,
            total=TOTAL_MIN,
            min_each=15,    # floor per sector
            max_share=0.60, # cap any one sector
            alpha=0.6,      # temper spiky risk
            mix_equal=0.25  # add some equal split
        )

        # Display table
        s_disp = s[[
            "Label","Priority","Risk level","Risk score","Risk share %","Cells","Area (ha)","Suggested minutes"
        ]].rename(columns={"Label":"Sector"})

        def _color_risk(col):
            colors = {"Low":"#E3F2FD","Medium":"#FFF3E0","High":"#FFEBEE","Very high":"#FFCDD2"}
            return [f"background-color: {colors.get(v,'')}" for v in col]
        try:
            st.dataframe(s_disp.style.apply(_color_risk, subset=["Risk level"]), use_container_width=True)
        except Exception:
            st.dataframe(s_disp, use_container_width=True)

        # Small glossary right under the table
        with st.expander("What do these terms mean?"):
            st.markdown("""
- **Grid cell (H3):** a small hexagon tile (~200 m) covering the map.  
- **Cell risk score:** how risky this cell is today for the selected time-of-day.  
- **Patrol sector:** a group of adjacent flagged cells merged into one patrol area.  
- **Priority:** rank of a sector among today’s sectors (1 = most urgent).  
- **Suggested minutes:** recommended time to spend in that area for this time-of-day.
""")

        # Downloads: friendly + raw
        st.download_button(
            "Download sectors (friendly CSV)",
            data=s_disp.to_csv(index=False).encode("utf-8"),
            file_name=f"patrol_sectors_{sel_date.date()}_{sel_dp.replace(':','-')}_friendly.csv",
        )
        st.download_button(
            "Download sectors (raw CSV)",
            data=open(SECT_CSV, "rb").read(),
            file_name="patrol_sectors_raw.csv",
        )
else:
    st.info("ops/patrol_sectors.* not found (optional).")


# ---------- WHY HERE ----------
st.subheader("Why here?")

# Build options tied to session state (so map clicks update dropdown)
# sectors for today/time (for labeling only; safe if file missing)
# sectors for today/time (for labeling only)
sect_today = None
if os.path.exists(SECT_GJ):
    try:
        sg = gpd.read_file(SECT_GJ).to_crs(4326)
        mask = pd.Series(True, index=sg.index)
        if "daypart" in sg.columns:
            mask &= (sg["daypart"] == sel_dp)
        if "date" in sg.columns:
            mask &= (pd.to_datetime(sg["date"]).dt.normalize() == sel_date.normalize())
        sect_today = sg.loc[mask].copy()
    except Exception:
        sect_today = None

options = top["h3"].astype(str).tolist()
labels_map = build_hex_dropdown_labels(options, top, grid, sect_today)

if st.session_state.get("sel_hex") in options:
    default_idx = options.index(st.session_state["sel_hex"])
else:
    default_idx = 0

sel_hex = st.selectbox(
    "Inspect grid cell (H3)",
    options=options,
    index=default_idx,
    format_func=lambda x: labels_map.get(str(x), str(x)),
    key="sel_hex_widget"
)
st.session_state["sel_hex"] = sel_hex
st.caption(f"Selected H3: {sel_hex}")




# Friendly labels/formatters
def friendly_label(col: str) -> str | None:
    if col == "ri_norm": return "Recent activity score (0–1)"
    if col == "neighbor_lag7d": return "Nearby incidents in last 7 days (avg)"
    if col == "same_daypart_last_week": return "Same time last week (flag)"
    if col == "days_since_last": return "Days since last incident"
    if col == "neighbor_dsl_min": return "Days since last in nearby cells (min)"
    m = re.match(r"^dist_(.+)_m$", col)
    if m:
        base = m.group(1).replace("_"," ").title().replace("Busstops","Bus stops").replace("Roads Primary","Primary road").replace("Roads Secondary","Secondary road")
        return f"Distance to {base} (m)"
    m = re.match(r"^cnt_(.+)_(\d+)m$", col)
    if m:
        what, r = m.groups()
        base = what.replace("_"," ").title().replace("Busstops","Bus stops")
        return f"{base} within {r} m (count)"
    m = re.match(r"^lag(\d+)d$", col)
    if m: return f"Incident {m.group(1)} days ago (flag)"
    m = re.match(r"^past(\d+)_sum$", col)
    if m: return f"Incidents in past {m.group(1)} days (sum)"
    m = re.match(r"^past(\d+)_exp$", col)
    if m: return f"Recent {m.group(1)}-day activity (weighted)"
    return None

def fmt_value(col: str, val):
    try: x = float(val)
    except: return val
    if col.startswith(("dist_","cnt_","days_since")) or re.match(r"^lag\d+d$", col):
        return int(round(x))
    if col in {"same_daypart_last_week"}: return int(round(x))
    if col in {"ri_norm"} or col.startswith("past") or col=="neighbor_lag7d":
        return round(x, 3)
    return round(x, 3)

if sel_hex:
    # risk & priority for this hex (today)
    today = pred.copy(); today["_h3"] = today["h3"].astype(str)
    risk_here = float(today.loc[today["_h3"]==sel_hex, "risk"].max()) if (today["_h3"]==sel_hex).any() else np.nan
    today = today.sort_values("risk", ascending=False).reset_index(drop=True); today["rank"] = today.index + 1
    if (today["_h3"]==sel_hex).any():
        rank_here = int(today.loc[today["_h3"]==sel_hex, "rank"].min())
        pct = 100.0 * (1.0 - (rank_here / len(today)))
        band = ("Very high" if pct >= 95 else "High" if pct >= 80 else "Medium" if pct >= 50 else "Low")
        st.markdown(f"**Risk score:** {risk_here:.3f}  |  **Priority:** {rank_here}  |  **Level:** {band} (top {pct:.1f}%)")
    else:
        st.markdown("Risk score: n/a")

    # which sector contains this hex?
    sector_row = find_sector_for_hex(sel_hex, sel_date, sel_dp, grid)
    if sector_row is not None:
        sid = str(sector_row.get("sector_id"))
        if os.path.exists(SECT_CSV):
            s_all = pd.read_csv(SECT_CSV, parse_dates=["date"])
            s_one = s_all[(s_all["date"]==sel_date) & (s_all["daypart"]==sel_dp) & (s_all["sector_id"]==sid)]
            if not s_one.empty:
                info = s_one.iloc[0]
                st.info(
                    f"This cell is in **Sector S{int(info['sector_rank'])}** "
                    f"(Priority {int(info['sector_rank'])}) — "
                    f"**Suggested minutes:** {int(info['dwell_min'])}, "
                    f"**Cells:** {int(info['n_hex'])}, "
                    f"**Area:** {round(float(info['area_ha']),1)} ha."
                )
            else:
                st.info(f"This cell is in **{sid}**.")
        else:
            st.info(f"This cell is in **{sid}**.")
    else:
        st.info("This cell is not in today’s Top-K selection for this time-of-day.")

    # friendly feature table
    row_X = Xsel.loc[rep["h3"].astype(str) == sel_hex]
    if row_X.empty:
        st.info("No feature row for that cell.")
    else:
        drop_like = [c for c in row_X.columns if c.endswith("_x") or c.endswith("_y") or c.startswith("daypart_")]
        row = row_X.drop(columns=drop_like, errors="ignore").iloc[0]
        items = []
        for col, val in row.items():
            if col in {"date","h3"}: continue
            lab = friendly_label(col)
            if not lab: continue
            items.append({"Feature": lab, "Value": fmt_value(col, val)})
        if items:
            df_nice = pd.DataFrame(items)
            def _grp(name):
                if "Distance" in name: return "Distances"
                if "within" in name:   return "Counts"
                if "Recent" in name or "last" in name or "flag" in name: return "Recent history"
                return "Other"
            df_nice["Group"] = df_nice["Feature"].map(_grp)
            order = {"Recent history":0, "Counts":1, "Distances":2, "Other":3}
            df_nice["__o"] = df_nice["Group"].map(order).fillna(9)
            df_nice = df_nice.sort_values(["__o","Feature"]).drop(columns="__o").reset_index(drop=True)
            st.dataframe(df_nice, use_container_width=True)
        else:
            st.info("No interpretable features to display.")

# after: st.dataframe(df_nice, use_container_width=True)

HELP_MD = """
**Feature** – the signal the model uses for this grid cell.  
**Value** – this cell’s current value for that signal (rounded).  
**Group** – which family the signal belongs to:

- **Recent history**: short-term activity near here (e.g., incident flags `3/7/28` days ago,
  “nearby incidents in last 7 days”, “recent 7-day activity”). Higher recent activity ⇒ **higher risk**.
- **Counts**: how many places within a radius (e.g., **“Bus stops within 300 m (count)”**).
  More relevant places nearby ⇒ can **raise risk** (crowding/exposure).
- **Distances**: how far to the nearest place (**“Distance to Primary road (m)”**, etc.).
  Smaller distance ⇒ **closer context**, often **higher risk**.
- **Other**: helpful flags like **“Same time last week”**.
"""

with st.expander("How to read this table"):
    st.markdown(HELP_MD)

    # Optional: SHAP drivers (advanced)
    show_shap = st.checkbox("Show top drivers (SHAP — slower)", value=False)
    if show_shap and not row_X.empty:
        try:
            import shap
            model_for_shap = None
            if os.path.exists(MODEL_RAW):
                with open(MODEL_RAW,"rb") as f: model_for_shap = pickle.load(f)
            if model_for_shap is None:
                model_for_shap = getattr(clf_cal,"base_estimator",None) or getattr(clf_cal,"estimator",None)
            sv = shap.TreeExplainer(model_for_shap).shap_values(row_X)
            if isinstance(sv, list): sv = sv[1]
            contrib = pd.Series(sv[0], index=row_X.columns).drop(index=drop_like, errors="ignore")
            top10 = contrib.reindex([c for c in contrib.index if friendly_label(c)]).sort_values(key=lambda s:-np.abs(s)).head(10)
            if not top10.empty:
                st.write("Top drivers: positive values ↑ increase risk; negative values ↓ reduce risk.")
                st.dataframe(pd.DataFrame({
                    "Feature": [friendly_label(c) for c in top10.index],
                    "Impact": [round(float(v),4) for v in top10.values]
                }), use_container_width=True)
            else:
                st.info("No interpretable drivers to show.")
        except Exception as e:
            st.info(f"SHAP not available ({e}).")
