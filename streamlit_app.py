import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Vaccine Adoption (Interactive)", layout="wide")

# ---------- Data ----------
@st.cache_data
def load_intro(path: str) -> pd.DataFrame:
    intro_df = pd.read_excel(path)
    # Keep only introductions marked "Yes"
    df = intro_df.loc[intro_df["INTRO"].eq("Yes")].copy()
    # standardize columns we use
    df.rename(columns={
        "DESCRIPTION": "Vaccine",
        "YEAR": "Year",
        "COUNTRYNAME": "Country"
    }, inplace=True)
    # guard types
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Vaccine"] = df["Vaccine"].astype(str)
    df["Country"] = df["Country"].astype(str)
    # De-dupe (one Country×Vaccine×Year)
    df = df.drop_duplicates(["Vaccine", "Year", "Country"])
    return df

DATA_PATH = "./data_sources/vaccine-introduction-data.xlsx"
df = load_intro(DATA_PATH)

# Universe (you can change 194 if you’re using a different country universe)
GLOBAL_DENOM = 194

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())

year_range = st.sidebar.slider("Year range", min_year, max_year, (2000, max_year), step=1)

all_countries = sorted(df["Country"].unique().tolist())
selected_countries = st.sidebar.multiselect("Countries (optional)", all_countries, default=[])

mode = st.sidebar.radio(
    "Mode",
    ["Global proportion", "Selected-countries proportion", "Single-country view"],
    index=0
)

# ---------- Prepare data ----------
df_year = df.query("@year_range[0] <= Year <= @year_range[1]").copy()

def make_heatmap_proportion(base_df: pd.DataFrame, denominator: int) -> pd.DataFrame:
    """
    Returns Vaccine x Year proportions (0..1)
    """
    counts = (
        base_df
        .groupby(["Vaccine", "Year"])["Country"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index(key=lambda s: s.str.lower())
    )
    props = counts.div(float(max(denominator, 1)))  # avoid zero-div
    # ensure integer year columns sorted
    props.columns = [int(c) for c in props.columns]
    props = props.reindex(sorted(props.columns), axis=1)
    return props

plot_title = "Proportion of countries adopting vaccines in their national immunization programme"
subtitle = "The continuous scale represents the proportion of countries where a vaccine is administered country-wide."

if mode == "Global proportion" or (mode == "Selected-countries proportion" and len(selected_countries) == 0):
    # Global (or selected-countries not chosen → fall back to global)
    base = df_year
    denom = GLOBAL_DENOM
    mode_note = "Global"
elif mode == "Selected-countries proportion" and len(selected_countries) > 0:
    base = df_year[df_year["Country"].isin(selected_countries)].copy()
    denom = len(selected_countries)
    mode_note = f"{denom} selected countr{'y' if denom==1 else 'ies'}"
else:
    # Single-country view
    base = df_year[df_year["Country"].isin(selected_countries)].copy()

# ---------- Plot ----------
st.title("Proportion of countries adopting vaccines in their national immunization programme")
st.subheader("Interactively filter by years and countries")
st.caption("Source: WHO Immunization Data portal (static extract). Visualization built with Streamlit + Matplotlib.")

if mode in ["Global proportion", "Selected-countries proportion"]:
    heat = make_heatmap_proportion(base, denom)

    # sort vaccines by the rightmost (latest) column descending
    latest_year = heat.columns.max()
    heat = heat.sort_values(by=latest_year, ascending=False)

    # right-hand labels for latest year
    labels_right = heat[latest_year].map(lambda v: f"{v:.0%}")

    n_rows, n_cols = heat.shape
    fig, ax = plt.subplots(figsize=(12, max(6, n_rows * 0.35)))

    im = ax.imshow(heat.values, aspect='auto', cmap='Blues', vmin=0, vmax=1)

    # ticks/labels
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(heat.columns, rotation=45, ha='center')
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(heat.index)

    # remove borders/ticks
    for s in ('top','right','left','bottom'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

    # cell gridlines
    edge_w = 1.0
    ax.vlines(np.arange(-0.5, n_cols, 1), -0.5, n_rows-0.5, colors='white', lw=edge_w, zorder=2)
    ax.hlines(np.arange(-0.5, n_rows, 1), -0.5, n_cols-0.5, colors='white', lw=edge_w, zorder=2)

    # right-hand latest-year % labels
    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(np.arange(n_rows))
    ax_right.set_yticklabels(labels_right)
    ax_right.tick_params(axis='y', right=True, labelright=True, length=0)
    for s in ('top','right','left','bottom'):
        ax_right.spines[s].set_visible(False)

    plt.suptitle(f"{plot_title}", fontsize=16, fontweight='bold', y=0.98)
    plt.figtext(0.00, -0.02,
                f"{subtitle}\nMode: {mode_note}. Denominator = {denom}. Year range: {year_range[0]}–{year_range[1]}.",
                ha='left', color='gray', fontsize=9)

    st.pyplot(fig, clear_figure=True)
    st.info(f"Mode: {mode_note}  •  Denominator: {denom}  •  Latest year shown on right: {latest_year}")

elif mode == "Single-country view":
    # Expect exactly one country; if many selected, let the user disambiguate
    if len(selected_countries) == 0:
        st.warning("Select one country in the sidebar to see its adoption grid.")
    else:
        country = selected_countries[0]
        st.subheader(f"Adoption by vaccine for {country} ({year_range[0]}–{year_range[1]})")

        # Build a binary grid: Vaccine x Year ∈ {0,1} for this country
        base_c = base[base["Country"] == country]
        if base_c.empty:
            st.info("No introductions recorded for the selected period.")
        else:
            pivot = (
                base_c.assign(val=1)
                      .pivot_table(index="Vaccine", columns="Year", values="val",
                                   aggfunc="max", fill_value=0)
                      .sort_index(key=lambda s: s.str.lower())
            )
            # ensure all years in range present as columns
            years_full = list(range(year_range[0], year_range[1] + 1))
            for y in years_full:
                if y not in pivot.columns:
                    pivot[y] = 0
            pivot = pivot[sorted(pivot.columns)]

            n_rows, n_cols = pivot.shape
            fig, ax = plt.subplots(figsize=(12, max(6, n_rows * 0.35)))
            im = ax.imshow(pivot.values, aspect='auto', cmap='Blues', vmin=0, vmax=1)

            ax.set_xticks(np.arange(n_cols))
            ax.set_xticklabels(pivot.columns, rotation=45, ha='center')
            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels(pivot.index)

            for s in ('top','right','left','bottom'):
                ax.spines[s].set_visible(False)
            ax.tick_params(axis='both', which='both', length=0)

            edge_w = 1.0
            ax.vlines(np.arange(-0.5, n_cols, 1), -0.5, n_rows-0.5, colors='white', lw=edge_w, zorder=2)
            ax.hlines(np.arange(-0.5, n_rows, 1), -0.5, n_cols-0.5, colors='white', lw=edge_w, zorder=2)

            plt.suptitle(f"Vaccine introductions in {country}", fontsize=16, fontweight='bold', y=0.98)
            plt.figtext(0.00, -0.02,
                        f"1 = introduced in that year (national). Year range: {year_range[0]}–{year_range[1]}.",
                        ha='left', color='gray', fontsize=9)

            st.pyplot(fig, clear_figure=True)

# ---------- Optional: export the current table ----------
st.download_button(
    "Download filtered data (CSV)",
    data=df_year.to_csv(index=False).encode("utf-8"),
    file_name=f"vaccine_intro_filtered_{year_range[0]}_{year_range[1]}.csv",
    mime="text/csv"
)

import io, requests, pandas as pd, numpy as np, streamlit as st
import plotly.express as px
import plotly.graph_objects as go

OWID_URL = "https://ourworldindata.org/grapher/global-vaccination-coverage.csv?v=1&csvType=full&useColumnShortNames=true"

VAX_COLS = {
    "coverage__antigen_dtpcv3": "DTP3",
    "coverage__antigen_pol3": "Polio (3 doses)",
    "coverage__antigen_mcv1": "Measles (MCV1)",
    "coverage__antigen_rcv1": "Rubella (RCV1)",
}

DEFAULT_ENTITIES = ["High-income countries", "Low-income countries"]
NICE_ENTITY = {"High-income countries": "High-income", "Low-income countries": "Low-income"}

@st.cache_data(ttl=3600)
def load_owid_df():
    r = requests.get(OWID_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df, r.headers.get("Last-Modified")

raw, last_mod = load_owid_df()

# --- Controls ---
st.sidebar.header("Controls (Plotly)")
vax_key = st.sidebar.selectbox("Vaccine", list(VAX_COLS.keys()), format_func=lambda k: VAX_COLS[k])

entities_all = sorted(raw["Entity"].unique().tolist())
entities_default = [e for e in DEFAULT_ENTITIES if e in entities_all] or entities_all[:2]
entities = st.sidebar.multiselect("Entities", entities_all, default=entities_default)

year_min = int(pd.to_numeric(raw["Year"], errors="coerce").min())
year_max = int(pd.to_numeric(raw["Year"], errors="coerce").max())
yr0, yr1 = st.sidebar.slider("Year range", year_min, year_max, (max(1980, year_min), year_max), step=1)

smooth = st.sidebar.checkbox("Smooth (rolling mean)", value=False)
window = st.sidebar.slider("Smoothing window (years)", 3, 11, 5, step=2, disabled=not smooth)
shade_gap = st.sidebar.checkbox("Shade gap (only if exactly 2 entities)", value=True)

# --- Tidy & filter ---
df = raw.loc[raw["Entity"].isin(entities), ["Entity", "Year", vax_key]].copy()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df[(df["Year"] >= yr0) & (df["Year"] <= yr1)]

# Convert to %, if needed
if df[vax_key].max(skipna=True) <= 1.5:
    df[vax_key] = df[vax_key] * 100.0

df["Entity_nice"] = df["Entity"].map(lambda x: NICE_ENTITY.get(x, x))

if smooth:
    df = (df.sort_values(["Entity_nice", "Year"])
            .groupby("Entity_nice", group_keys=False)
            .apply(lambda g: g.assign(**{vax_key: g[vax_key].rolling(window, center=True,
                                                                     min_periods=max(1, window//2)).mean()})))

# --- Plotly line chart ---
title = f"Share of 1-year-olds vaccinated — {VAX_COLS[vax_key]}"
subtitle = f"Hover to inspect. Click legend to toggle series. Range: {yr0}–{yr1}. Data last modified: {last_mod or 'unknown'}."

fig = px.line(
    df, x="Year", y=vax_key, color="Entity_nice",
    labels={"Year": "", vax_key: "Coverage (%)", "Entity_nice": "Entities"},
)

# Consistent 0–100% y-axis
fig.update_yaxes(range=[0, 100], ticksuffix="%", title="")
fig.update_xaxes(title="")

# Thicken lines a touch
for tr in fig.data:
    tr.update(line=dict(width=2))

# Optional: shade the gap when exactly two entities selected
unique_entities = df["Entity_nice"].unique().tolist()
if shade_gap and len(unique_entities) == 2:
    pivot = (df.pivot_table(index="Year", columns="Entity_nice", values=vax_key, aggfunc="mean")
               .reset_index().dropna())
    top = pivot[unique_entities].max(axis=1)
    bot = pivot[unique_entities].min(axis=1)

    # add invisible bottom trace to anchor fill
    fig.add_trace(go.Scatter(x=pivot["Year"], y=bot,
                             mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip", name="_gap_bot"))
    # fill to next y (top)
    fig.add_trace(go.Scatter(x=pivot["Year"], y=top,
                             mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(0,0,0,0.15)",
                             name="Gap", hoverinfo="skip"))

st.markdown(f"### {title}")
st.caption(subtitle)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

# Optional data table + download
with st.expander("Show data"):
    out = df.rename(columns={"Entity_nice": "Entity", vax_key: "Coverage (%)"})[["Entity", "Year", "Coverage (%)"]]
    st.dataframe(out.sort_values(["Entity", "Year"]), use_container_width=True, height=350)
    st.download_button(
        "Download shown data (CSV)",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"owid_{VAX_COLS[vax_key].replace(' ','_')}_{yr0}_{yr1}.csv",
        mime="text/csv"
    )

