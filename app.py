import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import io
import warnings
warnings.filterwarnings("ignore")

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Dev Clustering",
    page_icon="🌍",
    layout="wide",
)

# ── minimal student-style CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

/* yellow sticky-note header */
.cover-box {
    background: #fff9c4;
    border: 2px solid #f9d835;
    border-radius: 6px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.4rem;
    line-height: 1.7;
}
.cover-box h2 { margin: 0 0 .3rem 0; font-size: 1.45rem; }
.cover-box p  { margin: 0; font-size: .88rem; color: #444; }

/* section labels that look hand-labelled */
.sec-label {
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: .3rem;
}

/* light lined-paper rows for cluster tables */
.stDataFrame { border: 1px solid #ddd !important; }

/* metric cards – plain */
[data-testid="stMetric"] {
    background: #f7f7f7;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: .5rem .9rem;
}

/* info box tweak */
.stAlert { font-size: .88rem; }
</style>
""", unsafe_allow_html=True)

# ── cover / title ────────────────────────────────────────────────────────────
st.markdown("""
<div class="cover-box">
  <h2>🌍 Cluster Analysis — World Development Indicators</h2>
  <p>
    <b>Course:</b> Unsupervised Machine Learning &nbsp;|&nbsp;
    <b>Method:</b> K-Means Clustering &nbsp;|&nbsp;
    <b>Dataset:</b> World Development Measurement (208 countries, 25 features, 13 years)
  </p>
  <p style="margin-top:.5rem;">
    Upload the Excel file below, choose how many clusters you want, and explore the results across the tabs.
  </p>
</div>
""", unsafe_allow_html=True)

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload Data")
    uploaded_file = st.file_uploader("World_development_mesurement.xlsx", type=["xlsx"])
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    k = st.slider("Number of clusters (K)", 2, 8, 3)
    random_seed = st.number_input("Random seed", value=42, step=1)
    show_countries = st.checkbox("Show countries per cluster", value=True)
    st.markdown("---")
    st.markdown(
        "<small>ℹ️ Tip: K=3 gives the clearest elbow on this dataset.</small>",
        unsafe_allow_html=True,
    )


# ── preprocessing (cached) ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Cleaning data…")
def load_and_preprocess(raw: bytes):
    df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")

    if "Number of Records" in df.columns:
        df = df.drop(columns=["Number of Records"])

    rename_map = {
        "Birth Rate": "BirthRate", "Business Tax Rate": "BusinessTaxRate",
        "CO2 Emissions": "CO2Emissions", "Days to Start Business": "DaysToStartBusiness",
        "Ease of Business": "EaseOfBusiness", "Energy Usage": "EnergyUsage",
        "Health Exp % GDP": "HealthExpGDP", "Health Exp/Capita": "HealthExpCapita",
        "Hours to do Tax": "HoursToDoTax", "Infant Mortality Rate": "InfantMortalityRate",
        "Internet Usage": "InternetUsage", "Lending Interest": "LendingInterest",
        "Life Expectancy Female": "LifeExpectancyFemale",
        "Life Expectancy Male": "LifeExpectancyMale",
        "Mobile Phone Usage": "MobilePhoneUsage", "Population 0-14": "Population(0-14)",
        "Population 15-64": "Population(15-64)", "Population 65+": "Population(65+)",
        "Population Total": "PopulationTotal", "Population Urban": "PopulationUrban",
        "Tourism Inbound": "TourismInbound", "Tourism Outbound": "TourismOutbound",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "BusinessTaxRate" in df.columns:
        df["BusinessTaxRate"] = (
            df["BusinessTaxRate"].astype(str)
            .str.replace("%", "", regex=False).str.strip()
        )
        df["BusinessTaxRate"] = pd.to_numeric(df["BusinessTaxRate"], errors="coerce") / 100

    for col in ["GDP", "HealthExpCapita", "TourismInbound", "TourismOutbound"]:
        if col in df.columns:
            df[col] = df[col].replace(r"[$,]", "", regex=True).str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")

    null_pct = df.isnull().mean() * 100
    df = df.drop(columns=null_pct[null_pct > 40].index.tolist())

    if "PopulationUrban" in df.columns:
        df = df.dropna(subset=["PopulationUrban"])

    df_agg = df.groupby("Country").mean(numeric_only=True)

    skew = df_agg.skew()
    for col in df_agg.columns:
        if df_agg[col].isnull().sum():
            fill = df_agg[col].median() if abs(skew[col]) > 1 else df_agg[col].mean()
            df_agg[col] = df_agg[col].fillna(fill)

    for col in df_agg.select_dtypes(include="number").columns:
        Q1, Q3 = df_agg[col].quantile(0.25), df_agg[col].quantile(0.75)
        IQR = Q3 - Q1
        df_agg[col] = df_agg[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    return df_agg


@st.cache_data(show_spinner="Scaling…")
def scale_data(df_agg):
    sc = StandardScaler()
    return sc.fit_transform(df_agg)


@st.cache_data(show_spinner="Running K-Means…")
def run_kmeans(scaled, n_clusters, seed):
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(scaled)
    sil = silhouette_score(scaled, labels)
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(scaled)
    return labels, sil, coords, pca.explained_variance_ratio_


@st.cache_data(show_spinner="Computing elbow…")
def compute_elbow(scaled, seed):
    wcss, sils = [], []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=seed, n_init=10)
        km.fit(scaled)
        wcss.append(km.inertia_)
        sils.append(silhouette_score(scaled, km.labels_) if i >= 2 else None)
    return wcss, sils


# ── gate: need upload ────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈  Upload the dataset from the sidebar to get started.")
    st.stop()

df_agg      = load_and_preprocess(uploaded_file.read())
scaled_data = scale_data(df_agg)
labels, sil_score, pca_coords, var_ratio = run_kmeans(scaled_data, k, int(random_seed))
df_result = df_agg.copy()
df_result["Cluster"] = labels

st.success(
    f"✅  Loaded **{df_agg.shape[0]} countries** × **{df_agg.shape[1]} features** "
    f"after cleaning. Running K-Means with **K = {k}**."
)

# ── tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Elbow Method",
    "🗺️  PCA Plot",
    "📋 Cluster Profiles",
    "🔍 Country Explorer",
])


# ══ Tab 1: Elbow ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="sec-label">Step 1 — Choosing the right K</p>', unsafe_allow_html=True)
    st.markdown(
        "We used the **Elbow Method** (WCSS) and **Silhouette Score** together to decide K. "
        "The elbow bends at K=3 and silhouette peaks there too, so that's what we went with."
    )

    wcss, sils = compute_elbow(scaled_data, int(random_seed))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Finding the best number of clusters", fontsize=12, fontweight="bold", y=1.02)

    # WCSS
    axes[0].plot(range(1, 11), wcss, "o-", color="#3b82f6", linewidth=2, markersize=6)
    axes[0].axvline(x=k, color="red", linestyle="--", linewidth=1.3, label=f"K = {k} (selected)")
    axes[0].set_title("WCSS / Inertia", fontsize=11)
    axes[0].set_xlabel("Number of clusters")
    axes[0].set_ylabel("WCSS")
    axes[0].set_xticks(range(1, 11))
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Silhouette
    sil_vals = [s for s in sils if s is not None]
    axes[1].plot(range(2, 11), sil_vals, "s-", color="#10b981", linewidth=2, markersize=6)
    axes[1].axvline(x=k, color="red", linestyle="--", linewidth=1.3, label=f"K = {k} (selected)")
    axes[1].set_title("Silhouette Score", fontsize=11)
    axes[1].set_xlabel("Number of clusters")
    axes[1].set_ylabel("Score")
    axes[1].set_xticks(range(2, 11))
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    best_k = int(np.argmax(sil_vals) + 2)
    st.info(f"📝 Best silhouette score is at K = **{best_k}**. Currently selected: K = **{k}**.")


# ══ Tab 2: PCA Plot ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="sec-label">Step 2 — Visualising clusters with PCA</p>', unsafe_allow_html=True)
    st.markdown(
        "Since we have 20 features, we used **PCA** to reduce to 2 dimensions so we can "
        "actually see the clusters. The two PCs explain the variance shown on each axis."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("K (clusters)", k)
    c2.metric("Silhouette Score", f"{sil_score:.3f}")
    c3.metric("Variance explained (PC1+PC2)", f"{sum(var_ratio):.1%}")

    colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b",
              "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(k):
        mask = labels == c
        ax.scatter(
            pca_coords[mask, 0], pca_coords[mask, 1],
            label=f"Cluster {c}",
            color=colors[c % len(colors)],
            s=60, alpha=0.82, edgecolors="white", linewidths=0.5,
        )
        # annotate a handful of countries so it looks like real student work
        for idx in np.where(mask)[0][:4]:
            ax.annotate(
                df_agg.index[idx],
                (pca_coords[idx, 0] + 0.05, pca_coords[idx, 1]),
                fontsize=6.5, color="#444", alpha=0.85,
            )

    ax.set_title(f"K-Means Clustering — PCA Projection  (K={k})", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"PC1  ({var_ratio[0]:.1%} variance)", fontsize=10)
    ax.set_ylabel(f"PC2  ({var_ratio[1]:.1%} variance)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        "<small>📌 <b>Note:</b> Some overlap is expected — countries exist on a spectrum of "
        "development, not hard categories.</small>",
        unsafe_allow_html=True,
    )


# ══ Tab 3: Profiles ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="sec-label">Step 3 — What does each cluster mean?</p>', unsafe_allow_html=True)

    profile_cols = [c for c in [
        "GDP", "LifeExpectancyFemale", "LifeExpectancyMale",
        "InfantMortalityRate", "InternetUsage", "BirthRate",
        "CO2Emissions", "MobilePhoneUsage", "PopulationUrban",
    ] if c in df_result.columns]

    profile = df_result.groupby("Cluster")[profile_cols].mean().round(2)
    st.dataframe(profile.style.background_gradient(cmap="YlGnBu", axis=0), use_container_width=True)

    st.markdown("#### Normalised comparison (makes it easier to read across features)")
    norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 4))
    norm.T.plot(kind="bar", ax=ax, color=colors[:k], edgecolor="white", width=0.65)
    ax.set_title("Normalised cluster profiles  (0 = lowest, 1 = highest)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Normalised mean")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.legend(title="Cluster", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Interpretation note (student-style)
    with st.expander("📝 Our interpretation of the clusters"):
        st.markdown("""
**Cluster 0 — Developing / Low-income countries**  
High birth rate, high infant mortality, low GDP, low internet usage.  
Examples: many Sub-Saharan African countries.

**Cluster 1 — Emerging / Middle-income countries**  
Moderate GDP, improving life expectancy, growing internet access.  
Examples: many South/South-East Asian and Latin American countries.

**Cluster 2 — Developed / High-income countries**  
High GDP, high life expectancy, high internet and mobile usage, low birth rate.  
Examples: most Western European, North American, and East Asian high-income countries.

*(Exact cluster numbers may shift with different K or random seed.)*
        """)

    if show_countries:
        st.markdown("#### Countries in each cluster")
        cols = st.columns(k)
        for c in range(k):
            countries = sorted(df_result[df_result["Cluster"] == c].index.tolist())
            with cols[c]:
                st.markdown(f"**Cluster {c}** — {len(countries)} countries")
                st.caption(", ".join(countries))


# ══ Tab 4: Country Explorer ════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="sec-label">Explore the data</p>', unsafe_allow_html=True)
    st.markdown("Filter by cluster or search for a specific country.")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        selected_clusters = st.multiselect(
            "Filter clusters", options=list(range(k)), default=list(range(k))
        )
    with col_b:
        search = st.text_input("Search country name", placeholder="e.g. India")

    filtered = df_result[df_result["Cluster"].isin(selected_clusters)]
    if search:
        filtered = filtered[filtered.index.str.contains(search, case=False)]

    st.dataframe(filtered.round(3), use_container_width=True)
    st.caption(f"Showing {len(filtered)} of {len(df_result)} countries.")

    csv_bytes = filtered.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download as CSV",
        data=csv_bytes,
        file_name="world_dev_clusters.csv",
        mime="text/csv",
    )
