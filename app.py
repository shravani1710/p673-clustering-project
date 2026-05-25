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

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Development Clustering",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 World Development — K-Means Cluster Analysis")
st.markdown(
    "Upload the **World_development_mesurement.xlsx** file to run the full "
    "K-Means pipeline interactively."
)

# ─── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader(
        "Upload dataset (.xlsx)", type=["xlsx"]
    )
    k = st.slider("Number of clusters (K)", min_value=2, max_value=8, value=3)
    random_state = st.number_input("Random seed", value=42, step=1)
    show_countries = st.checkbox("Show country lists per cluster", value=True)

# ─── Helper: load & preprocess ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & preprocessing data…")
def load_and_preprocess(file_bytes: bytes):
    data = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    df = data.copy()

    # Drop administrative column
    if "Number of Records" in df.columns:
        df = df.drop(columns=["Number of Records"])

    # Rename columns
    rename_map = {
        "Birth Rate": "BirthRate",
        "Business Tax Rate": "BusinessTaxRate",
        "CO2 Emissions": "CO2Emissions",
        "Days to Start Business": "DaysToStartBusiness",
        "Ease of Business": "EaseOfBusiness",
        "Energy Usage": "EnergyUsage",
        "GDP": "GDP",
        "Health Exp % GDP": "HealthExpGDP",
        "Health Exp/Capita": "HealthExpCapita",
        "Hours to do Tax": "HoursToDoTax",
        "Infant Mortality Rate": "InfantMortalityRate",
        "Internet Usage": "InternetUsage",
        "Lending Interest": "LendingInterest",
        "Life Expectancy Female": "LifeExpectancyFemale",
        "Life Expectancy Male": "LifeExpectancyMale",
        "Mobile Phone Usage": "MobilePhoneUsage",
        "Population 0-14": "Population(0-14)",
        "Population 15-64": "Population(15-64)",
        "Population 65+": "Population(65+)",
        "Population Total": "PopulationTotal",
        "Population Urban": "PopulationUrban",
        "Tourism Inbound": "TourismInbound",
        "Tourism Outbound": "TourismOutbound",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Clean currency / percentage symbols
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

    # Drop high-null columns (>40 %)
    null_pct = df.isnull().mean() * 100
    high_null = null_pct[null_pct > 40].index.tolist()
    df = df.drop(columns=high_null)

    # Drop rows where PopulationUrban is null
    if "PopulationUrban" in df.columns:
        df = df.dropna(subset=["PopulationUrban"])

    # Aggregate to one row per country
    df_agg = df.groupby("Country").mean(numeric_only=True)

    # Impute based on skewness
    skewness = df_agg.skew()
    for col in df_agg.columns:
        if df_agg[col].isnull().sum() > 0:
            if abs(skewness[col]) > 1:
                df_agg[col] = df_agg[col].fillna(df_agg[col].median())
            else:
                df_agg[col] = df_agg[col].fillna(df_agg[col].mean())

    # Outlier capping (IQR)
    for col in df_agg.select_dtypes(include="number").columns:
        Q1, Q3 = df_agg[col].quantile(0.25), df_agg[col].quantile(0.75)
        IQR = Q3 - Q1
        df_agg[col] = df_agg[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    return df_agg


@st.cache_data(show_spinner="Scaling features…")
def scale(df_agg):
    scaler = StandardScaler()
    return scaler.fit_transform(df_agg), df_agg.columns.tolist()


# ─── Main app ──────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈  Upload the dataset using the sidebar to get started.")
    st.stop()

file_bytes = uploaded_file.read()
df_agg = load_and_preprocess(file_bytes)
scaled_data, feature_names = scale(df_agg)

st.success(f"✅  Dataset loaded — **{df_agg.shape[0]} countries**, **{df_agg.shape[1]} features**")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Elbow Method", "🗺️ PCA Visualisation", "📋 Cluster Profiles", "🔍 Explore Data"]
)

# ── Tab 1: Elbow ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Elbow Method — Within-Cluster Sum of Squares (WCSS)")

    @st.cache_data(show_spinner="Computing WCSS…")
    def compute_wcss(scaled, seed):
        wcss, scores = [], []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, random_state=seed, n_init=10)
            km.fit(scaled)
            wcss.append(km.inertia_)
            if i >= 2:
                lbl = km.predict(scaled)
                scores.append(silhouette_score(scaled, lbl))
            else:
                scores.append(None)
        return wcss, scores

    wcss, sil_scores = compute_wcss(scaled_data, int(random_state))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # WCSS plot
    axes[0].plot(range(1, 11), wcss, marker="o", linewidth=2, color="steelblue", markersize=7)
    axes[0].axvline(x=k, color="red", linestyle="--", linewidth=1.4, label=f"Selected K = {k}")
    axes[0].set_title("WCSS vs Number of Clusters")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("WCSS (Inertia)")
    axes[0].set_xticks(range(1, 11))
    axes[0].legend()

    # Silhouette plot
    axes[1].plot(range(2, 11), sil_scores[1:], marker="s", linewidth=2,
                 color="darkorange", markersize=7)
    axes[1].axvline(x=k, color="red", linestyle="--", linewidth=1.4, label=f"Selected K = {k}")
    axes[1].set_title("Silhouette Score vs Number of Clusters")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_xticks(range(2, 11))
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    best_k = int(np.argmax(sil_scores[1:]) + 2)
    st.info(f"💡 The highest silhouette score is at **K = {best_k}**. "
            f"Your selected K = **{k}**.")

# ── Tab 2: PCA Visualisation ────────────────────────────────────────────────────
with tab2:
    st.subheader(f"K-Means Clustering — PCA 2D Projection (K = {k})")

    @st.cache_data(show_spinner="Running K-Means…")
    def run_kmeans(scaled, n_clusters, seed):
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = km.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(scaled)
        return labels, score, coords, pca.explained_variance_ratio_

    labels, score, pca_coords, var_ratio = run_kmeans(scaled_data, k, int(random_state))
    df_result = df_agg.copy()
    df_result["Cluster"] = labels

    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters", k)
    col2.metric("Silhouette Score", f"{score:.4f}")
    col3.metric("PCA Variance Captured", f"{sum(var_ratio):.1%}")

    palette = plt.cm.get_cmap("tab10", k)
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(k):
        mask = labels == c
        ax.scatter(
            pca_coords[mask, 0], pca_coords[mask, 1],
            label=f"Cluster {c}", color=palette(c),
            s=65, alpha=0.85, edgecolors="white", linewidths=0.5,
        )
        # Label a few countries
        for idx in np.where(mask)[0][:3]:
            ax.annotate(
                df_agg.index[idx], (pca_coords[idx, 0], pca_coords[idx, 1]),
                fontsize=6, alpha=0.7,
            )

    ax.set_title(f"K-Means Clustering — PCA Projection (K={k})", fontsize=13)
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} variance)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Tab 3: Cluster Profiles ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Cluster Profiles — Mean Values per Key Indicator")

    profile_cols = [
        c for c in [
            "GDP", "LifeExpectancyFemale", "LifeExpectancyMale",
            "InfantMortalityRate", "InternetUsage", "BirthRate",
            "CO2Emissions", "MobilePhoneUsage", "PopulationUrban",
        ] if c in df_result.columns
    ]

    profile = df_result.groupby("Cluster")[profile_cols].mean().round(3)
    st.dataframe(profile.style.background_gradient(cmap="Blues", axis=0), use_container_width=True)

    # Radar-style bar chart
    st.markdown("#### Normalised Feature Comparison Across Clusters")
    norm_profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(12, 4))
    norm_profile.T.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white", width=0.7)
    ax.set_title("Normalised Cluster Profiles")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Normalised Mean (0–1)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.legend(title="Cluster")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Country lists
    if show_countries:
        st.markdown("#### Countries per Cluster")
        cols = st.columns(k)
        for c in range(k):
            countries = sorted(df_result[df_result["Cluster"] == c].index.tolist())
            with cols[c]:
                st.markdown(f"**Cluster {c}** ({len(countries)} countries)")
                st.write(", ".join(countries))

# ── Tab 4: Explore Raw Data ─────────────────────────────────────────────────────
with tab4:
    st.subheader("Aggregated Country-Level Data with Cluster Labels")
    cluster_filter = st.multiselect(
        "Filter by cluster", options=list(range(k)), default=list(range(k))
    )
    filtered = df_result[df_result["Cluster"].isin(cluster_filter)]
    st.dataframe(filtered.round(3), use_container_width=True)

    csv = filtered.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download filtered data as CSV", data=csv,
        file_name="world_dev_clusters.csv", mime="text/csv",
    )
