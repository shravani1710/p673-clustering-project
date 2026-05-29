import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import io
import warnings
warnings.filterwarnings("ignore")

st.title("🌍 World Development Clustering")

# Sidebar
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])
k = st.sidebar.slider("Number of Clusters (K)", 2, 8, 3)
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# Preprocessing
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
            df["BusinessTaxRate"].astype(str).str.replace("%", "", regex=False).str.strip()
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

@st.cache_data(show_spinner="Running K-Means…")
def run_kmeans(scaled, n_clusters, seed):
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(scaled)
    sil = silhouette_score(scaled, labels)
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(scaled)
    return labels, sil, coords, pca.explained_variance_ratio_

# Gate
if uploaded_file is None:
    st.info("👈 Upload the Excel dataset from the sidebar to begin.")
    st.stop()

# Run everything
df_agg = load_and_preprocess(uploaded_file.read())
scaled_data = StandardScaler().fit_transform(df_agg)
labels, sil_score, pca_coords, var_ratio = run_kmeans(scaled_data, k, int(random_seed))
df_result = df_agg.copy()
df_result["Cluster"] = labels

colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]

# ── Results ───────────────────────────────────────────────────────────────────
st.success(f"✅ {df_agg.shape[0]} countries clustered into {k} groups.")

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Countries", df_agg.shape[0])
col2.metric("Silhouette Score", f"{sil_score:.3f}")
col3.metric("Variance Explained", f"{sum(var_ratio):.1%}")

st.divider()

# PCA Scatter Plot
st.subheader("Cluster Map (PCA)")
fig, ax = plt.subplots(figsize=(9, 5))
for c in range(k):
    mask = labels == c
    ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
               label=f"Cluster {c}", color=colors[c % len(colors)],
               s=60, alpha=0.8, edgecolors="white", linewidths=0.5)
    for idx in np.where(mask)[0][:4]:
        ax.annotate(df_agg.index[idx], (pca_coords[idx, 0] + 0.05, pca_coords[idx, 1]),
                    fontsize=7, color="#444")
ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} variance)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# Cluster Summary Table
st.subheader("Cluster Averages")
profile_cols = [c for c in [
    "GDP", "LifeExpectancyFemale", "LifeExpectancyMale",
    "InfantMortalityRate", "InternetUsage", "BirthRate",
    "CO2Emissions", "MobilePhoneUsage", "PopulationUrban",
] if c in df_result.columns]
profile = df_result.groupby("Cluster")[profile_cols].mean().round(2)
st.dataframe(profile.style.background_gradient(cmap="YlGnBu", axis=0), use_container_width=True)

st.divider()

# Countries per Cluster
st.subheader("Countries per Cluster")
cols = st.columns(k)
for c in range(k):
    countries = sorted(df_result[df_result["Cluster"] == c].index.tolist())
    with cols[c]:
        st.write(f"**Cluster {c}** ({len(countries)} countries)")
        st.caption(", ".join(countries))

st.divider()

# Download
st.download_button(
    "⬇️ Download Results as CSV",
    data=df_result.round(3).to_csv().encode("utf-8"),
    file_name="world_dev_clusters.csv",
    mime="text/csv",
)
