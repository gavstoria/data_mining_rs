"""
Dashboard Analitik & Data Mining Rumah Sakit
=============================================
Dashboard interaktif berbasis Streamlit yang menerapkan 3 teknik data mining:
1. Klasifikasi  -> memprediksi hasil tes pasien (Test Results)
2. Regresi      -> memprediksi estimasi biaya perawatan (Billing Amount)
3. Clustering   -> segmentasi pasien (KMeans)

Jalankan dengan:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Data Mining Rumah Sakit",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main { padding-top: 0.5rem; }
    .block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1300px; }

    /* --- Metric Cards: each column gets a unique gradient --- */
    div[data-testid="stMetric"] {
        border: none;
        border-radius: 14px;
        padding: 18px 20px;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }

    /* 1st card - teal */
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
    }
    /* 2nd card - blue */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
    }
    /* 3rd card - violet */
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
    }
    /* 4th card - amber */
    div[data-testid="stHorizontalBlock"] > div:nth-child(4) div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
    }
    /* 5th card - rose */
    div[data-testid="stHorizontalBlock"] > div:nth-child(5) div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #e11d48 0%, #f43f5e 100%);
    }

    h1, h2, h3 { color: #10344c; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge {
        display:inline-block; padding:2px 10px; border-radius:999px;
        background:#eaf2ff; color:#1c5cab; font-size:0.8rem; font-weight:600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_PATH = "healthcare_dataset.csv"


# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Memuat data pasien...")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")
    df["Billing Amount"] = df["Billing Amount"].clip(lower=0)
    df["Admission Month"] = df["Date of Admission"].dt.to_period("M").astype(str)
    return df


try:
    raw_df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"File `{DATA_PATH}` tidak ditemukan. Letakkan file dataset pada folder "
        "yang sama dengan app.py, atau unggah manual di bawah ini."
    )
    uploaded = st.file_uploader("Unggah healthcare_dataset.csv", type="csv")
    if uploaded is not None:
        raw_df = load_data(uploaded)
    else:
        st.stop()

# ----------------------------------------------------------------------------
# SIDEBAR - NAVIGASI & FILTER GLOBAL
# ----------------------------------------------------------------------------
st.sidebar.title("🏥 Menu Dashboard")
page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "📊 Overview",
        "🧪 Klasifikasi (Test Results)",
        "💰 Regresi (Estimasi Biaya)",
        "🧩 Clustering (Segmentasi Pasien)",
        "📄 Data & Metodologi",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filter Data")

conditions = sorted(raw_df["Medical Condition"].unique())
sel_conditions = st.sidebar.multiselect("Kondisi Medis", conditions, default=conditions)

genders = sorted(raw_df["Gender"].unique())
sel_genders = st.sidebar.multiselect("Gender", genders, default=genders)

admission_types = sorted(raw_df["Admission Type"].unique())
sel_admission = st.sidebar.multiselect("Tipe Admisi", admission_types, default=admission_types)

insurers = sorted(raw_df["Insurance Provider"].unique())
sel_insurers = st.sidebar.multiselect("Asuransi", insurers, default=insurers)

age_min, age_max = int(raw_df["Age"].min()), int(raw_df["Age"].max())
sel_age = st.sidebar.slider("Rentang Usia", age_min, age_max, (age_min, age_max))

st.sidebar.markdown("---")
sample_note = st.sidebar.slider(
    "Batas baris untuk pemodelan (agar cepat)", 2000, min(20000, len(raw_df)), 6000, step=1000
)
st.sidebar.caption(
    "Model dilatih ulang secara live dari data terfilter. Batasi jumlah baris "
    "agar training tetap responsif di dashboard."
)

filtered_df = raw_df[
    raw_df["Medical Condition"].isin(sel_conditions)
    & raw_df["Gender"].isin(sel_genders)
    & raw_df["Admission Type"].isin(sel_admission)
    & raw_df["Insurance Provider"].isin(sel_insurers)
    & raw_df["Age"].between(sel_age[0], sel_age[1])
].copy()

if filtered_df.empty:
    st.warning("Tidak ada data yang cocok dengan filter yang dipilih. Silakan ubah filter.")
    st.stop()

model_df = (
    filtered_df.sample(n=min(sample_note, len(filtered_df)), random_state=42)
    if len(filtered_df) > sample_note
    else filtered_df
)

FEATURE_COLS_CAT = ["Gender", "Blood Type", "Medical Condition", "Admission Type",
                     "Insurance Provider", "Medication"]
FEATURE_COLS_NUM = ["Age", "Length of Stay", "Billing Amount"]


# --- Vibrant chart palette & template ------------------------------------------
CHART_COLORS = ["#0d9488", "#2563eb", "#7c3aed", "#d97706", "#e11d48",
                "#06b6d4", "#8b5cf6", "#f59e0b", "#10b981", "#ec4899"]

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        font=dict(family="Inter, sans-serif", color="#333"),
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
        colorway=CHART_COLORS,
        title=dict(font=dict(size=16, color="#10344c")),
    )
)

# ============================================================================
# HALAMAN 1: OVERVIEW
# ============================================================================
if page == "📊 Overview":
    st.title("📊 Overview Data Pasien Rumah Sakit")
    st.caption(
        f"Menampilkan **{len(filtered_df):,}** dari total {len(raw_df):,} rekam medis "
        "sesuai filter yang dipilih di sidebar."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Pasien", f"{len(filtered_df):,}")
    c2.metric("Rata-rata Usia", f"{filtered_df['Age'].mean():.1f} th")
    c3.metric("Rata-rata Biaya", f"${filtered_df['Billing Amount'].mean():,.0f}")
    c4.metric("Rata-rata Lama Rawat", f"{filtered_df['Length of Stay'].mean():.1f} hari")
    c5.metric("Kasus Darurat", f"{(filtered_df['Is_Emergency'].mean()*100):.1f}%")

    st.markdown("---")

    colA, colB = st.columns(2)
    with colA:
        fig = px.histogram(
            filtered_df, x="Medical Condition", color="Medical Condition",
            title="Distribusi Pasien per Kondisi Medis",
            color_discrete_sequence=CHART_COLORS,
        )
        fig.update_layout(showlegend=False, height=380, **PLOTLY_TEMPLATE["layout"].to_plotly_json())
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig = px.pie(
            filtered_df, names="Admission Type", title="Proporsi Tipe Admisi", hole=0.45,
            color_discrete_sequence=CHART_COLORS,
        )
        fig.update_layout(height=380, **PLOTLY_TEMPLATE["layout"].to_plotly_json())
        st.plotly_chart(fig, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        fig = px.box(
            filtered_df, x="Medical Condition", y="Billing Amount", color="Medical Condition",
            title="Sebaran Biaya per Kondisi Medis",
            color_discrete_sequence=CHART_COLORS,
        )
        fig.update_layout(showlegend=False, height=380, **PLOTLY_TEMPLATE["layout"].to_plotly_json())
        st.plotly_chart(fig, use_container_width=True)

    with colD:
        trend = (
            filtered_df.groupby("Admission Month")["Billing Amount"]
            .agg(["mean", "count"]).reset_index().sort_values("Admission Month")
        )
        fig = px.line(trend, x="Admission Month", y="mean", markers=True,
                      title="Tren Rata-rata Biaya per Bulan Admisi",
                      color_discrete_sequence=["#2563eb"])
        fig.update_layout(height=380, xaxis_title="Bulan", yaxis_title="Rata-rata Biaya ($)",
                          **PLOTLY_TEMPLATE["layout"].to_plotly_json())
        st.plotly_chart(fig, use_container_width=True)

    colE, colF = st.columns(2)
    with colE:
        fig = px.histogram(
            filtered_df, x="Test Results", color="Test Results",
            title="Distribusi Hasil Tes Pasien",
            color_discrete_sequence=CHART_COLORS,
        )
        fig.update_layout(showlegend=False, height=350, **PLOTLY_TEMPLATE["layout"].to_plotly_json())
        st.plotly_chart(fig, use_container_width=True)
    with colF:
        fig = px.histogram(
            filtered_df, x="Insurance Provider", color="Insurance Provider",
            title="Distribusi Penyedia Asuransi",
            color_discrete_sequence=CHART_COLORS,
        )
        fig.update_layout(showlegend=False, height=350, **PLOTLY_TEMPLATE["layout"].to_plotly_json())
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Tabel Data (dapat digulir / scroll)")
    st.dataframe(filtered_df, use_container_width=True, height=420)

    st.download_button(
        "⬇️ Unduh Data Terfilter (CSV)",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="data_pasien_terfilter.csv",
        mime="text/csv",
    )

# ============================================================================
# HALAMAN 2: KLASIFIKASI
# ============================================================================
elif page == "🧪 Klasifikasi (Test Results)":
    st.title("🧪 Klasifikasi: Prediksi Hasil Tes Pasien")

    with st.expander("📘 Penjelasan Rancangan Data Mining #1 - Klasifikasi", expanded=True):
        st.markdown(
            """
**Algoritma:** Random Forest Classifier (ensemble dari banyak decision tree)

**Kolom yang digunakan:**
- Fitur input: `Age`, `Gender`, `Blood Type`, `Medical Condition`, `Admission Type`,
  `Insurance Provider`, `Medication`, `Length of Stay`, `Billing Amount`
- Target/label: `Test Results` (Normal / Abnormal / Inconclusive)

**Cara kerja:**
1. Data kategorikal diubah menjadi angka menggunakan One-Hot Encoding, sedangkan
   data numerik distandarisasi.
2. Data dibagi menjadi data latih (train) dan data uji (test).
3. Random Forest membangun banyak pohon keputusan secara acak dari sub-sampel data
   dan fitur, lalu menggabungkan (voting) hasil prediksi setiap pohon untuk
   menentukan kelas akhir (Normal/Abnormal/Inconclusive).
4. Model dievaluasi menggunakan akurasi, F1-score, dan confusion matrix.

**Fungsi dalam pengambilan keputusan:**
Model ini membantu tim medis melakukan **triase awal** dan alokasi sumber daya —
mengidentifikasi profil pasien yang berisiko mendapatkan hasil tes *Abnormal*
sehingga rumah sakit dapat memprioritaskan pemeriksaan lanjutan, mempersiapkan
dokter spesialis, atau menjadwalkan tindak lanjut lebih cepat sesuai kebutuhan.
            """
        )

    @st.cache_resource(show_spinner="Melatih model klasifikasi...")
    def train_classifier(data: pd.DataFrame):
        X = data[FEATURE_COLS_CAT + FEATURE_COLS_NUM]
        y = data["Test Results"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLS_CAT),
                ("num", StandardScaler(), FEATURE_COLS_NUM),
            ]
        )
        clf = Pipeline(
            [("pre", pre),
             ("model", RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1))]
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

        cat_names = clf.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(FEATURE_COLS_CAT)
        all_names = list(cat_names) + FEATURE_COLS_NUM
        importances = clf.named_steps["model"].feature_importances_
        fi = pd.DataFrame({"fitur": all_names, "importance": importances}).sort_values(
            "importance", ascending=False
        ).head(12)
        return clf, acc, f1, cm, fi

    clf, acc, f1, cm, fi = train_classifier(model_df)

    m1, m2, m3 = st.columns(3)
    m1.metric("Akurasi Model", f"{acc*100:.1f}%")
    m2.metric("F1-Score (weighted)", f"{f1:.3f}")
    m3.metric("Jumlah Data Latih", f"{len(model_df):,}")

    col1, col2 = st.columns([1.1, 1])
    with col1:
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=clf.classes_, y=clf.classes_,
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
            title="Confusion Matrix",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            fi.sort_values("importance"), x="importance", y="fitur", orientation="h",
            title="Fitur Paling Berpengaruh",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔮 Simulasi Prediksi Interaktif")
    st.caption("Masukkan profil pasien untuk memprediksi kemungkinan hasil tes.")

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        in_age = st.number_input("Usia", 0, 100, 45)
        in_gender = st.selectbox("Gender", genders)
    with p2:
        in_blood = st.selectbox("Golongan Darah", sorted(raw_df["Blood Type"].unique()))
        in_condition = st.selectbox("Kondisi Medis", conditions)
    with p3:
        in_admission = st.selectbox("Tipe Admisi", admission_types)
        in_insurance = st.selectbox("Asuransi", insurers)
    with p4:
        in_medication = st.selectbox("Obat", sorted(raw_df["Medication"].unique()))
        in_los = st.number_input("Lama Rawat (hari)", 1, 60, 5)

    in_billing = st.slider("Estimasi Biaya ($)", float(raw_df["Billing Amount"].min()),
                            float(raw_df["Billing Amount"].max()),
                            float(raw_df["Billing Amount"].mean()))

    if st.button("Prediksi Hasil Tes", type="primary"):
        sample = pd.DataFrame([{
            "Gender": in_gender, "Blood Type": in_blood, "Medical Condition": in_condition,
            "Admission Type": in_admission, "Insurance Provider": in_insurance,
            "Medication": in_medication, "Age": in_age, "Length of Stay": in_los,
            "Billing Amount": in_billing,
        }])
        pred = clf.predict(sample)[0]
        proba = clf.predict_proba(sample)[0]
        st.success(f"**Prediksi Hasil Tes: {pred}**")
        proba_df = pd.DataFrame({"Kelas": clf.classes_, "Probabilitas": proba})
        fig = px.bar(proba_df, x="Kelas", y="Probabilitas", text_auto=".2f",
                     title="Probabilitas per Kelas")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# HALAMAN 3: REGRESI
# ============================================================================
elif page == "💰 Regresi (Estimasi Biaya)":
    st.title("💰 Regresi: Estimasi Biaya Perawatan (Billing Amount)")

    with st.expander("📘 Penjelasan Rancangan Data Mining #2 - Regresi", expanded=True):
        st.markdown(
            """
**Algoritma:** Random Forest Regressor

**Kolom yang digunakan:**
- Fitur input: `Age`, `Gender`, `Medical Condition`, `Admission Type`, `Insurance Provider`,
  `Medication`, `Blood Type`, `Length of Stay`
- Target: `Billing Amount` (nilai numerik kontinu)

**Cara kerja:**
1. Fitur kategorikal di-encode (One-Hot Encoding) dan fitur numerik distandarisasi.
2. Data dibagi menjadi data latih dan data uji.
3. Random Forest Regressor membangun banyak pohon regresi pada sub-sampel data,
   kemudian nilai prediksi akhir dihasilkan dari rata-rata prediksi seluruh pohon.
4. Model dievaluasi dengan metrik R², MAE (Mean Absolute Error), dan RMSE
   (Root Mean Squared Error), lalu dibandingkan hasil aktual vs prediksi.

**Fungsi dalam pengambilan keputusan:**
Estimasi biaya membantu manajemen rumah sakit dan pihak asuransi dalam
**perencanaan anggaran, penetapan tarif, serta deteksi dini potensi tagihan yang
tidak wajar** (jika nilai aktual jauh berbeda dari prediksi). Ini mendukung transparansi
biaya bagi pasien dan efisiensi keuangan bagi rumah sakit.
            """
        )

    @st.cache_resource(show_spinner="Melatih model regresi...")
    def train_regressor(data: pd.DataFrame):
        reg_cat = ["Gender", "Blood Type", "Medical Condition", "Admission Type",
                   "Insurance Provider", "Medication"]
        reg_num = ["Age", "Length of Stay"]
        X = data[reg_cat + reg_num]
        y = data["Billing Amount"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), reg_cat),
                ("num", StandardScaler(), reg_num),
            ]
        )
        reg = Pipeline(
            [("pre", pre),
             ("model", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))]
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        cat_names = reg.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(reg_cat)
        all_names = list(cat_names) + reg_num
        importances = reg.named_steps["model"].feature_importances_
        fi = pd.DataFrame({"fitur": all_names, "importance": importances}).sort_values(
            "importance", ascending=False
        ).head(12)
        return reg, r2, mae, rmse, y_test.reset_index(drop=True), pd.Series(y_pred), fi, reg_cat, reg_num

    reg, r2, mae, rmse, y_test, y_pred, fi, reg_cat, reg_num = train_regressor(model_df)

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{r2:.3f}")
    m2.metric("MAE", f"${mae:,.0f}")
    m3.metric("RMSE", f"${rmse:,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        comp = pd.DataFrame({"Aktual": y_test, "Prediksi": y_pred}).sample(
            n=min(1000, len(y_test)), random_state=1
        )
        fig = px.scatter(
            comp, x="Aktual", y="Prediksi", opacity=0.5,
            title="Aktual vs Prediksi Biaya",
        )
        min_v, max_v = comp[["Aktual", "Prediksi"]].min().min(), comp[["Aktual", "Prediksi"]].max().max()
        fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                      line=dict(color="red", dash="dash"))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(fi.sort_values("importance"), x="importance", y="fitur", orientation="h",
                     title="Fitur Paling Berpengaruh terhadap Biaya")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔮 Simulasi Estimasi Biaya")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        r_age = st.number_input("Usia ", 0, 100, 45, key="r_age")
        r_gender = st.selectbox("Gender ", genders, key="r_gender")
    with p2:
        r_blood = st.selectbox("Golongan Darah ", sorted(raw_df["Blood Type"].unique()), key="r_blood")
        r_condition = st.selectbox("Kondisi Medis ", conditions, key="r_condition")
    with p3:
        r_admission = st.selectbox("Tipe Admisi ", admission_types, key="r_admission")
        r_insurance = st.selectbox("Asuransi ", insurers, key="r_insurance")
    with p4:
        r_medication = st.selectbox("Obat ", sorted(raw_df["Medication"].unique()), key="r_medication")
        r_los = st.number_input("Lama Rawat (hari) ", 1, 60, 5, key="r_los")

    if st.button("Estimasi Biaya", type="primary"):
        sample = pd.DataFrame([{
            "Gender": r_gender, "Blood Type": r_blood, "Medical Condition": r_condition,
            "Admission Type": r_admission, "Insurance Provider": r_insurance,
            "Medication": r_medication, "Age": r_age, "Length of Stay": r_los,
        }])
        pred_bill = reg.predict(sample)[0]
        st.success(f"**Estimasi Biaya Perawatan: ${pred_bill:,.2f}**")

# ============================================================================
# HALAMAN 4: CLUSTERING
# ============================================================================
elif page == "🧩 Clustering (Segmentasi Pasien)":
    st.title("🧩 Clustering: Segmentasi Pasien")

    with st.expander("📘 Penjelasan Rancangan Data Mining #3 - Clustering", expanded=True):
        st.markdown(
            """
**Algoritma:** K-Means Clustering

**Kolom yang digunakan:**
`Age`, `Billing Amount`, `Length of Stay` (dinormalisasi dengan StandardScaler)

**Cara kerja:**
1. Fitur numerik distandarisasi agar memiliki skala yang setara.
2. K-Means menentukan sejumlah *k* titik pusat (centroid) secara acak, lalu setiap
   pasien dikelompokkan ke centroid terdekat berdasarkan jarak Euclidean.
3. Centroid diperbarui berulang kali (iteratif) hingga posisi centroid stabil/konvergen.
4. Jumlah cluster optimal dipilih menggunakan **Elbow Method** dan divalidasi dengan
   **Silhouette Score**, kemudian hasil divisualisasikan dalam ruang 2 dimensi (PCA).

**Fungsi dalam pengambilan keputusan:**
Segmentasi pasien membantu manajemen rumah sakit merancang **strategi layanan yang
lebih tepat sasaran** — misalnya kelompok pasien usia lanjut dengan biaya tinggi dan
lama rawat panjang dapat diarahkan ke program perawatan kronis khusus, sementara
kelompok biaya rendah dan rawat singkat dapat dioptimalkan pada jalur rawat jalan,
sehingga alokasi sumber daya (kamar, staf, anggaran) menjadi lebih efisien.
            """
        )

    cluster_features = ["Age", "Billing Amount", "Length of Stay"]

    k = st.slider("Jumlah Cluster (k)", 2, 10, 4)

    @st.cache_resource(show_spinner="Menjalankan K-Means...")
    def run_kmeans(data: pd.DataFrame, k: int):
        X = data[cluster_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        inertias = []
        k_range = range(2, 11)
        for kk in k_range:
            km_tmp = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(X_scaled)
            inertias.append(km_tmp.inertia_)

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)

        result = data.copy()
        result["Cluster_Baru"] = labels.astype(str)
        result["PCA1"] = coords[:, 0]
        result["PCA2"] = coords[:, 1]
        return result, list(k_range), inertias, sil

    clustered_df, k_range, inertias, sil = run_kmeans(model_df, k)

    m1, m2 = st.columns(2)
    m1.metric("Silhouette Score", f"{sil:.3f}", help="Semakin mendekati 1, cluster semakin baik terpisah.")
    m2.metric("Jumlah Data", f"{len(clustered_df):,}")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers"))
        fig.update_layout(title="Elbow Method (Menentukan k optimal)",
                           xaxis_title="Jumlah Cluster (k)", yaxis_title="Inertia", height=420)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(
            clustered_df, x="PCA1", y="PCA2", color="Cluster_Baru",
            hover_data=["Age", "Billing Amount", "Length of Stay", "Medical Condition"],
            title="Visualisasi Cluster (proyeksi PCA 2D)",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Profil Tiap Cluster")
    profile = clustered_df.groupby("Cluster_Baru").agg(
        Jumlah_Pasien=("Age", "count"),
        Rata2_Usia=("Age", "mean"),
        Rata2_Biaya=("Billing Amount", "mean"),
        Rata2_LamaRawat=("Length of Stay", "mean"),
    ).round(1).reset_index().sort_values("Cluster_Baru")
    st.dataframe(profile, use_container_width=True)

    fig = px.bar(profile, x="Cluster_Baru", y="Rata2_Biaya", color="Cluster_Baru",
                 title="Rata-rata Biaya per Cluster", text_auto=".0f")
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Data dengan Label Cluster (scroll untuk melihat lebih banyak)")
    st.dataframe(
        clustered_df[["Name", "Age", "Gender", "Medical Condition", "Billing Amount",
                      "Length of Stay", "Admission Type", "Cluster_Baru"]],
        use_container_width=True, height=400,
    )

# ============================================================================
# HALAMAN 5: DATA & METODOLOGI
# ============================================================================
else:
    st.title("📄 Data & Metodologi Data Mining")
    st.markdown(
        """
### Ringkasan Dataset
Dataset berisi rekam medis pasien rumah sakit, mencakup data demografis
(usia, gender, golongan darah), data administratif (tanggal admisi, tipe
admisi, rumah sakit, dokter, asuransi), data klinis (kondisi medis, obat,
hasil tes), serta data finansial (biaya perawatan/`Billing Amount`).

### Tujuan Penerapan Data Mining
Data mining diterapkan untuk mengubah data historis pasien menjadi **wawasan yang
dapat ditindaklanjuti (actionable insight)**, sehingga mendukung pengambilan
keputusan manajemen rumah sakit dalam tiga aspek utama: **pelayanan medis**,
**perencanaan keuangan**, dan **efisiensi operasional**.
        """
    )

    st.markdown("### Ringkasan 3 Rancangan Data Mining")
    tab1, tab2, tab3 = st.tabs(["1️⃣ Klasifikasi", "2️⃣ Regresi", "3️⃣ Clustering"])

    with tab1:
        st.markdown(
            """
| Aspek | Penjelasan |
|---|---|
| **Algoritma** | Random Forest Classifier |
| **Kolom Digunakan** | Age, Gender, Blood Type, Medical Condition, Admission Type, Insurance Provider, Medication, Length of Stay, Billing Amount → Target: Test Results |
| **Cara Kerja** | Membentuk banyak decision tree dari sub-sampel data secara acak, hasil akhir ditentukan lewat voting mayoritas antar pohon |
| **Fungsi Keputusan** | Mendukung triase klinis: memprioritaskan pasien berisiko hasil tes *Abnormal* untuk pemeriksaan lanjutan |
            """
        )
    with tab2:
        st.markdown(
            """
| Aspek | Penjelasan |
|---|---|
| **Algoritma** | Random Forest Regressor |
| **Kolom Digunakan** | Age, Gender, Medical Condition, Admission Type, Insurance Provider, Medication, Blood Type, Length of Stay → Target: Billing Amount |
| **Cara Kerja** | Banyak pohon regresi dilatih pada sub-sampel data, prediksi akhir adalah rata-rata dari seluruh pohon |
| **Fungsi Keputusan** | Mendukung perencanaan anggaran rumah sakit dan deteksi anomali tagihan (biaya jauh di luar estimasi) |
            """
        )
    with tab3:
        st.markdown(
            """
| Aspek | Penjelasan |
|---|---|
| **Algoritma** | K-Means Clustering |
| **Kolom Digunakan** | Age, Billing Amount, Length of Stay |
| **Cara Kerja** | Mengelompokkan pasien ke centroid terdekat secara iteratif hingga konvergen; k optimal dipilih via Elbow Method & Silhouette Score |
| **Fungsi Keputusan** | Mendukung segmentasi pasien untuk strategi layanan dan alokasi sumber daya (kamar, staf, program perawatan) yang lebih tepat sasaran |
            """
        )

    st.markdown("### Contoh Data Mentah")
    st.dataframe(raw_df.head(50), use_container_width=True, height=400)

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard Data Mining Rumah Sakit • Dibangun dengan Streamlit")