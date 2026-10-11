from pathlib import Path
from io import BytesIO
import base64

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image
from folium.raster_layers import ImageOverlay
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium

# =========================================================
# KONFIGURASI APLIKASI
# =========================================================
st.set_page_config(
    page_title="LULC Jawa Timur | Random Forest",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
RASTER_PATH = OUTPUT_DIR / "LULC_Jawa_Timur_RandomForest.tif"
GEOJSON_PATH = DATA_DIR / "jawatimur.geojson"

true_color_candidates = []
if DATA_DIR.exists():
    for pattern in ("*True_color*.tif", "*True_color*.tiff", "*True*color*.tif", "*True*color*.tiff"):
        true_color_candidates.extend(DATA_DIR.glob(pattern))
TRUE_COLOR_PATH = true_color_candidates[0] if true_color_candidates else None

# Harus cocok dengan kode kelas pada raster keluaran notebook.
CLASS_INFO = {
    1: {
        "name": "Build up",
        "label_id": "Kawasan terbangun",
        "color": "#808080",
        "description": "Area yang didominasi bangunan, permukiman, jalan, atau permukaan buatan.",
    },
    2: {
        "name": "Farm land",
        "label_id": "Lahan pertanian",
        "color": "#FFD700",
        "description": "Area yang diberi label lahan pertanian pada polygon sampel; tidak otomatis berarti seluruh piksel adalah sawah.",
    },
    3: {
        "name": "Forest",
        "label_id": "Hutan",
        "color": "#228B22",
        "description": "Area bervegetasi yang diberi label hutan pada data sampel.",
    },
    4: {
        "name": "Mangrove",
        "label_id": "Hutan mangrove",
        "color": "#00CED1",
        "description": "Vegetasi mangrove di kawasan pesisir sesuai polygon sampel.",
    },
    5: {
        "name": "Water body",
        "label_id": "Badan air",
        "color": "#2196F3",
        "description": "Piksel yang diprediksi sebagai badan air. Tanpa mask wilayah yang tepat, kelas ini dapat mencakup perairan laut.",
    },
}

CLASS_ALIASES = {
    "fram land": "farm land",
    "farmland": "farm land",
    "farm land": "farm land",
    "built-up": "build up",
    "built up": "build up",
    "builtup": "build up",
    "hutan mangrove": "mangrove",
    "waterbody": "water body",
    "water body": "water body",
    "none": "unknown",
    "nan": "unknown",
    "": "unknown",
}
VALID_NAMES = {v["name"].lower() for v in CLASS_INFO.values()}

# =========================================================
# GAYA TAMPILAN
# =========================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px;}
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px; margin-bottom: 1rem;
        background: linear-gradient(115deg, #0b2538 0%, #14534c 58%, #23816d 100%);
        color: white; border: 1px solid rgba(255,255,255,.12);
    }
    .hero h1 {font-size: 2.15rem; margin: 0 0 .35rem 0; color: white;}
    .hero p {font-size: 1rem; margin: 0; color: #e0f2ed;}
    .section-note {color: #aab5c2; font-size: .92rem;}
    .legend-row {
        display: flex; align-items: flex-start; gap: 12px; padding: 11px 0;
        border-bottom: 1px solid rgba(128,128,128,.2);
    }
    .swatch {width: 18px; height: 18px; border-radius: 5px; flex: 0 0 18px; margin-top: 3px;
        border: 1px solid rgba(255,255,255,.35);}
    .legend-title {font-weight: 650;}
    .legend-desc {font-size: .88rem; opacity: .82; margin-top: 2px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🛰️ Peta Tutupan Lahan Jawa Timur</h1>
      <p>Sentinel-2A Level-2A · Supervised Classification · Random Forest</p>
      <p style="margin-top:.6rem;font-size:.9rem">
        Dashboard eksplorasi hasil klasifikasi raster dan polygon sampel GeoJSON.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# VALIDASI BERKAS
# =========================================================
missing = []
if not RASTER_PATH.exists():
    missing.append(str(RASTER_PATH))
if not GEOJSON_PATH.exists():
    missing.append(str(GEOJSON_PATH))
if TRUE_COLOR_PATH is None:
    missing.append("Citra True Color tidak ditemukan di folder data.")

if missing:
    st.error("Ada berkas yang belum ditemukan:")
    for item in missing:
        st.write(f"- `{item}`")
    st.info("Pastikan folder `data` dan `output` berada satu tingkat dengan `app.py`.")
    st.stop()


def rgba_data_url(rgba):
    buffer = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def bounds_wgs84(src):
    if src.crs is None:
        raise ValueError(f"Raster tidak memiliki CRS: {src.name}")
    west, south, east, north = transform_bounds(
        src.crs, "EPSG:4326",
        src.bounds.left, src.bounds.bottom,
        src.bounds.right, src.bounds.top,
        densify_pts=21,
    )
    return [[south, west], [north, east]]


@st.cache_data(show_spinner="Membaca raster klasifikasi...")
def load_classification(path):
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("Raster klasifikasi tidak memiliki CRS.")
        scale = min(1.0, 1500 / max(src.width, src.height))
        width = max(1, round(src.width * scale))
        height = max(1, round(src.height * scale))
        small = src.read(
            1, out_shape=(height, width), resampling=Resampling.nearest
        )
        small_mask = src.dataset_mask(out_shape=(height, width)) > 0
        full = src.read(1)
        full_mask = src.dataset_mask() > 0
        bounds = bounds_wgs84(src)
        meta = {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "bounds": bounds,
            "pixel_size": src.res,
            "dtype": str(src.dtypes[0]),
        }

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for code, info in CLASS_INFO.items():
        color = info["color"].lstrip("#")
        rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
        selected = (small == code) & small_mask
        rgba[selected, :3] = rgb
        rgba[selected, 3] = 220
    return rgba_data_url(rgba), bounds, full, full_mask, meta


@st.cache_data(show_spinner="Membaca citra Sentinel-2A...")
def load_true_color(path):
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("Citra True Color tidak memiliki CRS.")
        scale = min(1.0, 1500 / max(src.width, src.height))
        width = max(1, round(src.width * scale))
        height = max(1, round(src.height * scale))
        rgb = src.read(
            [1, 2, 3],
            out_shape=(3, height, width),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        mask = src.dataset_mask(out_shape=(height, width)) > 0
        bounds = bounds_wgs84(src)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for band_idx in range(3):
        band = rgb[band_idx]
        values = band[np.isfinite(band) & mask]
        if values.size:
            low, high = np.percentile(values, [2, 98])
            if high > low:
                rgba[:, :, band_idx] = (
                    np.clip((band - low) / (high - low), 0, 1) * 255
                ).astype(np.uint8)
    rgba[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)
    return rgba_data_url(rgba), bounds


@st.cache_data(show_spinner="Membaca polygon sampel...")
def load_samples(path):
    data = gpd.read_file(path)
    if data.crs is None:
        raise ValueError("GeoJSON tidak memiliki CRS.")
    data = data.to_crs("EPSG:4326")
    source_col = next(
        (c for c in ["class", "nama", "kelas", "class_name", "name"] if c in data.columns),
        None,
    )
    if source_col is None:
        raise ValueError("Kolom label kelas tidak ditemukan di GeoJSON.")
    data["class_original"] = data[source_col].astype(str).str.strip()
    data["class"] = (
        data[source_col].astype(str).str.strip().str.lower().replace(CLASS_ALIASES)
    )
    return data


try:
    class_image, class_bounds, raster_data, raster_valid, raster_meta = load_classification(str(RASTER_PATH))
    true_image, true_bounds = load_true_color(str(TRUE_COLOR_PATH))
    samples_all = load_samples(str(GEOJSON_PATH))
except Exception as exc:
    st.error(f"Gagal membaca data: {exc}")
    st.stop()

samples = samples_all[samples_all["class"].isin(VALID_NAMES)].copy()
excluded_samples = samples_all[~samples_all.index.isin(samples.index)].copy()

valid_values = raster_data[raster_valid]
total_valid = int(raster_valid.sum())
total_pixels = int(raster_data.size)
class_counts = {code: int(np.count_nonzero(valid_values == code)) for code in CLASS_INFO}
unknown_values = sorted(int(v) for v in np.unique(valid_values) if v not in CLASS_INFO)

# =========================================================
# RINGKASAN
# =========================================================
st.subheader("Ringkasan data")
st.caption("Angka di bawah berasal dari berkas yang dibaca aplikasi, bukan angka contoh.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Polygon sampel valid", f"{len(samples):,}")
c2.metric("Kelas sampel valid", f"{samples['class'].nunique()}")
c3.metric("Piksel valid raster", f"{total_valid:,}")
c4.metric("Piksel NoData", f"{total_pixels - total_valid:,}")

if len(samples) and samples["class"].nunique() != 5:
    st.warning(
        f"GeoJSON yang valid saat ini hanya berisi {samples['class'].nunique()} dari 5 kelas. "
        "Periksa kembali label kelas sampel."
    )
if unknown_values:
    st.warning(f"Raster berisi kode kelas yang belum dipetakan: {unknown_values}. Periksa pemetaan kode kelas pada notebook.")
if len(excluded_samples):
    with st.expander(f"Label sampel yang dikeluarkan dari hitungan kelas ({len(excluded_samples)} fitur)"):
        st.write("Label-label ini bukan salah satu dari lima kelas model sehingga tidak dihitung sebagai kelas valid.")
        st.dataframe(
            excluded_samples["class_original"].value_counts().rename_axis("Label asli").reset_index(name="Jumlah"),
            hide_index=True,
            use_container_width=True,
        )

# =========================================================
# STATISTIK KLASIFIKASI
# =========================================================
st.subheader("Komposisi hasil klasifikasi")
st.markdown(
    '<p class="section-note">Persentase dihitung dari seluruh piksel raster yang valid. Ini adalah distribusi prediksi model, bukan otomatis persentase resmi luas Provinsi Jawa Timur.</p>',
    unsafe_allow_html=True,
)

stats_rows = []
for code, info in CLASS_INFO.items():
    count = class_counts[code]
    stats_rows.append({
        "Kode": code,
        "Kelas": info["name"],
        "Jumlah piksel": count,
        "Persentase (%)": round(count / total_valid * 100, 2) if total_valid else 0,
        "Warna": info["color"],
    })
stats_df = pd.DataFrame(stats_rows)
col_table, col_bar = st.columns([1.2, 1])
with col_table:
    st.dataframe(
        stats_df[["Kode", "Kelas", "Jumlah piksel", "Persentase (%)"]],
        hide_index=True,
        use_container_width=True,
    )
with col_bar:
    chart_df = stats_df.set_index("Kelas")[["Persentase (%)"]]
    st.bar_chart(chart_df, horizontal=True, height=260)

with st.expander("Apa arti angka-angka statistik ini?"):
    st.markdown(
        """
        - **Jumlah piksel**: banyaknya piksel raster yang memiliki kode kelas tersebut.
        - **Persentase**: jumlah piksel suatu kelas dibagi jumlah seluruh piksel valid, lalu dikali 100%.
        - **Piksel valid**: piksel yang tidak ditandai sebagai NoData oleh raster.
        - **NoData**: piksel yang tidak memiliki data valid untuk ditampilkan/dihitung.
        - **Polygon sampel**: bentuk geometri yang didigitasi untuk memberi label kelas kepada data pelatihan. Ini bukan batas administrasi provinsi.
        """
    )

# =========================================================
# LEGENDA
# =========================================================
st.subheader("Legenda dan definisi kelas")
legend_cols = st.columns(5)
for col, (code, info) in zip(legend_cols, CLASS_INFO.items()):
    with col:
        st.markdown(
            f"""
            <div style="border:1px solid rgba(128,128,128,.28);border-radius:12px;padding:14px;height:185px">
              <div style="width:26px;height:26px;background:{info['color']};border-radius:7px;margin-bottom:10px;border:1px solid rgba(128,128,128,.35)"></div>
              <div style="font-weight:700;font-size:1rem">{info['name']}</div>
              <div style="font-size:.86rem;opacity:.8;margin:.15rem 0 .55rem">{info['label_id']}</div>
              <div style="font-size:.82rem;line-height:1.35">{info['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# PETA INTERAKTIF
# =========================================================
st.subheader("Peta interaktif")
st.caption("Aktifkan/nonaktifkan layer melalui kontrol peta. Filter polygon hanya memengaruhi sampel, bukan hasil prediksi raster.")

with st.sidebar:
    st.header("Kontrol peta")
    show_true_color = st.checkbox("Citra Sentinel-2A True Color", value=True)
    show_classification = st.checkbox("Prediksi Random Forest", value=True)
    show_samples = st.checkbox("Polygon sampel GeoJSON", value=True)
    selected_names = st.multiselect(
        "Kelas polygon sampel",
        [info["name"] for info in CLASS_INFO.values()],
        default=[info["name"] for info in CLASS_INFO.values()],
    )
    opacity = st.slider("Transparansi hasil klasifikasi", 0.2, 1.0, 0.82, 0.05)
    st.divider()
    st.caption("Sumber: raster klasifikasi, citra True Color, dan GeoJSON yang tersimpan pada folder proyek.")

south, west = class_bounds[0]
north, east = class_bounds[1]
center = [(south + north) / 2, (west + east) / 2]
map_view = folium.Map(location=center, zoom_start=7, tiles="OpenStreetMap", control_scale=True)

if show_true_color:
    ImageOverlay(
        true_image, bounds=true_bounds, name="Sentinel-2A True Color",
        opacity=1, interactive=False, cross_origin=False, zindex=1,
    ).add_to(map_view)

if show_classification:
    ImageOverlay(
        class_image, bounds=class_bounds, name="Prediksi Random Forest",
        opacity=opacity, interactive=False, cross_origin=False, zindex=2,
    ).add_to(map_view)

if show_samples:
    for code, info in CLASS_INFO.items():
        if info["name"] not in selected_names:
            continue
        subset = samples[samples["class"] == info["name"].lower()]
        if subset.empty:
            continue
        feature_group = folium.FeatureGroup(name=f"Sampel: {info['name']}", show=True)
        folium.GeoJson(
            subset.to_json(),
            name=info["name"],
            style_function=lambda feature, color=info["color"]: {
                "color": color, "weight": 2, "fillColor": color, "fillOpacity": 0.24,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["class_original"],
                aliases=["Label sampel:"],
            ),
        ).add_to(feature_group)
        feature_group.add_to(map_view)

folium.LayerControl(collapsed=False).add_to(map_view)
map_view.fit_bounds([[south, west], [north, east]])
st_folium(map_view, height=620, use_container_width=True, key="lulc_map_v2")

# =========================================================
# PENJELASAN METODE DAN BATASAN
# =========================================================
st.subheader("Cara membaca hasil dan metode")
with st.expander("Metode yang digunakan", expanded=True):
    st.markdown(
        """
        **Supervised Classification dengan algoritma Random Forest.**

        1. Polygon GeoJSON memberikan label kelas untuk sampel.
        2. Nilai spektral dari band Sentinel-2A menjadi fitur masukan model.
        3. Random Forest mempelajari hubungan antara fitur spektral dan label sampel.
        4. Model yang sudah dilatih memprediksi kelas untuk piksel raster.
        5. Raster prediksi disajikan sebagai peta berwarna dan diringkas sebagai jumlah piksel serta persentase kelas.

        Dashboard ini menampilkan hasil raster yang sudah ada; dashboard tidak melatih ulang model dan tidak menghitung akurasi model dari data uji.
        """
    )

with st.expander("Berkas dan metadata yang sedang dibaca"):
    st.write(f"**Raster klasifikasi:** `{RASTER_PATH.name}`")
    st.write(f"**Citra True Color:** `{TRUE_COLOR_PATH.name}`")
    st.write(f"**Sampel GeoJSON:** `{GEOJSON_PATH.name}`")
    st.write(f"**CRS raster:** `{raster_meta['crs']}`")
    st.write(f"**Dimensi raster:** {raster_meta['width']:,} × {raster_meta['height']:,} piksel")
    st.write(f"**Tipe data raster:** `{raster_meta['dtype']}`")
    st.write(f"**Nilai NoData metadata:** `{raster_meta['nodata']}`")
    st.write(f"**Resolusi/grid raster:** `{raster_meta['pixel_size']}`")

st.warning(
    "Batasan interpretasi: GeoJSON yang dipakai merupakan polygon sampel kelas, bukan batas administrasi Provinsi Jawa Timur. "
    "Sebelum menghitung luas resmi per provinsi atau mengklaim peta hanya mencakup daratan, gunakan batas administrasi/mask "
    "yang benar dan verifikasi kesesuaian spasialnya. Kualitas prediksi juga harus dinilai menggunakan metrik pengujian "
    "dari notebook, bukan hanya dari tampilan peta."
)

st.caption("Dashboard eksplorasi LULC · Sentinel-2A · Random Forest")
