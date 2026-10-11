KODE LULC JAWA TIMUR - VERSI DIRAPIKAN

Isi:
- klasifikasi_lahan_jatim_diperbaiki.ipynb: notebook analisis yang sudah dibersihkan.
- app.py: dashboard Streamlit.
- requirements.txt: daftar dependensi Python.

PENTING:
Salin notebook dan app.py ke folder proyek utama yang memiliki folder data/ dan output/.
GeoJSON data/jawatimur.geojson berisi polygon sampel, bukan batas administrasi/daratan provinsi.
Karena itu, jangan klaim statistik hanya daratan atau bahwa laut sudah terpotong sampai tersedia mask/batas daratan yang valid.

Jalankan dashboard dari terminal VS Code di folder proyek:
python -m pip install -r requirements.txt
python -m streamlit run app.py
