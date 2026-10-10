# Portofolio Teknik Informatika

Template Jupyter Book untuk mengumpulkan dokumentasi tugas, praktikum, notebook, dan proyek Teknik Informatika.

## Isi
- `intro.md`: halaman pembuka
- `profil.md`: profil mahasiswa
- `proyek/klasifikasi-lahan.md`: dokumentasi proyek klasifikasi tutupan lahan
- `proyek/analisis-data.ipynb`: tempat untuk notebook analisis
- `images/profil.jpg`: ganti dengan foto profil milik sendiri

## Menjalankan
Pasang dependensi:

```bash
python -m pip install -r requirements.txt
```

Bangun website:

```bash
jupyter-book build .
```

Hasil HTML biasanya berada di `_build/html/`.

## Catatan
Sebelum membangun, letakkan foto bernama `profil.jpg` di folder `images/`, dan tambahkan notebook Anda dengan nama `proyek/analisis-data.ipynb`.
