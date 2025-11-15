# Deploy to Streamlit Community Cloud (Share.streamlit.io)

Panduan singkat untuk mendeploy aplikasi `app/Home.py` ke Streamlit Community Cloud.

Prasyarat:
- Repo sudah berada di GitHub dan bisa diakses oleh akun Streamlit Anda (public memudahkan).
- Pastikan `app/requirements.txt` mencantumkan semua dependency (sudah ditambahkan matplotlib & seaborn).

Langkah:

1. Push semua perubahan ke GitHub (branch `main` atau branch yang Anda pilih):

```bash
git add .
git commit -m "Prepare for Streamlit deploy: requirements + docs"
git push origin main
```

2. Buka https://share.streamlit.io dan login dengan akun GitHub Anda.

3. Klik "New app" → pilih repository: `sulaemandeni97/MPCIM_Thesis` → branch: `main`.

4. Set "Main file path" ke `app/Home.py` (pastikan path persis ini).

5. Click "Deploy". Streamlit akan membangun environment menggunakan `app/requirements.txt`.

6. Setelah build selesai, Anda akan mendapatkan URL publik (mis. `https://<your-app>.streamlit.app`).

Troubleshooting:
- Jika build gagal dengan error dependency: buka logs di halaman app (tab "Logs") dan tambahkan paket yang hilang ke `app/requirements.txt` lalu commit & push.
- Jika app mencoba memuat data yang tidak ada di repo: pastikan `data/final/integrated_performance_behavioral.csv` ada di repo atau ubah kode untuk memuat data dari URL eksternal (S3, Google Drive) dan set secrets jika perlu.
- Untuk rahasia (API keys dsb), gunakan menu Settings → Secrets di Streamlit Cloud (jangan commit credentials ke repo).

Tips:
- Jika ukuran repo besar (dataset/model besar), pertimbangkan menyimpan artefak di cloud storage dan hanya memuatnya saat runtime.
- Perbarui `INSTALL.md` di repo agar kontributor tahu cara deploy.

Jika Anda mau, saya bisa membantu memverifikasi build log setelah Anda klik "Deploy" dan memperbaiki requirement yang diperlukan.
