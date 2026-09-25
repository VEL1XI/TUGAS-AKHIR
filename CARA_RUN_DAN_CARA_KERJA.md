# Panduan Lengkap: Cara Menjalankan & Cara Kerja Sistem
## Nutrisi Nusantara AI — Sistem Klasifikasi & Estimasi Nutrisi 100 Kelas Makanan Indonesia
### Berbasis Convolutional Neural Network SOTA & Tabel Komposisi Pangan Indonesia (TKPI 2020)

Dokumen ini berisi dokumentasi teknis dan panduan operasional menyeluruh untuk mereplikasi, melatih, mengevaluasi, dan menjalankan sistem **Nutrisi Nusantara AI** baik pada lingkungan *Machine Learning Pipeline* (Python) maupun aplikasi *Mobile Client* (Flutter).

---

## Daftar Isi
1. [Gambaran Umum Sistem & Arsitektur](#1-gambaran-umum-sistem--arsitektur)
2. [Alur & Cara Kerja Sistem (End-to-End Workflow)](#2-alur--cara-kerja-sistem-end-to-end-workflow)
3. [Struktur Direktori Proyek](#3-struktur-direktori-proyek)
4. [Prasyarat & Persiapan Lingkungan (Environment Setup)](#4-prasyarat--persiapan-lingkungan-environment-setup)
5. [Panduan Langkah-demi-Langkah Menjalankan (Pipeline Python)](#5-panduan-langkah-demi-langkah-menjalankan-pipeline-python)
   - [Tahap 1: Persiapan & Pembagian Dataset](#tahap-1-persiapan--pembagian-dataset-01_dataset_preppy)
   - [Tahap 2: Preprocessing & Augmentasi Data](#tahap-2-preprocessing--augmentasi-data-02_preprocessingpy)
   - [Tahap 3: Setup & Validasi Basis Data Nutrisi](#tahap-3-setup--validasi-basis-data-nutrisi-03_nutrition_setuppy)
   - [Tahap 4: Pembentukan & Profiling Arsitektur Model](#tahap-4-pembentukan--profiling-arsitektur-model-04_modelpy)
   - [Tahap 5: Pelatihan & Fine-Tuning Model](#tahap-5-pelatihan--fine-tuning-model-05_trainingpy)
   - [Tahap 6: Evaluasi Mendalam Model Klasifikasi](#tahap-6-evaluasi-mendalam-model-klasifikasi-06_evaluationpy)
   - [Tahap 7: Evaluasi Estimasi Nutrisi End-to-End](#tahap-7-evaluasi-estimasi-nutrisi-end-to-end-07_nutrition_mappingpy)
   - [Tahap 8: Konversi Model ke ONNX & Kuantisasi Mobile](#tahap-8-konversi-model-ke-onnx--kuantisasi-mobile-08_mobile_conversionpy)
   - [Tahap 9: Uji Integrasi Sistem Lengkap](#tahap-9-uji-integrasi-sistem-lengkap-09_integration_testpy)
   - [Tahap 10: Pembuatan Materi Bab IV Skripsi Otomatis](#tahap-10-pembuatan-materi-bab-iv-skripsi-otomatis-10_report_generatorpy)
6. [Panduan Menjalankan Aplikasi Mobile (Flutter)](#6-panduan-menjalankan-aplikasi-mobile-flutter)
7. [Cheatsheet Perintah Cepat (Quickrun)](#7-cheatsheet-perintah-cepat-quickrun)
8. [Troubleshooting & Solusi Kendala Umum](#8-troubleshooting--solusi-kendala-umum)
9. [Poin-Poin Ilmiah Kunci untuk Sidang Sarjana (Defense Arguments)](#9-poin-poin-ilmiah-kunci-untuk-sidang-sarjana-defense-arguments)

---

## 1. Gambaran Umum Sistem & Arsitektur

Sistem ini dirancang untuk mengatasi permasalahan estimasi gizi makanan tradisional Indonesia yang memiliki variasi bumbu, tekstur kuah, dan bentuk visual kompleks. Berbeda dari pendekatan konvensional yang kaku, sistem ini menggunakan **pola arsitektur decoupled (terpisah)**:

1. **Visual Recognition Layer (Deep Learning CNN)**:
   - Menggunakan model SOTA terbaru: **ConvNeXt V2-Nano** (Woo et al., CVPR 2023) sebagai model utama, serta **MobileNetV4-Conv-Medium** (Google Research, ECCV 2024) dan **EfficientNetV2-S** (Tan & Le, ICML 2021) sebagai model komparasi.
   - Mengidentifikasi secara akurat 100 kelas makanan Indonesia dari 8 kategori kuliner nusantara.
2. **Semantic Nutrition Mapping Layer**:
   - Menghubungkan label hasil klasifikasi dengan basis data gizi terstandarisasi **TKPI 2020 (Tabel Komposisi Pangan Indonesia / Kemenkes RI)**.
   - Memvalidasi hukum termodinamika Atwater (4 kkal/g protein, 4 kkal/g karbohidrat, 9 kkal/g lemak).
   - Memungkinkan perhitungan porsi dinamis (skala gram) secara instan tanpa perlu retraining model CNN.
3. **On-Device Edge Inference Engine**:
   - Model dikonversi ke format **ONNX Runtime Mobile (Opset 17)** dengan teknik **Post-Training Quantization (PTQ INT8)**.
   - Inferensi berjalan 100% lokal di smartphone pengguna (offline, tanpa dependensi server internet, menjaga privasi pengguna, dan latensi ultra-rendah).

---

## 2. Alur & Cara Kerja Sistem (End-to-End Workflow)

Berikut diagram alur kerja komprehensif mulai dari pemrosesan citra hingga visualisasi pada layar pengguna:

```mermaid
flowchart TD
    subgraph Data_Pipeline ["1. Data Pipeline (Python)"]
        A[Dataset Citra 100 Kelas Makanan] --> B[Stratified Split 70:15:15<br/>01_dataset_prep.py]
        B --> C[Preprocessing & Augmentasi<br/>RandAugment, MixUp, CutMix<br/>02_preprocessing.py]
        D[Tabel Gizi TKPI Kemenkes] --> E[Validasi Atwater & JSON Export<br/>03_nutrition_setup.py]
    end

    subgraph Training_Pipeline ["2. Training & Evaluation"]
        C --> F[Fine-Tuning 3-Fase SOTA<br/>ConvNeXt V2 / MobileNetV4 / EfficientNetV2<br/>04_model.py & 05_training.py]
        F --> G[Evaluasi Klasifikasi & Confusion Matrix<br/>06_evaluation.py]
        G --> H[Evaluasi Error Gizi MAE/MAPE<br/>07_nutrition_mapping.py]
    end

    subgraph Deployment_Pipeline ["3. Mobile Export & Integration"]
        F --> I[Export PyTorch ke ONNX Opset 17<br/>08_mobile_conversion.py]
        I --> J[Kuantisasi Dynamic INT8<br/>Ukuran Memori Turun ~75%]
        J --> K[Uji Integrasi End-to-End<br/>09_integration_test.py]
        K --> L[Generate Laporan Bab IV<br/>10_report_generator.py]
    end

    subgraph Mobile_App ["4. Aplikasi Mobile Flutter (Client)"]
        J -. Model .onnx .-> M[mobile_app/assets/models/]
        E -. nutrition.json .-> N[mobile_app/assets/nutrition.json]
        O[Input Kamera / Galeri] --> P[Preprocessing Piksel & Resize 224x224]
        P --> Q[ONNX Runtime Mobile Engine]
        M --> Q
        Q --> R[Top-1 & Top-K Prediksi Kelas]
        R --> S[Semantic Lookup Gizi]
        N --> S
        S --> T[Kalkulasi Porsi Gram & Nilai AKG]
        T --> U[UI Interaktif: Chart Makro, Slider Porsi, History]
    end
```

### Mekanisme Kerja Tiap Lapisan:
1. **Preprocessing Citra**: Gambar makanan dari kamera dinormalisasi pikselnya ke rentang mean `[0.485, 0.456, 0.406]` dan std `[0.229, 0.224, 0.225]` sesuai standar bobot pre-trained ImageNet, kemudian di-resize ke resolusi input model (224×224 untuk ConvNeXt V2-Nano).
2. **Ekstraksi Fitur & Klasifikasi**: Convolutional layer mengekstrak pola hierarkis visual (tekstur nasi, kilau minyak, warna kuah, potongan daging). Head linear memproyeksikan fitur menjadi 100 nilai logit, lalu dihitung nilai probabilitasnya menggunakan fungsi Softmax.
3. **Lookup & Skalasi Porsi Nutrisi**: Mengambil entri nutrisi referensi per porsi standar (misal: 1 porsi nasi goreng = 200 gram). Jika pengguna menggeser slider porsi ke nilai $W$ gram, maka setiap nilai nutrisi dihitung ulang secara linier:
   $$\text{Nutrisi}_{\text{aktual}} = \text{Nutrisi}_{\text{standar}} \times \left(\frac{W}{W_{\text{standar}}}\right)$$
4. **Analisis Angka Kecukupan Gizi (AKG)**: Nilai makronutrisi dibandingkan dengan rekomendasi harian rata-rata dewasa Indonesia (2150 kkal energi, 60g protein, 65g lemak, 340g karbohidrat, 30g serat) untuk menyajikan progress bar AKG interaktif.

---

## 3. Struktur Direktori Proyek

```
TUGAS-AKHIR/
├── config.py                     # Konfigurasi terpusat (100 kelas, hyperparameter, direktori)
├── requirements.txt              # Daftar pustaka Python yang dibutuhkan
├── README.md                     # Ringkasan proyek di GitHub
├── CARA_RUN_DAN_CARA_KERJA.md    # Panduan komprehensif ini
│
├── 01_dataset_prep.py            # Tahap 1: Pembuatan folder & pembagian dataset stratified
├── 02_preprocessing.py           # Tahap 2: DataLoader, augmentasi RandAugment/MixUp/CutMix
├── 03_nutrition_setup.py         # Tahap 3: Penyusunan basis data gizi TKPI 2020 & validasi Atwater
├── 04_model.py                   # Tahap 4: Arsitektur ConvNeXt V2, MobileNetV4, EfficientNetV2
├── 05_training.py                # Tahap 5: Training multi-fase, early stopping, FP16 AMP
├── 06_evaluation.py              # Tahap 6: Top-1/5, Macro F1, 100-class & Macro Confusion Matrix
├── 07_nutrition_mapping.py       # Tahap 7: Inferensi end-to-end, evaluasi MAE & MAPE nutrisi
├── 08_mobile_conversion.py       # Tahap 8: Konversi ONNX Opset 17, INT8 PTQ, uji paritas numerik
├── 09_integration_test.py        # Tahap 9: Stress test batch & validasi format output
├── 10_report_generator.py        # Tahap 10: Otomasi pembuatan tabel Bab IV skripsi (CSV & teks)
│
├── data/                         # Direktori dataset citra (tidak ditrack git)
│   ├── raw/                      # 100 sub-folder gambar mentah per kelas
│   ├── train/                    # 70% data training hasil split
│   ├── val/                      # 15% data validasi tuning hyperparameter
│   └── test/                     # 15% data pengujian akhir
│
├── models/                       # Penyimpanan checkpoint model (tidak ditrack git)
│   ├── saved_models/             # Bobot PyTorch (.pth)
│   └── onnx/                     # Model ONNX (.onnx) hasil ekspor & kuantisasi
│
├── nutrition/                    # Basis data gizi terstruktur
│   ├── food_nutrition.csv        # Tabel komposisi gizi 100 kelas makanan
│   ├── food_nutrition.json       # Format JSON untuk lookup cepat
│   └── food_nutrition_expert_review.csv # Template validasi ahli gizi (SPPG)
│
├── reports/                      # Laporan performa & Bab IV skripsi
│   ├── figures/                  # Gambar visualisasi kurva loss, akurasi, confusion matrix
│   ├── bab4/                     # Tabel 4.1 - 4.4 format CSV/JSON siap kutip skripsi
│   ├── bab_4_panduan_laporan.md  # Panduan ringkas penulisan Bab IV
│   └── BAB_IV_HASIL_DAN_PEMBAHASAN.md # Draf naskah lengkap Bab IV Hasil & Pembahasan
│
└── mobile_app/                   # Aplikasi Mobile Flutter
    ├── README.md                 # Dokumentasi lengkap aplikasi mobile Flutter
    ├── pubspec.yaml              # Konfigurasi dependensi Flutter
    ├── assets/
    │   ├── nutrition.json        # Basis data gizi offline di dalam bundle aplikasi
    │   └── models/               # Model ONNX mobile (convnextv2_nano.onnx, dll)

    └── lib/
        ├── main.dart             # Titik masuk aplikasi, inisialisasi tema modern
        ├── models/
        │   └── food_item.dart    # Data model Dart untuk parsing nutrisi & prediksi
        ├── services/
        │   ├── classifier_service.dart # Engine inferensi onnxruntime mobile
        │   └── nutrition_service.dart  # Layanan lookup nutrisi & kalkulasi AKG
        └── screens/
            ├── splash_screen.dart      # Splash screen dengan logo dan loading model
            ├── home_screen.dart        # Beranda, input kamera/galeri, tips nutrisi
            ├── result_screen.dart      # Hasil deteksi, slider gram, chart gizi
            └── history_screen.dart     # Riwayat pemindaian makanan pengguna
```

---

## 4. Prasyarat & Persiapan Lingkungan (Environment Setup)

### 4.1. Kebutuhan Perangkat Lunak
* **Sistem Operasi**: Windows 10/11, macOS, atau Ubuntu Linux.
* **Python**: Versi `3.10.x` atau `3.11.x` (disarankan 64-bit).
* **Flutter SDK**: Versi `>= 3.0.0` (Dart SDK `>= 3.0.0 < 4.0.0`).
* **Android Studio**: Dilengkapi Android SDK Platform-Tools & Emulator (atau smartphone fisik dengan USB Debugging aktif).
* **Git**: Untuk manajemen versi source code.

### 4.2. Persiapan Python Virtual Environment
Buka terminal (PowerShell pada Windows atau Terminal pada macOS/Linux) di folder proyek:

```bash
# 1. Buka folder root proyek
cd "d:\KAMPUS\SEM 7\TA\app TA"

# 2. Buat virtual environment bernama venv
python -m venv venv

# 3. Aktifkan virtual environment
# Pada Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Atau pada Windows Command Prompt (CMD):
.\venv\Scripts\activate.bat
# Atau pada Linux/macOS:
source venv/bin/activate

# 4. Upgrade pip ke versi terbaru
python -m pip install --upgrade pip

# 5. Instal seluruh dependensi pustaka Python
pip install -r requirements.txt
```

> **Catatan Instalasi PyTorch GPU (CUDA)**:
> Jika perangkat memiliki kartu grafis NVIDIA, disarankan menginstal PyTorch berkemampuan CUDA agar training berjalan jauh lebih cepat:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 5. Panduan Langkah-demi-Langkah Menjalankan (Pipeline Python)

Seluruh pipeline dirancang modular, bernomor urut dari `01_` hingga `10_`, sehingga dapat dijalankan berurutan secara sistematis.

---

### Tahap 1: Persiapan & Pembagian Dataset (`01_dataset_prep.py`)
Skrip ini menginisialisasi direktori data dan membagi kumpulan citra mentah menjadi data latih, validasi, dan uji menggunakan metode **Stratified Splitting**.

1. **Jalankan inisialisasi folder**:
   ```bash
   python 01_dataset_prep.py
   ```
   *Skrip akan secara otomatis membuat 100 sub-folder di dalam `data/raw/` sesuai nama 100 kelas makanan yang ada di `config.py`.*
2. **Masukkan data citra**: Salin gambar makanan yang telah dikumpulkan ke folder masing-masing, misalnya: `data/raw/nasi_goreng/`, `data/raw/rendang/`, dst.
3. **Jalankan kembali skrip untuk pembagian dataset**:
   ```bash
   python 01_dataset_prep.py
   ```
   *Skrip akan mendeteksi seluruh citra, mencatat distribusi kelas, lalu membaginya ke folder `data/train/` (70%), `data/val/` (15%), dan `data/test/` (15%) dengan proporsi kelas yang terjamin seragam.*

---

### Tahap 2: Preprocessing & Augmentasi Data (`02_preprocessing.py`)
Membangun PyTorch DataLoader dan memverifikasi teknik augmentasi data canggih (**RandAugment**, **MixUp**, **CutMix**, serta **WeightedRandomSampler** untuk menangani ketidakseimbangan kelas).

```bash
python 02_preprocessing.py
```
* **Fungsi**: Memvalidasi pipeline citra, menghitung bobot kelas, dan mengekspor visualisasi sampel augmentasi ke folder `reports/figures/` sebagai lampiran laporan skripsi.

---

### Tahap 3: Setup & Validasi Basis Data Nutrisi (`03_nutrition_setup.py`)
Mengelola data komposisi nutrisi 100 kelas makanan berdasarkan standar **TKPI 2020 / Panganku Kemenkes RI**.

```bash
python 03_nutrition_setup.py
```
* **Hasil**:
  1. Menghasilkan `nutrition/food_nutrition.csv` dan `nutrition/food_nutrition.json`.
  2. Menghasilkan template review ahli gizi: `nutrition/food_nutrition_expert_review.csv`.
  3. Memvalidasi kepatuhan nilai energi berdasarkan konversi faktor Atwater.
  4. Secara otomatis menyalin file referensi ke aset aplikasi mobile: `mobile_app/assets/nutrition.json`.

---

### Tahap 4: Pembentukan & Profiling Arsitektur Model (`04_model.py`)
Membangun 3 arsitektur CNN State-of-the-Art melalui *factory pattern* menggunakan pustaka `timm` dan melakukan audit profil model.

```bash
python 04_model.py
```
* **Hasil Profiling**:
  - Menghitung jumlah parameter, ukuran memori bobot (MB), estimasi FLOPs, dan benchmark latensi inferensi untuk:
    1. **ConvNeXt V2-Nano** (15.6M parameter, input 224×224)
    2. **MobileNetV4-Conv-Medium** (9.7M parameter, input 256×256)
    3. **EfficientNetV2-S** (21.5M parameter, input 384×384)

---

### Tahap 5: Pelatihan & Fine-Tuning Model (`05_training.py`)
Melatih model klasifikasi menggunakan strategi **3-Fase Fine-Tuning**:
- **Fase 1 (Warmup)**: Backbone di-freeze, hanya classifier head baru yang dilatih.
- **Fase 2 (Gradual Unfreeze)**: Stage tertinggi backbone dibuka secara bertahap.
- **Fase 3 (Full Fine-Tuning)**: Seluruh lapisan dibuka dengan discriminative learning rates.

```bash
# Melatih model utama (ConvNeXt V2-Nano)
python 05_training.py --model convnextv2_nano --epochs 50 --batch-size 32

# Opsi: Melatih dengan dry-run data sintetis jika dataset asli belum lengkap
python 05_training.py --dry-run
```
* **Fitur**: Dilengkapi *Label Smoothing (0.1)*, *Mixed Precision Training (torch.cuda.amp)*, *Early Stopping*, serta penyimpanan bobot terbaik di `models/saved_models/best_convnextv2_nano.pth`.

---

### Tahap 6: Evaluasi Mendalam Model Klasifikasi (`06_evaluation.py`)
Menguji performa model terlatih pada *Test Set* yang belum pernah dilihat saat pelatihan.

```bash
python 06_evaluation.py --model convnextv2_nano
```
* **Metrik yang Dihasilkan**:
  - Akurasi Top-1 dan Akurasi Top-5.
  - Macro F1-Score dan Weighted F1-Score.
  - Classification Report per 100 kelas (Precision, Recall, F1) yang diekspor ke format CSV.
  - Visualisasi **Macro Confusion Matrix (8x8 Kategori)** dan analisis pasangan makanan yang paling sering tertukar (*Top-15 Most Confused Pairs*).

---

### Tahap 7: Evaluasi Estimasi Nutrisi End-to-End (`07_nutrition_mapping.py`)
Menguji dampak kesalahan prediksi klasifikasi terhadap nilai estimasi gizi yang diterima pengguna akhir.

```bash
python 07_nutrition_mapping.py --model convnextv2_nano
```
* **Metrik Evaluasi**:
  - Menghitung **MAE (Mean Absolute Error)** dalam satuan kkal/gram.
  - Menghitung **MAPE (Mean Absolute Percentage Error)** untuk Kalori, Protein, Karbohidrat, dan Lemak.

---

### Tahap 8: Konversi Model ke ONNX & Kuantisasi Mobile (`08_mobile_conversion.py`)
Mempersiapkan model PyTorch untuk dideploy ke lingkungan mobile dengan performa tinggi.

```bash
python 08_mobile_conversion.py
```
* **Alur Eksekusi**:
  1. Mengekspor bobot PyTorch ke format universal **ONNX (Opset 17)**.
  2. Menerapkan **Dynamic INT8 Quantization** (memperkecil bobot model hingga ~75% dengan dampak akurasi minimal).
  3. Menguji paritas numerik (*Cosine Similarity > 0.999* dan konsistensi Top-1 100% terhadap PyTorch CPU).
  4. File hasil disimpan di `models/onnx/convnextv2_nano.onnx` dan `models/onnx/convnextv2_nano_int8.onnx`.

> **PENTING UNTUK APLIKASI MOBILE**:
> Salin model ONNX yang telah dihasilkan ke direktori aset Flutter:
> ```powershell
> # Buat folder tujuan jika belum ada
> New-Item -ItemType Directory -Force -Path "mobile_app/assets/models"
> 
> # Salin file ONNX
> Copy-Item "models/onnx/convnextv2_nano.onnx" "mobile_app/assets/models/"
> Copy-Item "models/onnx/convnextv2_nano_int8.onnx" "mobile_app/assets/models/"
> ```

---

### Tahap 9: Uji Integrasi Sistem Lengkap (`09_integration_test.py`)
Memastikan keselarasan antara output inferensi model, parsing JSON data nutrisi, dan format respons yang diharapkan aplikasi Flutter.

```bash
python 09_integration_test.py
```
* Melakukan validasi konsistensi 100 indeks kelas, uji toleransi nilai gizi, benchmark latensi end-to-end (preprocessing + inferensi + semantic lookup), dan batch stress test.

---

### Tahap 10: Pembuatan Materi Bab IV Skripsi Otomatis (`10_report_generator.py`)
Mengompilasi seluruh metrik evaluasi menjadi tabel ringkasan yang siap disalin ke naskah skripsi Bab IV.

```bash
python 10_report_generator.py
```
* **Output Berada di `reports/bab4/`**:
  - `tabel_4_1_arsitektur_komparasi.csv` (Perbandingan parameter, FLOPs, memori)
  - `tabel_4_2_evaluasi_klasifikasi.csv` (Akurasi Top-1/5 & F1)
  - `tabel_4_3_evaluasi_nutrisi_mae_mape.csv` (Error gizi fisik & persentase)
  - `tabel_4_4_audit_konversi_mobile.csv` (Paritas PyTorch vs ONNX)
  - `checklist_sidang_skripsi.txt` (Daftar kelengkapan materi pengujian untuk sidang)

---

## 6. Panduan Menjalankan Aplikasi Mobile (Flutter)

Aplikasi mobile **Nutrisi Nusantara AI** dibangun menggunakan Flutter dengan dukungan inferensi offline on-device melalui **ONNX Runtime Mobile**.

### 6.1. Verifikasi Prasyarat Flutter
Pastikan Flutter telah terpasang dengan baik:
```bash
flutter doctor
```
*Pastikan Android toolchain dan perangkat target (emulator atau smartphone fisik) terdeteksi.*

### 6.2. Menyiapkan Aset Model & Basis Data Gizi
Sebelum menjalankan aplikasi, pastikan file-file berikut tersedia di dalam folder aset:
1. `mobile_app/assets/nutrition.json` (dihasilkan oleh Tahap 3)
2. `mobile_app/assets/models/convnextv2_nano.onnx` (dihasilkan oleh Tahap 8)
3. `mobile_app/assets/models/convnextv2_nano_int8.onnx` (opsional untuk model kuantisasi)

### 6.3. Instalasi Dependensi Flutter
Pindah ke direktori aplikasi mobile dan unduh pustaka pub:
```bash
cd "mobile_app"
flutter pub get
```

### 6.4. Menjalankan Aplikasi (Mode Debug)
Pastikan emulator Android telah menyala atau smartphone fisik terhubung via kabel data (USB Debugging aktif):
```bash
# Periksa perangkat yang terdeteksi
flutter devices

# Jalankan aplikasi
flutter run
```

### 6.5. Membangun Berkas APK Siap Instal (Release APK)
Untuk menginstal aplikasi secara permanen pada smartphone tanpa perlu terhubung ke komputer:
```bash
flutter build apk --release
```
*Berkas APK rilis akan dihasilkan di:*
`mobile_app/build/app/outputs/flutter-apk/app-release.apk`
*Salin file APK tersebut ke smartphone Android Anda dan lakukan instalasi langsung.*

### 6.6. Fitur-Fitur Utama Aplikasi Mobile
1. **Pemindaian Cerdas (Kamera & Galeri)**:
   - Pengguna dapat memotret hidangan makanan secara langsung atau memilih foto dari galeri HP.
2. **Inferensi On-Device**:
   - Model ConvNeXt V2 melakukan prediksi label makanan dalam hitungan milidetik secara lokal tanpa koneksi internet.
3. **Penyesuaian Porsi Dinamis (Interactive Gram Slider)**:
   - Pengguna dapat menggeser berat porsi makanan (misal: 150 gram, 250 gram), dan aplikasi akan mengkalkulasi ulang total kalori dan makronutrisi secara instan.
4. **Visualisasi Grafik Makronutrisi**:
   - Menampilkan komposisi persentase karbohidrat, protein, dan lemak dalam bentuk donat chart interaktif.
5. **Indikator Angka Kecukupan Gizi (AKG Harian)**:
   - Membantu pengguna memantau persentase asupan gizi terhadap standar kebutuhan harian.
6. **Riwayat Pemindaian (Scan History)**:
   - Menyimpan riwayat makanan yang telah dianalisis untuk pemantauan konsumsi harian.

---

## 7. Cheatsheet Perintah Cepat (Quickrun)

Berikut ringkasan urutan perintah ringkas untuk menjalankan seluruh tahapan dari awal hingga akhir:

```bash
# === 1. SETUP LINGKUNGAN ===
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# === 2. PIPELINE MACHINE LEARNING ===
python 01_dataset_prep.py           # Inisialisasi folder & stratified split
python 02_preprocessing.py          # Verifikasi data loader & augmentasi
python 03_nutrition_setup.py        # Setup & ekspor basis data gizi TKPI 2020
python 04_model.py                  # Profiling arsitektur model SOTA
python 05_training.py --dry-run     # Uji coba pipeline training
python 06_evaluation.py            # Evaluasi metrik klasifikasi test set
python 07_nutrition_mapping.py      # Evaluasi akurasi nilai gizi (MAE/MAPE)
python 08_mobile_conversion.py      # Konversi model ke ONNX & INT8 PTQ
python 09_integration_test.py       # Uji integrasi end-to-end
python 10_report_generator.py       # Generate seluruh tabel Bab IV skripsi

# === 3. DEPLOY & JALANKAN APLIKASI MOBILE ===
# Salin model ONNX ke folder mobile_app
New-Item -ItemType Directory -Force -Path "mobile_app/assets/models"
Copy-Item "models/onnx/convnextv2_nano.onnx" "mobile_app/assets/models/"
Copy-Item "models/onnx/convnextv2_nano_int8.onnx" "mobile_app/assets/models/"

# Jalankan Flutter
cd mobile_app
flutter pub get
flutter run
```

---

## 8. Troubleshooting & Solusi Kendala Umum

### Masalah 1: `OutOfMemoryError` (CUDA OOM) Saat Pelatihan
* **Penyebab**: Kapasitas VRAM GPU tidak mencukupi untuk batch size atau resolusi input saat ini.
* **Solusi**:
  Buka file [config.py](file:///d:/KAMPUS/SEM%207/TA/app%20TA/config.py) dan turunkan parameter `BATCH_SIZE` dari `32` ke `16` atau `8`.
  Aktifkan *Mixed Precision Training* (sudah bawaan di `05_training.py`).

### Masalah 2: File Model ONNX Tidak Ditemukan Saat Aplikasi Mobile Dibuka
* **Penyebab**: Folder `mobile_app/assets/models/` belum berisi file `convnextv2_nano.onnx` atau `convnextv2_nano_int8.onnx`.
* **Solusi**:
  Pastikan Anda telah menjalankan `python 08_mobile_conversion.py` dan menyalin file model dari `models/onnx/` ke `mobile_app/assets/models/`. Jangan lupa lakukan `flutter clean` dan `flutter pub get`.

### Masalah 3: Masalah Eksekusi Script PowerShell (`ExecutionPolicy`)
* **Penyebab**: PowerShell memblokir aktivasi script virtual environment.
* **Solusi**:
  Jalankan perintah berikut di PowerShell dengan hak akses administrator:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### Masalah 4: `onnxruntime` Error pada Android Emulator
* **Penyebab**: Beberapa emulator Android lama berbasis x86 tidak mendukung instruksi binary native ONNX ARM64.
* **Solusi**:
  Gunakan Android Virtual Device (AVD) modern dengan sistem operasi Android 11+ (API 30+) atau langsung uji coba pada smartphone Android fisik dengan arsitektur ARM64.

---

## 9. Poin-Poin Ilmiah Kunci untuk Sidang Sarjana (Defense Arguments)

1. **Mengapa Menggunakan ConvNeXt V2 dan Bukan CNN Konvensional?**
   - *Jawaban Ilmiah*: CNN klasik (seperti ResNet standar) memiliki keterbatasan receptive field dan rentan jenuh. ConvNeXt V2 mengintegrasikan filosofi modern Vision Transformer (7×7 depthwise convolutions, inverted bottleneck, LayerNorm menggantikan BatchNorm, dan mekanisme Global Response Normalization / GRN) yang terbukti unggul pada data citra natural dengan variasi tekstur tinggi seperti masakan Indonesia.

2. **Mengapa Klasifikasi Citra dan Estimasi Nutrisi Dipisahkan (Decoupled Architecture)?**
   - *Jawaban Ilmiah*: Estimasi gizi secara langsung melalui regresi citra (*end-to-end visual regression*) menderita ambiguitas tinggi karena kandungan tersembunyi (kadar gula, garam, minyak) tidak dapat diukur secara optik dari piksel. Arsitektur decoupled memisahkan tugas *object recognition* (domain visi komputer) dari *semantic lookup* berbasis standar laboratorium gizi terverifikasi (**TKPI Kemenkes**), sehingga data porsi dan komposisi gizi dapat diperbarui atau divalidasi oleh ahli gizi kapan saja tanpa perlu melatih ulang jaringan syaraf tiruan.

3. **Mengapa Memilih ONNX Runtime Mobile Dibandingkan TensorFlow Lite (TFLite)?**
   - *Jawaban Ilmiah*: Arsitektur model SOTA modern (2023–2024) memanfaatkan operator-operator baru seperti GRN dan LayerNorm. Konversi ke TFLite sering mengalami kegagalan *Unsupported Operations* yang membutuhkan implementasi C++ custom kernel. ONNX Opset 17 menyediakan dukungan native standar industri terhadap seluruh operator SOTA, menjamin paritas numerik (*cosine similarity > 0.999*) identik dengan lingkungan pelatihan PyTorch.
