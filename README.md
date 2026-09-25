# Nutrisi Nusantara AI — Aplikasi Mobile Klasifikasi & Estimasi Nutrisi Makanan Indonesia

## Deskripsi
Aplikasi mobile berbasis Flutter yang menggunakan Convolutional Neural Network (CNN)
state-of-the-art untuk mengklasifikasikan 100 kelas makanan Indonesia dan mengestimasi
kandungan nutrisinya secara otomatis.

## Arsitektur Model
- **Model Utama**: ConvNeXt V2-Nano (SOTA 2023, FCMAE pre-trained, IN-22K fine-tuned)
- **Model Pembanding**: MobileNetV4-Conv-Medium, EfficientNetV2-S
- **Format Mobile**: ONNX Runtime Mobile (Opset 17)

## Teknologi
- **Frontend**: Flutter (Dart) — Cross-platform Android & iOS
- **AI Engine**: ONNX Runtime Mobile
- **Backend ML**: PyTorch + timm library
- **Basis Data Gizi**: TKPI 2020 / Kemenkes RI (100 kelas, JSON/CSV)

## Dokumentasi Lengkap
> 📖 **Panduan Menyeluruh**: Untuk panduan lengkap cara menjalankan seluruh tahapan (Python pipeline Tahap 1 - 10 hingga aplikasi Flutter) serta penjelasan mendalam mengenai cara kerja dan arsitektur sistem, silakan baca:
> **[CARA_RUN_DAN_CARA_KERJA.md](CARA_RUN_DAN_CARA_KERJA.md)**

## Cara Menjalankan Singkat

### 1. Prerequisites
```bash
# Python 3.10+ & Virtual Environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Flutter SDK (untuk mobile)
flutter doctor
```

### 2. Clone & Setup
```bash
git clone https://github.com/VEL1XI/TUGAS-AKHIR.git
cd TUGAS-AKHIR/mobile_app
flutter pub get
```

### 3. Taruh File Model ONNX
Salin file model hasil konversi ke folder `assets/models/`:
```
mobile_app/assets/models/convnextv2_nano.onnx        # Model utama FP32
mobile_app/assets/models/convnextv2_nano_int8.onnx    # Model quantized INT8
```

### 4. Jalankan Aplikasi
```bash
flutter run                     # Debug mode (USB/emulator)
flutter build apk --release     # Build APK release
```

## Struktur Proyek

```
├── config.py                    # Konfigurasi global (100 kelas, path, hyperparameter)
├── 01_dataset_prep.py           # Persiapan & splitting dataset
├── 02_preprocessing.py          # Augmentasi & DataLoader
├── 03_nutrition_setup.py        # Setup basis data nutrisi
├── 04_model.py                  # Builder arsitektur SOTA (3 model)
├── 05_training.py               # Pipeline training multi-fase
├── 06_evaluation.py             # Evaluasi model (Top-1/5, F1, Confusion Matrix)
├── 07_nutrition_mapping.py      # Mapping prediksi → nutrisi (MAE/MAPE)
├── 08_mobile_conversion.py      # Konversi ONNX & quantization
├── mobile_app/                  # Aplikasi Flutter
│   ├── lib/
│   │   ├── main.dart            # Entry point
│   │   ├── models/              # Data model Dart
│   │   ├── screens/             # UI screens (Splash, Home, Result, History)
│   │   └── services/            # Classifier & Nutrition service
│   ├── assets/
│   │   ├── nutrition.json       # Basis data gizi offline
│   │   └── models/              # File model ONNX
│   └── pubspec.yaml             # Dependencies Flutter
├── nutrition/                   # Data nutrisi CSV/JSON
└── reports/                     # Laporan evaluasi & Bab IV
```

## Lisensi
Proyek Tugas Akhir S1 Informatika. Hak cipta dilindungi.
