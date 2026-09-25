# ==============================================================================
# config.py — KONFIGURASI GLOBAL PROYEK
# ==============================================================================
# File ini menyimpan semua konstanta dan path yang digunakan di seluruh skrip.
# Dengan memusatkan konfigurasi di satu file, perubahan (misal: jumlah kelas,
# ukuran input, rasio split) cukup dilakukan di sini tanpa mengedit banyak file.
# ==============================================================================

import os
from pathlib import Path

# ==============================================================================
# 1. PATH DIREKTORI PROYEK
# ==============================================================================
# BASE_DIR = root folder proyek (di mana file config.py ini berada)
BASE_DIR = Path(__file__).resolve().parent

# Direktori data
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"           # Data mentah, satu sub-folder per kelas
TRAIN_DIR = DATA_DIR / "train"            # Hasil split: data training
VAL_DIR = DATA_DIR / "val"                # Hasil split: data validasi
TEST_DIR = DATA_DIR / "test"              # Hasil split: data testing

# Direktori model
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"   # Model PyTorch (.pth)
ONNX_DIR = MODELS_DIR / "onnx"                   # Model ONNX (.onnx)
TFLITE_DIR = MODELS_DIR / "tflite"               # Model TFLite (.tflite)

# Direktori output
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"     # Grafik & visualisasi untuk laporan
LOGS_DIR = BASE_DIR / "logs"             # TensorBoard logs

# Direktori data nutrisi
NUTRITION_DIR = BASE_DIR / "nutrition"    # File CSV/JSON data nutrisi

# Direktori aplikasi mobile
MOBILE_APP_DIR = BASE_DIR / "mobile_app"  # Kode Flutter

# ==============================================================================
# 2. PARAMETER DATASET
# ==============================================================================
# Rasio pembagian dataset (harus berjumlah 1.0)
TRAIN_RATIO = 0.70    # 70% untuk training
VAL_RATIO = 0.15      # 15% untuk validasi (tuning hyperparameter)
TEST_RATIO = 0.15     # 15% untuk testing (evaluasi akhir, tidak boleh dipakai saat training)

# Random seed untuk reprodusibilitas — PENTING untuk TA agar hasil bisa direplikasi
RANDOM_SEED = 42

# Ekstensi file gambar yang dianggap valid
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Batas minimum jumlah gambar per kelas (kelas dengan gambar kurang dari ini
# akan ditandai sebagai "under-represented" dan perlu augmentasi lebih agresif)
MIN_IMAGES_PER_CLASS = 100

# ==============================================================================
# 3. DAFTAR 100 KELAS MAKANAN INDONESIA
# ==============================================================================
# Daftar ini HARUS konsisten dengan nama folder di data/raw/
# Format: nama folder (lowercase, underscore) yang juga menjadi label kelas
# Urutan di list ini menentukan indeks kelas (0-99)
CLASS_NAMES = [
    # === NASI & OLAHAN BERAS (12 kelas, indeks 0-11) ===
    "nasi_goreng",          # 0
    "nasi_putih",           # 1
    "nasi_padang",          # 2
    "nasi_uduk",            # 3
    "nasi_kuning",          # 4
    "nasi_pecel",           # 5
    "gudeg",                # 6
    "ketupat",              # 7
    "lontong_sayur",        # 8
    "bubur_ayam",           # 9
    "nasi_liwet",           # 10
    "nasi_timbel",          # 11

    # === MIE & BAKSO (8 kelas, indeks 12-19) ===
    "mie_goreng",           # 12
    "mie_ayam",             # 13
    "mie_aceh",             # 14
    "bakso",                # 15
    "bakmi_jawa",           # 16
    "kwetiau_goreng",       # 17
    "bihun_goreng",         # 18
    "indomie",              # 19

    # === SUP, SOTO & KUAH (13 kelas, indeks 20-32) ===
    "soto_ayam",            # 20
    "soto_betawi",          # 21
    "soto_lamongan",        # 22
    "soto_banjar",          # 23
    "rawon",                # 24
    "tongseng",             # 25
    "sup_buntut",           # 26
    "sayur_asem",           # 27
    "sayur_lodeh",          # 28
    "opor_ayam",            # 29
    "gulai_ayam",           # 30
    "gulai_ikan",           # 31
    "sop_kambing",          # 32

    # === DAGING & UNGGAS (17 kelas, indeks 33-49) ===
    "rendang",              # 33
    "ayam_goreng",          # 34
    "ayam_bakar",           # 35
    "ayam_geprek",          # 36
    "ayam_penyet",          # 37
    "ayam_betutu",          # 38
    "sate_ayam",            # 39
    "sate_kambing",         # 40
    "sate_padang",          # 41
    "sate_lilit",           # 42
    "sate_madura",          # 43
    "dendeng_balado",       # 44
    "empal",                # 45
    "bebek_goreng",         # 46
    "gulai_kambing",        # 47
    "semur_daging",         # 48
    "krengsengan",          # 49

    # === IKAN & SEAFOOD (10 kelas, indeks 50-59) ===
    "ikan_bakar",           # 50
    "ikan_goreng",          # 51
    "pempek",               # 52
    "udang_goreng_tepung",  # 53
    "cumi_goreng_tepung",   # 54
    "pepes_ikan",           # 55
    "ikan_asam_manis",      # 56
    "kepiting_saus_padang", # 57
    "otak_otak",            # 58
    "ikan_pindang",         # 59

    # === SAYURAN & SALAD (10 kelas, indeks 60-69) ===
    "gado_gado",            # 60
    "karedok",              # 61
    "pecel",                # 62
    "urap",                 # 63
    "lalapan",              # 64
    "cap_cay",              # 65
    "tumis_kangkung",       # 66
    "capcay_goreng",        # 67
    "asinan_jakarta",       # 68
    "rujak",                # 69

    # === GORENGAN & CAMILAN (19 kelas, indeks 70-88) ===
    "tahu_goreng",          # 70
    "tempe_goreng",         # 71
    "bakwan",               # 72
    "pisang_goreng",        # 73
    "tahu_isi",             # 74
    "risoles",              # 75
    "lumpia",               # 76
    "pastel",               # 77
    "siomay",               # 78
    "batagor",              # 79
    "martabak_telur",       # 80
    "martabak_manis",       # 81
    "cireng",               # 82
    "cilok",                # 83
    "ketoprak",             # 84
    "perkedel",             # 85
    "telur_balado",         # 86
    "telur_dadar",          # 87
    "tahu_gejrot",          # 88

    # === KUE & DESSERT (11 kelas, indeks 89-99) ===
    "serabi",               # 89
    "klepon",               # 90
    "kue_lapis",            # 91
    "dadar_gulung",         # 92
    "onde_onde",            # 93
    "getuk",                # 94
    "kue_putu",             # 95
    "es_cendol",            # 96
    "es_teler",             # 97
    "kolak",                # 98
    "bika_ambon",           # 99
]

# Jumlah kelas total (dihitung otomatis dari list di atas)
NUM_CLASSES = len(CLASS_NAMES)  # Harus = 100

# Mapping nama kelas → indeks dan sebaliknya (untuk prediksi)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ==============================================================================
# 4. PARAMETER MODEL & TRAINING
# ==============================================================================
# Ukuran input gambar — berbeda per arsitektur, tapi kita standarkan ke 224x224
# untuk perbandingan fair antar model. Bisa di-override per model jika perlu.
IMG_SIZE = 224

# Batch size — sesuaikan dengan VRAM GPU Anda
# - RTX 3060 (12GB): batch_size=32 aman untuk semua 3 arsitektur
# - Google Colab Free (T4 16GB): batch_size=32-64
# - Colab Pro (A100 40GB): batch_size=64-128
BATCH_SIZE = 32

# Jumlah epoch training
NUM_EPOCHS = 50

# Learning rate awal (akan diatur oleh scheduler)
LEARNING_RATE = 1e-4

# Jumlah worker untuk DataLoader (0 = main thread saja, cocok untuk Windows)
# Di Linux/Mac bisa naikkan ke 4-8 untuk loading data paralel
NUM_WORKERS = 0 if os.name == 'nt' else 4

# ==============================================================================
# 5. KONFIGURASI TIGA ARSITEKTUR SOTA
# ==============================================================================
# Dictionary berisi nama model di timm library dan parameter spesifiknya.
# Ini memudahkan loop perbandingan antar arsitektur di Tahap 5.
MODEL_CONFIGS = {
    "efficientnetv2_s": {
        "timm_name": "tf_efficientnetv2_s.in21k_ft_in1k",  # Pre-trained ImageNet-21K, fine-tuned 1K
        "input_size": 384,      # Default input size untuk EfficientNetV2-S
        "paper_name": "EfficientNetV2-S",
        "year": 2021,
        "params_millions": 21.5,
    },
    "convnextv2_nano": {
        "timm_name": "convnextv2_nano.fcmae_ft_in22k_in1k",  # FCMAE pre-trained, IN-22K fine-tuned
        "input_size": 224,      # Default input size untuk ConvNeXt V2-Nano
        "paper_name": "ConvNeXt V2-Nano",
        "year": 2023,
        "params_millions": 15.6,
    },
    "mobilenetv4_conv_medium": {
        "timm_name": "mobilenetv4_conv_medium.e500_r256_in1k",  # ImageNet-1K trained
        "input_size": 256,      # Default input size untuk MobileNetV4-Conv-Medium
        "paper_name": "MobileNetV4-Conv-Medium",
        "year": 2024,
        "params_millions": 9.7,
    },
}

# ==============================================================================
# 6. LABEL KATEGORI (untuk visualisasi dan laporan)
# ==============================================================================
# Mapping kelas → kategori makanan (berguna untuk analisis per kategori di Bab IV)
CATEGORY_MAP = {
    "Nasi & Olahan Beras": CLASS_NAMES[0:12],
    "Mie & Bakso": CLASS_NAMES[12:20],
    "Sup, Soto & Kuah": CLASS_NAMES[20:33],
    "Daging & Unggas": CLASS_NAMES[33:50],
    "Ikan & Seafood": CLASS_NAMES[50:60],
    "Sayuran & Salad": CLASS_NAMES[60:70],
    "Gorengan & Camilan": CLASS_NAMES[70:89],
    "Kue & Dessert": CLASS_NAMES[89:100],
}

# Reverse mapping: nama kelas → kategori
CLASS_TO_CATEGORY = {}
for category, classes in CATEGORY_MAP.items():
    for cls in classes:
        CLASS_TO_CATEGORY[cls] = category
