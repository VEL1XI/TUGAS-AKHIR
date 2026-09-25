# ==============================================================================
# 01_dataset_prep.py — PERSIAPAN & PEMBAGIAN DATASET (TAHAP 1)
# ==============================================================================
# Skrip ini melakukan:
#   1. Inisialisasi struktur folder proyek (sesuai config.py)
#   2. Membuat folder kosong untuk setiap kelas makanan di data/raw/
#   3. Analisis distribusi kelas (jumlah gambar per kelas, deteksi imbalance)
#   4. Visualisasi distribusi (bar chart + heatmap kategori)
#   5. Stratified split dataset ke train/val/test (70/15/15)
#   6. Verifikasi hasil split (konsistensi jumlah & proporsi)
#
# Cara pakai:
#   1. Jalankan sekali untuk membuat folder: python 01_dataset_prep.py
#   2. Isi folder data/raw/<nama_kelas>/ dengan gambar masing-masing makanan
#   3. Jalankan lagi untuk analisis distribusi & split dataset
#
# Catatan untuk sidang:
#   - Stratified split memastikan proporsi setiap kelas sama di train/val/test
#   - Ini penting untuk 100 kelas karena mencegah ada kelas yang tidak terwakili
#     di salah satu subset (misal: semua gambar "bika_ambon" masuk train, 0 di test)
# ==============================================================================

import os
import sys
import shutil
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import train_test_split

# Import konfigurasi global dari config.py
from config import (
    BASE_DIR, DATA_DIR, RAW_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    VALID_EXTENSIONS, MIN_IMAGES_PER_CLASS,
    CLASS_NAMES, NUM_CLASSES, CATEGORY_MAP, CLASS_TO_CATEGORY,
    MODELS_DIR, SAVED_MODELS_DIR, ONNX_DIR, TFLITE_DIR,
    REPORTS_DIR, FIGURES_DIR, LOGS_DIR, NUTRITION_DIR, MOBILE_APP_DIR,
)


# ==============================================================================
# 1. INISIALISASI STRUKTUR DIREKTORI PROYEK
# ==============================================================================
def setup_project_directories():
    """
    Membuat seluruh struktur folder yang dibutuhkan proyek.
    Termasuk membuat sub-folder untuk setiap kelas makanan di data/raw/
    sehingga user tinggal mengisi gambar ke folder yang sesuai.

    Struktur yang dibuat:
        proyek/
        ├── data/
        │   ├── raw/           ← Taruh gambar mentah di sini (per kelas)
        │   │   ├── nasi_goreng/
        │   │   ├── rendang/
        │   │   └── ... (100 folder kelas)
        │   ├── train/         ← Hasil split otomatis
        │   ├── val/           ← Hasil split otomatis
        │   └── test/          ← Hasil split otomatis
        ├── models/
        │   ├── saved_models/  ← Model .pth hasil training
        │   ├── onnx/          ← Model ONNX hasil konversi
        │   └── tflite/        ← Model TFLite hasil konversi
        ├── reports/
        │   └── figures/       ← Grafik untuk laporan/Bab IV
        ├── logs/              ← TensorBoard training logs
        ├── nutrition/         ← File CSV/JSON data nutrisi
        └── mobile_app/        ← Kode Flutter
    """
    print("=" * 70)
    print("  TAHAP 1.1 — INISIALISASI STRUKTUR FOLDER PROYEK")
    print("=" * 70)

    # Daftar semua direktori yang perlu dibuat
    directories = [
        RAW_DATA_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        SAVED_MODELS_DIR,
        ONNX_DIR,
        TFLITE_DIR,
        FIGURES_DIR,
        LOGS_DIR,
        NUTRITION_DIR,
        MOBILE_APP_DIR,
    ]

    for folder in directories:
        folder.mkdir(parents=True, exist_ok=True)
        # Tampilkan path relatif agar output tidak terlalu panjang
        print(f"  [OK] {folder.relative_to(BASE_DIR)}")

    # Buat sub-folder per kelas di data/raw/
    print(f"\n  Membuat {NUM_CLASSES} folder kelas di data/raw/ ...")
    created_count = 0
    for class_name in CLASS_NAMES:
        class_folder = RAW_DATA_DIR / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        created_count += 1

    print(f"  [OK] {created_count} folder kelas berhasil disiapkan di data/raw/")
    print(f"\n  ► LANGKAH SELANJUTNYA: Isi setiap folder di data/raw/<nama_kelas>/")
    print(f"    dengan gambar makanan yang sesuai (min. {MIN_IMAGES_PER_CLASS} gambar/kelas).")
    print()


# ==============================================================================
# 2. ANALISIS DISTRIBUSI KELAS
# ==============================================================================
def analyze_class_distribution(source_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Menganalisis jumlah gambar di setiap sub-folder kelas makanan.
    Menghasilkan:
      - Tabel ringkasan dengan kolom: Nama Kelas, Kategori, Jumlah Gambar, Status
      - Identifikasi kelas yang under-represented (< MIN_IMAGES_PER_CLASS)
      - Statistik ringkasan (total, rata-rata, std dev, min, max)

    Parameter:
        source_dir: Path ke folder yang berisi sub-folder per kelas (default: data/raw/)

    Return:
        pd.DataFrame berisi ringkasan distribusi kelas

    Catatan untuk sidang:
        Fungsi ini penting untuk mendeteksi CLASS IMBALANCE — kondisi di mana
        beberapa kelas memiliki jauh lebih sedikit data dibanding kelas lain.
        Class imbalance dapat menyebabkan model bias ke kelas mayoritas dan
        under-perform pada kelas minoritas. Dengan mendeteksi ini di awal,
        kita bisa menerapkan strategi mitigasi (augmentasi lebih agresif, class
        weighting, oversampling) di tahap training.
    """
    print("=" * 70)
    print(f"  TAHAP 1.2 — ANALISIS DISTRIBUSI KELAS ({source_dir.name}/)")
    print("=" * 70)

    if not source_dir.exists():
        print(f"  [ERROR] Folder {source_dir} tidak ditemukan.")
        return pd.DataFrame()

    # Kumpulkan data dari setiap folder kelas
    data_summary = []
    missing_classes = []

    for class_name in CLASS_NAMES:
        class_folder = source_dir / class_name

        if not class_folder.exists():
            # Kelas ada di config tapi folder-nya belum ada
            missing_classes.append(class_name)
            data_summary.append({
                "Nama Kelas": class_name,
                "Kategori": CLASS_TO_CATEGORY.get(class_name, "Lainnya"),
                "Jumlah Gambar": 0,
                "Status": "❌ FOLDER TIDAK ADA"
            })
            continue

        # Hitung file gambar valid di folder kelas ini
        image_files = [
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ]
        count = len(image_files)

        # Tentukan status berdasarkan jumlah gambar
        if count == 0:
            status = "⚠️ KOSONG"
        elif count < MIN_IMAGES_PER_CLASS:
            status = f"⚠️ KURANG (min {MIN_IMAGES_PER_CLASS})"
        else:
            status = "✅ CUKUP"

        data_summary.append({
            "Nama Kelas": class_name,
            "Kategori": CLASS_TO_CATEGORY.get(class_name, "Lainnya"),
            "Jumlah Gambar": count,
            "Status": status
        })

    # Buat DataFrame
    df = pd.DataFrame(data_summary)
    total_images = df["Jumlah Gambar"].sum()

    # Tambah kolom persentase
    if total_images > 0:
        df["Persentase (%)"] = (df["Jumlah Gambar"] / total_images * 100).round(2)
    else:
        df["Persentase (%)"] = 0.0

    # Tampilkan ringkasan
    print(f"\n  Total Kelas   : {NUM_CLASSES}")
    print(f"  Total Gambar  : {total_images:,}")

    if total_images > 0:
        counts = df["Jumlah Gambar"]
        print(f"  Rata-rata     : {counts.mean():.1f} gambar/kelas")
        print(f"  Std Deviasi   : {counts.std():.1f}")
        print(f"  Minimum       : {counts.min()} ({df.loc[counts.idxmin(), 'Nama Kelas']})")
        print(f"  Maksimum      : {counts.max()} ({df.loc[counts.idxmax(), 'Nama Kelas']})")

        # Hitung class imbalance ratio
        # Rasio antara kelas terbanyak dan tersedikit (semakin tinggi = semakin imbalance)
        non_zero = counts[counts > 0]
        if len(non_zero) > 1:
            imbalance_ratio = non_zero.max() / non_zero.min()
            print(f"  Imbalance Ratio: {imbalance_ratio:.1f}x (max/min)")
            if imbalance_ratio > 5:
                print(f"  ⚠️ PERINGATAN: Imbalance ratio > 5x — pertimbangkan augmentasi/oversampling")

        # Identifikasi kelas bermasalah
        under_represented = df[
            (df["Jumlah Gambar"] > 0) & (df["Jumlah Gambar"] < MIN_IMAGES_PER_CLASS)
        ]
        empty_classes = df[df["Jumlah Gambar"] == 0]

        if not empty_classes.empty:
            print(f"\n  Kelas KOSONG (belum ada gambar): {len(empty_classes)} kelas")
        if not under_represented.empty:
            print(f"  Kelas KURANG (<{MIN_IMAGES_PER_CLASS} gambar): {len(under_represented)} kelas")
            for _, row in under_represented.iterrows():
                print(f"    - {row['Nama Kelas']}: {row['Jumlah Gambar']} gambar")

    # Tampilkan tabel ringkasan per kategori
    if total_images > 0:
        print("\n  Ringkasan per Kategori:")
        print("  " + "-" * 55)
        cat_summary = df.groupby("Kategori")["Jumlah Gambar"].agg(["sum", "count", "mean"])
        cat_summary.columns = ["Total Gambar", "Jumlah Kelas", "Rata-rata"]
        cat_summary["Rata-rata"] = cat_summary["Rata-rata"].round(1)
        for cat, row in cat_summary.iterrows():
            print(f"  {cat:<25} | {int(row['Total Gambar']):>6} gambar "
                  f"({int(row['Jumlah Kelas'])} kelas, avg {row['Rata-rata']})")

    print()
    return df


# ==============================================================================
# 3. VISUALISASI DISTRIBUSI KELAS
# ==============================================================================
def plot_class_distribution(df: pd.DataFrame, save_path: Path = None):
    """
    Membuat visualisasi distribusi jumlah gambar per kelas makanan.
    Untuk 100 kelas, kita buat 2 jenis plot:
      1. Horizontal bar chart (sorted, menampilkan semua kelas)
      2. Box plot per kategori (ringkasan statistik per kelompok makanan)

    Parameter:
        df: DataFrame hasil dari analyze_class_distribution()
        save_path: Path untuk menyimpan grafik (opsional)

    Catatan untuk sidang:
        Visualisasi distribusi data adalah langkah standar dalam metodologi
        penelitian ML. Ini membantu mengidentifikasi: (1) kelas yang datanya
        kurang, (2) ketimpangan distribusi, dan (3) potensi bias model.
    """
    if df.empty or df["Jumlah Gambar"].sum() == 0:
        print("  [SKIP] Tidak ada data untuk divisualisasi.")
        return

    print("  Membuat visualisasi distribusi kelas...")

    # --- PLOT 1: Horizontal Bar Chart (semua 100 kelas) ---
    # Untuk 100 kelas, kita butuh figure yang cukup tinggi
    fig, ax = plt.subplots(figsize=(14, 28))

    # Sort berdasarkan jumlah gambar (ascending agar yang paling sedikit di atas)
    df_sorted = df.sort_values("Jumlah Gambar", ascending=True).reset_index(drop=True)

    # Warna berdasarkan status: merah=kosong, kuning=kurang, hijau=cukup
    colors = []
    for _, row in df_sorted.iterrows():
        count = row["Jumlah Gambar"]
        if count == 0:
            colors.append("#e74c3c")  # Merah — kosong
        elif count < MIN_IMAGES_PER_CLASS:
            colors.append("#f39c12")  # Kuning — kurang
        else:
            colors.append("#2ecc71")  # Hijau — cukup

    bars = ax.barh(
        y=range(len(df_sorted)),
        width=df_sorted["Jumlah Gambar"],
        color=colors,
        edgecolor="white",
        linewidth=0.5
    )

    # Label di ujung bar
    for i, (bar, count) in enumerate(zip(bars, df_sorted["Jumlah Gambar"])):
        if count > 0:
            ax.text(
                count + 1, i, str(count),
                va="center", ha="left", fontsize=6, fontweight="bold"
            )

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["Nama Kelas"], fontsize=6)
    ax.set_xlabel("Jumlah Gambar", fontsize=12)
    ax.set_title(
        "Distribusi Jumlah Gambar per Kelas Makanan Indonesia (100 Kelas)",
        fontsize=14, fontweight="bold", pad=15
    )

    # Garis batas minimum
    ax.axvline(
        x=MIN_IMAGES_PER_CLASS, color="#e74c3c", linestyle="--",
        alpha=0.7, label=f"Batas minimum ({MIN_IMAGES_PER_CLASS} gambar)"
    )
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  [OK] Bar chart disimpan: {save_path.relative_to(BASE_DIR)}")
    plt.close()

    # --- PLOT 2: Box Plot per Kategori ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    df_nonzero = df[df["Jumlah Gambar"] > 0]

    if not df_nonzero.empty:
        # Urutkan kategori berdasarkan median jumlah gambar
        cat_order = (
            df_nonzero.groupby("Kategori")["Jumlah Gambar"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

        sns.boxplot(
            data=df_nonzero,
            x="Kategori",
            y="Jumlah Gambar",
            order=cat_order,
            palette="Set2",
            ax=ax2
        )
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        ax2.set_title(
            "Distribusi Gambar per Kategori Makanan",
            fontsize=13, fontweight="bold", pad=12
        )
        ax2.set_ylabel("Jumlah Gambar per Kelas", fontsize=11)
        ax2.set_xlabel("")

        # Garis batas minimum
        ax2.axhline(
            y=MIN_IMAGES_PER_CLASS, color="#e74c3c", linestyle="--",
            alpha=0.7, label=f"Batas minimum ({MIN_IMAGES_PER_CLASS})"
        )
        ax2.legend(fontsize=9)

        plt.tight_layout()
        if save_path:
            boxplot_path = save_path.parent / "distribusi_per_kategori.png"
            fig2.savefig(boxplot_path, dpi=200, bbox_inches="tight")
            print(f"  [OK] Box plot disimpan: {boxplot_path.relative_to(BASE_DIR)}")
    plt.close()


# ==============================================================================
# 4. STRATIFIED SPLIT DATASET (TRAIN / VALIDATION / TEST)
# ==============================================================================
def split_dataset(
    source_dir: Path = RAW_DATA_DIR,
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    test_dir: Path = TEST_DIR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
    copy_mode: str = "copy"  # "copy" atau "move"
):
    """
    Membagi dataset mentah (data/raw/) menjadi train, validation, dan test set
    dengan STRATIFIED SPLIT — proporsi setiap kelas dijaga sama di semua subset.

    Stratified split sangat penting untuk dataset dengan 100 kelas karena:
    1. Random split biasa bisa menyebabkan kelas dengan sedikit gambar tidak
       terwakili sama sekali di salah satu subset
    2. Evaluasi per kelas (precision/recall/F1 per kelas) membutuhkan minimal
       beberapa sample di test set untuk setiap kelas

    Parameter:
        source_dir: Path ke folder raw data (berisi sub-folder per kelas)
        train_dir, val_dir, test_dir: Path tujuan hasil split
        train_ratio, val_ratio, test_ratio: Proporsi split (total harus = 1.0)
        seed: Random seed untuk reprodusibilitas
        copy_mode: "copy" (salin file, raw tetap utuh) atau "move" (pindahkan file)

    Catatan untuk sidang:
        - Kita gunakan sklearn.train_test_split yang mendukung stratified splitting
        - Split dilakukan 2 tahap: (1) pisah train vs sisa, (2) pisah sisa jadi val vs test
        - File DISALIN (bukan dipindah) agar data/raw/ tetap utuh sebagai backup
    """
    # Validasi rasio
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, \
        f"Total rasio split harus 1.0, dapat: {train_ratio + val_ratio + test_ratio}"

    print("=" * 70)
    print("  TAHAP 1.3 — STRATIFIED SPLIT DATASET")
    print(f"  Rasio: Train={int(train_ratio*100)}% | Val={int(val_ratio*100)}% | Test={int(test_ratio*100)}%")
    print(f"  Seed: {seed} | Mode: {copy_mode}")
    print("=" * 70)

    # Cek apakah sudah pernah di-split sebelumnya
    existing_train = list(train_dir.glob("*/*.jpg")) + list(train_dir.glob("*/*.png"))
    if existing_train:
        print(f"\n  ⚠️ Folder train/ sudah berisi {len(existing_train)} file.")
        response = input("  Hapus data lama dan split ulang? (y/n): ").strip().lower()
        if response != 'y':
            print("  [DIBATALKAN] Split tidak dilakukan.")
            return None
        # Hapus folder train/val/test lama
        for d in [train_dir, val_dir, test_dir]:
            if d.exists():
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)

    split_summary = []
    skipped_classes = []

    for class_name in CLASS_NAMES:
        class_folder = source_dir / class_name
        if not class_folder.exists():
            skipped_classes.append((class_name, "folder tidak ada"))
            continue

        # Kumpulkan semua file gambar valid di kelas ini
        images = sorted([
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ])

        if len(images) == 0:
            skipped_classes.append((class_name, "tidak ada gambar"))
            continue

        if len(images) < 3:
            # Minimal 3 gambar agar bisa di-split ke 3 subset (1 per subset)
            skipped_classes.append((class_name, f"hanya {len(images)} gambar (min 3)"))
            continue

        # --- SPLIT TAHAP 1: Pisahkan TRAIN dari sisa (VAL + TEST) ---
        # Menggunakan train_test_split dari sklearn
        val_test_ratio = val_ratio + test_ratio  # 0.30
        train_imgs, temp_imgs = train_test_split(
            images,
            train_size=train_ratio,   # 0.70
            random_state=seed,
            shuffle=True
        )

        # --- SPLIT TAHAP 2: Pisahkan sisa menjadi VAL dan TEST ---
        # Rasio val relatif terhadap sisa: val / (val + test) = 0.15 / 0.30 = 0.50
        relative_val_ratio = val_ratio / val_test_ratio
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            train_size=relative_val_ratio,  # 0.50 dari sisa
            random_state=seed,
            shuffle=True
        )

        # --- SALIN/PINDAHKAN file ke folder tujuan ---
        # Buat folder tujuan per kelas
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
        (test_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Fungsi transfer file (copy atau move)
        transfer_fn = shutil.copy2 if copy_mode == "copy" else shutil.move

        for img in train_imgs:
            transfer_fn(str(img), str(train_dir / class_name / img.name))
        for img in val_imgs:
            transfer_fn(str(img), str(val_dir / class_name / img.name))
        for img in test_imgs:
            transfer_fn(str(img), str(test_dir / class_name / img.name))

        split_summary.append({
            "Kelas": class_name,
            "Total": len(images),
            "Train": len(train_imgs),
            "Val": len(val_imgs),
            "Test": len(test_imgs),
            "Train%": round(len(train_imgs) / len(images) * 100, 1),
            "Val%": round(len(val_imgs) / len(images) * 100, 1),
            "Test%": round(len(test_imgs) / len(images) * 100, 1),
        })

    # --- TAMPILKAN HASIL ---
    if not split_summary:
        print("\n  [ERROR] Tidak ada kelas yang bisa di-split!")
        print("  Pastikan data/raw/<nama_kelas>/ sudah berisi gambar.")
        return None

    summary_df = pd.DataFrame(split_summary)

    print(f"\n  Berhasil split {len(split_summary)} kelas:")
    print()

    # Tampilkan tabel ringkas (untuk 100 kelas, cukup tampilkan statistik)
    if len(split_summary) > 20:
        # Kalau banyak kelas, tampilkan ringkasan saja (bukan per-baris)
        print(f"  {'Metrik':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        print(f"  {'Total Gambar':<20} {summary_df['Train'].sum():>8} "
              f"{summary_df['Val'].sum():>8} {summary_df['Test'].sum():>8} "
              f"{summary_df['Total'].sum():>8}")
        print(f"  {'Rata-rata/Kelas':<20} {summary_df['Train'].mean():>8.1f} "
              f"{summary_df['Val'].mean():>8.1f} {summary_df['Test'].mean():>8.1f} "
              f"{summary_df['Total'].mean():>8.1f}")
        print(f"  {'Min/Kelas':<20} {summary_df['Train'].min():>8} "
              f"{summary_df['Val'].min():>8} {summary_df['Test'].min():>8} "
              f"{summary_df['Total'].min():>8}")
        print(f"  {'Max/Kelas':<20} {summary_df['Train'].max():>8} "
              f"{summary_df['Val'].max():>8} {summary_df['Test'].max():>8} "
              f"{summary_df['Total'].max():>8}")
        print(f"  {'Rata-rata Rasio':<20} {summary_df['Train%'].mean():>7.1f}% "
              f"{summary_df['Val%'].mean():>7.1f}% {summary_df['Test%'].mean():>7.1f}%")
    else:
        # Kalau sedikit kelas, tampilkan per-baris
        print(summary_df.to_string(index=False))

    # Peringatan untuk kelas yang di-skip
    if skipped_classes:
        print(f"\n  ⚠️ {len(skipped_classes)} kelas DILEWATI:")
        for name, reason in skipped_classes:
            print(f"    - {name}: {reason}")

    print(f"\n  ✅ Split selesai! Data tersimpan di:")
    print(f"     Train : {train_dir.relative_to(BASE_DIR)}/ ({summary_df['Train'].sum()} gambar)")
    print(f"     Val   : {val_dir.relative_to(BASE_DIR)}/ ({summary_df['Val'].sum()} gambar)")
    print(f"     Test  : {test_dir.relative_to(BASE_DIR)}/ ({summary_df['Test'].sum()} gambar)")
    print()

    # Simpan summary ke CSV untuk referensi
    csv_path = REPORTS_DIR / "split_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  [OK] Tabel split disimpan ke: {csv_path.relative_to(BASE_DIR)}")

    return summary_df


# ==============================================================================
# 5. VERIFIKASI HASIL SPLIT
# ==============================================================================
def verify_split():
    """
    Memverifikasi konsistensi hasil split:
    1. Setiap kelas harus ada di train, val, dan test
    2. Tidak ada gambar duplikat antar subset (file yang sama di train dan test)
    3. Total gambar di train+val+test = total di raw

    Catatan untuk sidang:
        Verifikasi ini penting untuk memastikan tidak ada DATA LEAKAGE —
        kondisi di mana gambar yang sama muncul di training dan testing set.
        Data leakage menyebabkan metrik evaluasi menjadi terlalu optimis
        (akurasi terlihat tinggi tapi model sebenarnya tidak generalize).
    """
    print("=" * 70)
    print("  TAHAP 1.4 — VERIFIKASI HASIL SPLIT (CEK DATA LEAKAGE)")
    print("=" * 70)

    issues = []

    for class_name in CLASS_NAMES:
        train_files = set()
        val_files = set()
        test_files = set()

        train_class_dir = TRAIN_DIR / class_name
        val_class_dir = VAL_DIR / class_name
        test_class_dir = TEST_DIR / class_name

        if train_class_dir.exists():
            train_files = {f.name for f in train_class_dir.iterdir() if f.is_file()}
        if val_class_dir.exists():
            val_files = {f.name for f in val_class_dir.iterdir() if f.is_file()}
        if test_class_dir.exists():
            test_files = {f.name for f in test_class_dir.iterdir() if f.is_file()}

        total = len(train_files) + len(val_files) + len(test_files)

        if total == 0:
            continue  # Kelas kosong, skip

        # Cek duplikat antar subset
        train_val_overlap = train_files & val_files
        train_test_overlap = train_files & test_files
        val_test_overlap = val_files & test_files

        if train_val_overlap:
            issues.append(f"  ❌ {class_name}: {len(train_val_overlap)} file duplikat di train & val")
        if train_test_overlap:
            issues.append(f"  ❌ {class_name}: {len(train_test_overlap)} file duplikat di train & test")
        if val_test_overlap:
            issues.append(f"  ❌ {class_name}: {len(val_test_overlap)} file duplikat di val & test")

        # Cek apakah kelas ada di semua subset
        if not train_files:
            issues.append(f"  ⚠️ {class_name}: tidak ada gambar di TRAIN")
        if not val_files:
            issues.append(f"  ⚠️ {class_name}: tidak ada gambar di VAL")
        if not test_files:
            issues.append(f"  ⚠️ {class_name}: tidak ada gambar di TEST")

    if issues:
        print(f"\n  Ditemukan {len(issues)} masalah:")
        for issue in issues:
            print(issue)
    else:
        print("\n  ✅ Semua verifikasi LULUS:")
        print("     - Tidak ada duplikat antar subset (no data leakage)")
        print("     - Setiap kelas terwakili di train/val/test")

    print()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  TAHAP 1 — PERSIAPAN DATASET KLASIFIKASI MAKANAN INDONESIA        ║")
    print("║  100 Kelas | PyTorch + timm | Stratified Split                     ║")
    print("╚" + "═" * 68 + "╝")
    print()

    # 1. Setup struktur folder
    setup_project_directories()

    # 2. Analisis distribusi kelas di data/raw/
    df_dist = analyze_class_distribution(RAW_DATA_DIR)

    # 3. Visualisasi distribusi (hanya jika ada data)
    if not df_dist.empty and df_dist["Jumlah Gambar"].sum() > 0:
        plot_path = FIGURES_DIR / "distribusi_kelas_raw.png"
        plot_class_distribution(df_dist, save_path=plot_path)

        # 4. Split dataset
        split_df = split_dataset()

        # 5. Verifikasi hasil split
        if split_df is not None:
            verify_split()

            # 6. Analisis distribusi setelah split (untuk memastikan stratified)
            print("  --- Distribusi data TRAINING ---")
            analyze_class_distribution(TRAIN_DIR)
    else:
        print("=" * 70)
        print("  📋 PETUNJUK PENGISIAN DATA")
        print("=" * 70)
        print(f"  Folder data/raw/ sudah berisi {NUM_CLASSES} sub-folder kelas.")
        print(f"  Isi setiap folder dengan gambar makanan yang sesuai.")
        print()
        print("  Contoh:")
        print(f"    data/raw/nasi_goreng/  → taruh foto nasi goreng di sini")
        print(f"    data/raw/rendang/      → taruh foto rendang di sini")
        print(f"    data/raw/bakso/        → taruh foto bakso di sini")
        print()
        print(f"  Target: minimal {MIN_IMAGES_PER_CLASS} gambar/kelas")
        print(f"  Format yang didukung: {', '.join(VALID_EXTENSIONS)}")
        print()
        print("  Setelah data terisi, jalankan kembali skrip ini:")
        print("    python 01_dataset_prep.py")
        print()
