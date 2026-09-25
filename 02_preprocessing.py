# ==============================================================================
# 02_preprocessing.py — PREPROCESSING & AUGMENTASI DATA (TAHAP 2)
# ==============================================================================
# Skrip ini membangun pipeline data untuk PyTorch, meliputi:
#   1. Custom Dataset class yang membaca gambar dari folder per kelas
#   2. Transform pipeline per arsitektur (beda input size & normalisasi)
#   3. Augmentasi data standar (flip, rotasi, color jitter)
#   4. Augmentasi agresif (RandAugment, CutOut, MixUp, CutMix) untuk kelas
#      dengan data terbatas — penting karena 100 kelas rawan imbalance
#   5. DataLoader dengan class weighting untuk menangani imbalance
#   6. Visualisasi contoh augmentasi (untuk lampiran laporan Bab IV)
#
# Cara pakai:
#   - File ini di-import oleh skrip training (04_model.py, 05_training.py)
#   - Jika dijalankan langsung (python 02_preprocessing.py), akan menampilkan
#     contoh gambar augmentasi dan statistik dataset
#
# Catatan untuk sidang:
#   - Augmentasi data meningkatkan variasi training data tanpa menambah gambar
#     baru, sehingga membantu model generalize lebih baik (mengurangi overfitting)
#   - RandAugment (Cubuk et al., 2020) adalah metode augmentasi SOTA yang
#     menerapkan N transformasi acak dengan magnitude M — lebih efektif dan
#     sederhana dibanding AutoAugment
#   - MixUp & CutMix mencampur 2 gambar menjadi 1 sample training baru,
#     terbukti meningkatkan robustness dan kalibrasi model pada dataset besar
# ==============================================================================

import os
import sys
import random
import math
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Coba import timm.data untuk augmentasi lanjutan
# timm menyediakan Mixup, RandAugment yang sudah terintegrasi
try:
    from timm.data import create_transform
    from timm.data.mixup import Mixup
    from timm.data.auto_augment import rand_augment_transform
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARNING] timm belum terinstall. Install: pip install timm>=1.0.0")

# Import konfigurasi global
from config import (
    BASE_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, FIGURES_DIR,
    VALID_EXTENSIONS, MIN_IMAGES_PER_CLASS,
    CLASS_NAMES, NUM_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS,
    IMG_SIZE, BATCH_SIZE, NUM_WORKERS, RANDOM_SEED,
    MODEL_CONFIGS, CATEGORY_MAP, CLASS_TO_CATEGORY,
)


# ==============================================================================
# 1. KONSTANTA NORMALISASI
# ==============================================================================
# ImageNet mean & std — digunakan karena semua 3 arsitektur kita pre-trained
# di ImageNet. Input gambar HARUS dinormalisasi dengan nilai yang sama agar
# fitur yang dipelajari model tetap valid saat fine-tuning.
#
# Catatan untuk sidang:
#   Nilai ini bukan "angka ajaib" — ini adalah rata-rata dan standar deviasi
#   pixel RGB dari 1.28 juta gambar ImageNet. Semua model yang pre-trained di
#   ImageNet mengharapkan input yang dinormalisasi dengan nilai ini.
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Mean per channel (R, G, B)
IMAGENET_STD = [0.229, 0.224, 0.225]   # Std per channel (R, G, B)


# ==============================================================================
# 2. CUSTOM DATASET CLASS
# ==============================================================================
class IndonesianFoodDataset(Dataset):
    """
    Custom PyTorch Dataset untuk dataset makanan Indonesia.

    Membaca gambar dari struktur folder:
        data/{split}/{nama_kelas}/gambar.jpg

    Setiap sub-folder menjadi satu kelas, dengan indeks sesuai CLASS_NAMES
    di config.py (bukan urutan alfabet otomatis — ini penting agar indeks
    konsisten antara training dan inferensi).

    Parameter:
        root_dir: Path ke folder split (misal: data/train/)
        transform: Pipeline transformasi gambar (augmentasi + normalisasi)
        class_names: List nama kelas (default dari config.py)

    Catatan untuk sidang:
        Kita membuat Dataset class sendiri (bukan pakai ImageFolder dari
        torchvision) karena:
        1. ImageFolder mengurutkan kelas secara alfabet — indeks tidak konsisten
           jika ada kelas yang ditambah/dihapus
        2. Kita perlu kontrol penuh terhadap mapping kelas → indeks sesuai
           config.py agar konsisten di training, evaluasi, dan deployment mobile
    """

    def __init__(
        self,
        root_dir: Path,
        transform: transforms.Compose = None,
        class_names: List[str] = CLASS_NAMES
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names

        # Mapping nama kelas → indeks (dari config.py)
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Kumpulkan semua path gambar dan labelnya
        self.samples = []  # List of (path_gambar, indeks_kelas)
        self.class_counts = {}  # {nama_kelas: jumlah_gambar}

        for class_name in class_names:
            class_folder = self.root_dir / class_name
            if not class_folder.exists():
                self.class_counts[class_name] = 0
                continue

            # Cari semua file gambar valid di folder kelas ini
            images = sorted([
                f for f in class_folder.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ])

            self.class_counts[class_name] = len(images)
            class_idx = self.class_to_idx[class_name]

            for img_path in images:
                self.samples.append((img_path, class_idx))

        # Hitung total dan statistik
        self.num_samples = len(self.samples)
        self.num_classes_with_data = sum(
            1 for count in self.class_counts.values() if count > 0
        )

    def __len__(self) -> int:
        """Jumlah total sample di dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Mengambil satu sample (gambar + label) berdasarkan indeks.

        Proses:
        1. Baca gambar dari disk menggunakan PIL
        2. Konversi ke RGB (menangani gambar RGBA, grayscale, dll.)
        3. Terapkan transformasi (resize, augmentasi, normalisasi)
        4. Return tensor gambar dan indeks kelas

        Parameter:
            idx: Indeks sample (0 sampai len(dataset)-1)

        Return:
            Tuple (image_tensor, class_index)
            - image_tensor: torch.Tensor shape [3, H, W] (channel-first)
            - class_index: int (0-99)
        """
        img_path, class_idx = self.samples[idx]

        # Baca gambar sebagai PIL Image
        # .convert("RGB") penting karena:
        # - Beberapa gambar PNG punya 4 channel (RGBA) — alpha channel harus dihapus
        # - Beberapa gambar bisa grayscale (1 channel) — perlu dikonversi ke 3 channel
        # - Model CNN mengharapkan input 3 channel (RGB)
        image = Image.open(img_path).convert("RGB")

        # Terapkan transformasi jika ada
        if self.transform:
            image = self.transform(image)

        return image, class_idx

    def get_class_weights(self) -> torch.Tensor:
        """
        Menghitung weight per kelas untuk menangani CLASS IMBALANCE.

        Formula: weight[i] = total_samples / (num_classes * count[i])

        Kelas dengan lebih sedikit gambar mendapat weight lebih tinggi sehingga
        loss-nya lebih berpengaruh saat training. Ini mencegah model bias ke
        kelas mayoritas.

        Catatan untuk sidang:
            Class weighting adalah salah satu teknik standar untuk menangani
            class imbalance. Alternatif lain: oversampling (WeightedRandomSampler),
            focal loss, atau augmentasi khusus kelas minoritas.

        Return:
            torch.Tensor shape [NUM_CLASSES] berisi weight per kelas
        """
        counts = []
        for class_name in self.class_names:
            count = self.class_counts.get(class_name, 0)
            counts.append(max(count, 1))  # Hindari division by zero

        counts = np.array(counts, dtype=np.float64)
        total = counts.sum()
        num_classes = len(counts)

        # Inverse frequency weighting
        weights = total / (num_classes * counts)

        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> List[float]:
        """
        Menghitung weight per SAMPLE (bukan per kelas) untuk WeightedRandomSampler.

        Setiap sample mendapat weight = weight kelas yang bersangkutan.
        Digunakan oleh WeightedRandomSampler agar saat training, kelas minoritas
        lebih sering di-sample sehingga model melihat proporsi kelas yang
        lebih seimbang.

        Return:
            List[float] dengan panjang = jumlah sample
        """
        class_weights = self.get_class_weights().numpy()
        sample_weights = [class_weights[class_idx] for _, class_idx in self.samples]
        return sample_weights

    def get_stats(self) -> Dict:
        """Mengembalikan statistik ringkas dataset."""
        counts = list(self.class_counts.values())
        non_zero = [c for c in counts if c > 0]

        return {
            "total_samples": self.num_samples,
            "total_classes": len(self.class_names),
            "classes_with_data": self.num_classes_with_data,
            "classes_empty": len(self.class_names) - self.num_classes_with_data,
            "min_per_class": min(non_zero) if non_zero else 0,
            "max_per_class": max(non_zero) if non_zero else 0,
            "mean_per_class": np.mean(non_zero) if non_zero else 0,
            "std_per_class": np.std(non_zero) if non_zero else 0,
        }


# ==============================================================================
# 3. TRANSFORM PIPELINES (AUGMENTASI + NORMALISASI)
# ==============================================================================

def get_train_transforms(img_size: int = IMG_SIZE, use_randaugment: bool = True) -> transforms.Compose:
    """
    Membuat pipeline transformasi untuk DATA TRAINING.

    Pipeline ini menerapkan augmentasi data acak setiap kali gambar di-load,
    sehingga model melihat variasi berbeda di setiap epoch — ini yang membuat
    augmentasi efektif (bukan augmentasi offline yang menghasilkan file baru).

    Urutan transformasi:
    1. RandomResizedCrop  — crop acak + resize ke target size
    2. RandomHorizontalFlip — flip horizontal 50% kemungkinan
    3. RandAugment (opsional) — N transformasi acak (rotasi, brightness, dll.)
    4. ToTensor — konversi PIL Image → torch.Tensor, skala [0, 255] → [0.0, 1.0]
    5. Normalize — normalisasi dengan ImageNet mean & std
    6. RandomErasing (CutOut) — hapus area acak di gambar → memaksa model
       tidak bergantung pada satu fitur lokal saja

    Parameter:
        img_size: Ukuran target (lebar = tinggi = img_size)
        use_randaugment: Gunakan RandAugment (lebih agresif, SOTA)

    Catatan untuk sidang:
        - RandomResizedCrop lebih baik dari Resize+RandomCrop karena juga
          menerapkan scale augmentation (crop 8-100% area asli)
        - RandAugment (Cubuk et al., NeurIPS 2020) menggantikan AutoAugment
          yang butuh search mahal — RandAugment hanya punya 2 hyperparameter:
          N (jumlah transform) dan M (magnitude), jauh lebih praktis
    """
    transform_list = [
        # RandomResizedCrop: crop area acak (8%-100% dari gambar asli),
        # dengan rasio aspek 3/4 hingga 4/3, lalu resize ke img_size×img_size.
        # Ini menggabungkan cropping, scaling, dan resizing dalam satu langkah.
        # Interpolation BICUBIC menghasilkan kualitas lebih baik dari BILINEAR.
        transforms.RandomResizedCrop(
            size=img_size,
            scale=(0.08, 1.0),      # Crop antara 8% hingga 100% area
            ratio=(3./4., 4./3.),   # Rasio aspek crop
            interpolation=transforms.InterpolationMode.BICUBIC
        ),

        # Flip horizontal: 50% kemungkinan gambar dibalik kiri-kanan.
        # Cocok untuk makanan karena orientasi horizontal tidak mengubah identitas
        # (nasi goreng tetap nasi goreng meski di-flip).
        # TIDAK pakai flip vertikal karena makanan jarang difoto terbalik.
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    # RandAugment: menerapkan N transformasi acak dari pool standar
    # (rotasi, shear, translate, brightness, contrast, dll.) dengan magnitude M.
    # N=2, M=9 adalah setting yang umum dipakai di riset (paper asli merekomendasikan
    # N=2-3 dan M=5-15 tergantung dataset).
    if use_randaugment and TIMM_AVAILABLE:
        # Format: "rand-m{magnitude}-n{num_ops}-mstd{std}"
        # m9 = magnitude 9, n2 = 2 operasi per gambar, mstd0.5 = std variasi magnitude
        transform_list.append(
            rand_augment_transform(
                config_str="rand-m9-n2-mstd0.5",
                hparams={"translate_const": int(img_size * 0.45)}
            )
        )
    elif not use_randaugment:
        # Fallback: augmentasi manual jika RandAugment dinonaktifkan
        transform_list.extend([
            # Rotasi acak ±15 derajat
            transforms.RandomRotation(degrees=15),

            # Color jitter: ubah brightness, contrast, saturation, hue secara acak
            # Ini mensimulasikan kondisi pencahayaan berbeda saat foto makanan
            transforms.ColorJitter(
                brightness=0.3,   # ±30% brightness
                contrast=0.3,     # ±30% contrast
                saturation=0.3,   # ±30% saturation
                hue=0.1           # ±10% hue shift
            ),

            # Perspective transform: mensimulasikan sudut foto yang berbeda
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        ])

    # Konversi PIL Image ke PyTorch Tensor
    # Mengubah: PIL Image (H, W, C) uint8 [0-255] → Tensor (C, H, W) float32 [0-1]
    transform_list.append(transforms.ToTensor())

    # Normalisasi dengan ImageNet mean & std
    # Mengubah: [0, 1] → [~-2.1, ~2.6] per channel
    # WAJIB karena model pre-trained mengharapkan distribusi input ini
    transform_list.append(
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    )

    # Random Erasing (CutOut variant)
    # Menghapus area persegi panjang acak di gambar dan mengisinya dengan
    # nilai acak. Memaksa model untuk tidak bergantung pada satu region saja
    # — meningkatkan robustness terutama untuk kelas yang punya fitur dominan
    # (misal: warna kuah kuning di soto bisa jadi "shortcut" yang mudah).
    transform_list.append(
        transforms.RandomErasing(
            p=0.25,             # 25% gambar akan di-erase
            scale=(0.02, 0.33), # Area yang dihapus: 2%-33% dari gambar
            ratio=(0.3, 3.3),   # Rasio aspek area yang dihapus
            value="random"      # Isi dengan nilai pixel acak
        )
    )

    return transforms.Compose(transform_list)


def get_val_test_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    Membuat pipeline transformasi untuk DATA VALIDASI dan TEST.

    TIDAK ADA augmentasi — hanya resize dan normalisasi.
    Data validasi/test harus dalam kondisi "asli" agar evaluasi objektif.
    Jika kita augmentasi data test, metrik yang dilaporkan tidak mencerminkan
    performa sebenarnya di dunia nyata.

    Urutan:
    1. Resize — perbesar gambar ke img_size + 10% margin (untuk center crop)
    2. CenterCrop — ambil bagian tengah sebesar img_size×img_size
    3. ToTensor
    4. Normalize

    Parameter:
        img_size: Ukuran target (sama dengan training)

    Catatan untuk sidang:
        Resize ke ukuran sedikit lebih besar lalu CenterCrop adalah teknik
        standar yang menghindari distorsi. Jika langsung Resize ke img_size,
        gambar non-square akan ditarik/ditekan (distorted).
    """
    # crop_pct = 0.95 berarti resize ke img_size/0.95, lalu crop tengah img_size
    # Ini adalah default yang dipakai timm dan banyak paper SOTA
    resize_size = int(img_size / 0.95)

    return transforms.Compose([
        # Resize ke ukuran sedikit lebih besar dari target
        transforms.Resize(
            size=resize_size,
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        # Ambil area tengah (center crop) sebesar img_size × img_size
        transforms.CenterCrop(img_size),
        # Konversi ke tensor
        transforms.ToTensor(),
        # Normalisasi ImageNet
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_transforms_for_model(model_key: str, split: str = "train") -> transforms.Compose:
    """
    Membuat transform pipeline yang sesuai dengan arsitektur model tertentu.
    Setiap arsitektur punya input size default yang berbeda.

    Parameter:
        model_key: Key dari MODEL_CONFIGS di config.py
                   ("efficientnetv2_s", "convnextv2_nano", "mobilenetv4_conv_medium")
        split: "train", "val", atau "test"

    Return:
        transforms.Compose yang siap dipakai di Dataset
    """
    # Ambil input size dari konfigurasi model
    model_cfg = MODEL_CONFIGS.get(model_key, {})
    img_size = model_cfg.get("input_size", IMG_SIZE)

    if split == "train":
        return get_train_transforms(img_size=img_size, use_randaugment=True)
    else:
        return get_val_test_transforms(img_size=img_size)


# ==============================================================================
# 4. MIXUP & CUTMIX (AUGMENTASI LEVEL BATCH)
# ==============================================================================

class MixUpCutMix:
    """
    Wrapper untuk MixUp dan CutMix — augmentasi yang bekerja di level BATCH,
    bukan level gambar individual.

    MixUp (Zhang et al., 2018):
        Mencampur 2 gambar: x_new = λ*x1 + (1-λ)*x2
        Label juga dicampur: y_new = λ*y1 + (1-λ)*y2
        → Membuat decision boundary model lebih smooth

    CutMix (Yun et al., 2019):
        Memotong area dari gambar lain dan menempelkan ke gambar target.
        Label dicampur proporsional dengan area yang dipotong.
        → Lebih baik dari CutOut karena area yang dihapus diganti dengan
          informasi berguna (bukan noise)

    Catatan untuk sidang:
        MixUp+CutMix sudah menjadi teknik augmentasi standar di semua riset
        SOTA (EfficientNetV2, ConvNeXt, dll. semuanya ditraining dengan ini).
        Teknik ini terbukti meningkatkan akurasi 1-3% pada dataset besar.

    Parameter:
        mixup_alpha: Parameter alpha untuk distribusi Beta (MixUp). Semakin
                     tinggi, semakin banyak pencampuran. Default 0.8.
        cutmix_alpha: Parameter alpha untuk CutMix. Default 1.0.
        prob: Probabilitas menerapkan MixUp/CutMix per batch. Default 1.0.
        switch_prob: Probabilitas memilih CutMix vs MixUp. Default 0.5.
        num_classes: Jumlah kelas (untuk membuat soft label)
    """

    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        num_classes: int = NUM_CLASSES
    ):
        if TIMM_AVAILABLE:
            # Gunakan implementasi timm yang sudah teruji
            self.mixup_fn = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=prob,
                switch_prob=switch_prob,
                num_classes=num_classes,
                mode="batch",      # Terapkan ke seluruh batch sekaligus
                label_smoothing=0.1  # Smooth label 0→0.1, 1→0.9
            )
        else:
            self.mixup_fn = None
            print("[WARNING] timm tidak tersedia, MixUp/CutMix dinonaktifkan")

    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Terapkan MixUp atau CutMix ke batch gambar.

        Parameter:
            images: Tensor shape [B, C, H, W]
            targets: Tensor shape [B] (hard labels, indeks kelas)

        Return:
            Tuple (mixed_images, mixed_targets)
            - mixed_images: Tensor shape [B, C, H, W]
            - mixed_targets: Tensor shape [B, NUM_CLASSES] (soft labels)
        """
        if self.mixup_fn is not None:
            return self.mixup_fn(images, targets)
        return images, targets


# ==============================================================================
# 5. DATALOADER FACTORY
# ==============================================================================

def create_dataloaders(
    model_key: str = "efficientnetv2_s",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    use_weighted_sampler: bool = True,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Membuat DataLoader untuk train, val, dan test dataset.

    DataLoader mengatur:
    - Batching: mengelompokkan sample menjadi batch
    - Shuffling: mengacak urutan sample setiap epoch (hanya train)
    - Parallel loading: memuat data di background thread (num_workers)
    - Pin memory: mempercepat transfer CPU→GPU

    Parameter:
        model_key: Key arsitektur dari MODEL_CONFIGS (menentukan input size)
        batch_size: Jumlah gambar per batch
        num_workers: Jumlah worker thread untuk loading data paralel
        use_weighted_sampler: Gunakan WeightedRandomSampler untuk class balancing
        pin_memory: Pin memory di RAM untuk transfer GPU lebih cepat

    Return:
        Dict {"train": DataLoader, "val": DataLoader, "test": DataLoader}

    Catatan untuk sidang:
        WeightedRandomSampler memberikan probabilitas sampling lebih tinggi
        untuk kelas minoritas. Ini BERBEDA dari class weighting di loss function:
        - WeightedRandomSampler → mengubah FREKUENSI kemunculan kelas saat training
        - Class weighting → mengubah BOBOT loss per kelas
        Keduanya bisa dipakai bersamaan, tapi hati-hati karena bisa over-correct.
        Di sini kita pakai WeightedRandomSampler saja (sudah cukup efektif).
    """
    # Buat transform sesuai arsitektur
    train_transform = get_transforms_for_model(model_key, split="train")
    val_test_transform = get_transforms_for_model(model_key, split="val")

    # Buat dataset objects
    train_dataset = IndonesianFoodDataset(
        root_dir=TRAIN_DIR,
        transform=train_transform
    )
    val_dataset = IndonesianFoodDataset(
        root_dir=VAL_DIR,
        transform=val_test_transform
    )
    test_dataset = IndonesianFoodDataset(
        root_dir=TEST_DIR,
        transform=val_test_transform
    )

    # Print statistik dataset
    for name, ds in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        stats = ds.get_stats()
        print(f"  {name:5s}: {stats['total_samples']:>6,} gambar | "
              f"{stats['classes_with_data']:>3} kelas berisi data | "
              f"min {stats['min_per_class']} / max {stats['max_per_class']} per kelas")

    # Buat sampler untuk training (menangani class imbalance)
    train_sampler = None
    train_shuffle = True

    if use_weighted_sampler and train_dataset.num_samples > 0:
        # WeightedRandomSampler: setiap sample punya weight proporsional
        # terhadap inverse frekuensi kelasnya. Kelas dengan sedikit gambar
        # akan lebih sering di-sample → distribusi training lebih seimbang.
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),  # Sample sebanyak dataset asli per epoch
            replacement=True  # Boleh sample ulang (karena kelas minoritas perlu di-sample berkali-kali)
        )
        train_shuffle = False  # Sampler dan shuffle tidak bisa bersamaan

    # Buat DataLoader
    # persistent_workers=True menjaga worker thread tetap hidup antar epoch
    # (mengurangi overhead spawn worker baru) — hanya berlaku jika num_workers>0
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=True,  # Drop batch terakhir jika tidak penuh (stabilitas training)
            persistent_workers=num_workers > 0,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # JANGAN shuffle val/test — urutan harus konsisten untuk evaluasi
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=num_workers > 0,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=num_workers > 0,
        ),
    }

    return dataloaders


# ==============================================================================
# 6. VISUALISASI CONTOH AUGMENTASI
# ==============================================================================

def visualize_augmentations(
    dataset_dir: Path = TRAIN_DIR,
    class_name: str = "nasi_goreng",
    model_key: str = "efficientnetv2_s",
    num_augmented: int = 8,
    save_path: Path = None
):
    """
    Menampilkan 1 gambar asli beserta beberapa variasi augmentasinya.
    Berguna untuk lampiran laporan dan untuk memverifikasi bahwa augmentasi
    tidak terlalu destruktif (gambar masih recognizable).

    Parameter:
        dataset_dir: Path ke folder dataset (misal: data/train/)
        class_name: Nama kelas yang akan divisualisasi
        model_key: Key arsitektur (menentukan input size)
        num_augmented: Jumlah variasi augmentasi yang ditampilkan
        save_path: Path untuk menyimpan gambar (opsional)

    Catatan untuk sidang:
        Visualisasi augmentasi penting untuk membuktikan bahwa:
        1. Augmentasi menghasilkan variasi yang realistis
        2. Gambar hasil augmentasi masih bisa dikenali sebagai kelas aslinya
        3. Augmentasi tidak terlalu lemah (tidak efektif) atau terlalu kuat
           (merusak informasi penting)
    """
    class_folder = dataset_dir / class_name
    if not class_folder.exists():
        print(f"  [ERROR] Folder {class_folder} tidak ditemukan.")
        return

    # Cari gambar pertama di folder kelas
    images = sorted([
        f for f in class_folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ])

    if not images:
        print(f"  [ERROR] Tidak ada gambar di folder {class_name}/")
        return

    # Ambil gambar pertama
    img_path = images[0]
    original_img = Image.open(img_path).convert("RGB")

    # Buat transform training (dengan augmentasi)
    model_cfg = MODEL_CONFIGS.get(model_key, {})
    img_size = model_cfg.get("input_size", IMG_SIZE)
    train_transform = get_train_transforms(img_size=img_size)

    # Transform validasi (tanpa augmentasi) untuk gambar "asli"
    val_transform = get_val_test_transforms(img_size=img_size)

    # Generate variasi augmentasi
    fig, axes = plt.subplots(2, (num_augmented + 2) // 2, figsize=(16, 7))
    axes = axes.flatten()

    # Gambar pertama: original (dengan val transform — resize saja)
    original_tensor = val_transform(original_img)
    original_display = denormalize_tensor(original_tensor)
    axes[0].imshow(original_display)
    axes[0].set_title("ORIGINAL", fontsize=10, fontweight="bold", color="green")
    axes[0].axis("off")

    # Gambar augmentasi
    for i in range(1, min(num_augmented + 1, len(axes))):
        augmented_tensor = train_transform(original_img)
        augmented_display = denormalize_tensor(augmented_tensor)
        axes[i].imshow(augmented_display)
        axes[i].set_title(f"Augmentasi #{i}", fontsize=9)
        axes[i].axis("off")

    # Sembunyikan axes yang tidak terpakai
    for i in range(num_augmented + 1, len(axes)):
        axes[i].axis("off")

    display_name = class_name.replace("_", " ").title()
    fig.suptitle(
        f'Contoh Augmentasi Data: "{display_name}" (Input Size: {img_size}×{img_size})',
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  [OK] Visualisasi augmentasi disimpan: {save_path}")

    plt.close()


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Mengembalikan tensor yang sudah dinormalisasi ke rentang [0, 1]
    agar bisa ditampilkan dengan matplotlib.

    Proses: pixel = pixel * std + mean (kebalikan dari normalisasi)

    Parameter:
        tensor: Tensor shape [C, H, W] yang sudah dinormalisasi

    Return:
        numpy array shape [H, W, C] dengan nilai [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    # Denormalize: x_original = x_normalized * std + mean
    img = tensor.clone().cpu()
    img = img * std + mean
    # Clip ke [0, 1] untuk keamanan (augmentasi kadang menghasilkan nilai outlier)
    img = torch.clamp(img, 0, 1)
    # Transpose dari (C, H, W) → (H, W, C) untuk matplotlib
    return img.permute(1, 2, 0).numpy()


# ==============================================================================
# 7. UTILITAS: COMPUTE DATASET MEAN & STD
# ==============================================================================

def compute_dataset_mean_std(
    dataset_dir: Path = TRAIN_DIR,
    img_size: int = IMG_SIZE,
    num_samples: int = 1000
) -> Tuple[List[float], List[float]]:
    """
    Menghitung mean dan std dataset kita sendiri (bukan ImageNet).

    Berguna jika ingin membandingkan apakah distribusi pixel dataset makanan
    Indonesia berbeda signifikan dari ImageNet. Jika sangat berbeda,
    MUNGKIN perlu normalisasi custom — tapi dalam praktik, menggunakan
    ImageNet mean/std sudah cukup baik untuk transfer learning.

    Parameter:
        dataset_dir: Path ke folder training data
        img_size: Ukuran resize gambar
        num_samples: Jumlah sample acak untuk estimasi (1000 sudah cukup)

    Return:
        Tuple (mean_per_channel, std_per_channel)

    Catatan untuk sidang:
        Ini bisa jadi bahan analisis menarik di Bab IV — jika distribusi pixel
        dataset kita mirip ImageNet, itu menjelaskan mengapa transfer learning
        efektif. Jika berbeda jauh, itu argumen untuk fine-tuning lebih banyak layer.
    """
    simple_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # Konversi ke [0, 1] tanpa normalisasi
    ])

    dataset = IndonesianFoodDataset(
        root_dir=dataset_dir,
        transform=simple_transform
    )

    if len(dataset) == 0:
        print("  [WARNING] Dataset kosong, mengembalikan ImageNet mean/std")
        return IMAGENET_MEAN, IMAGENET_STD

    # Ambil subset acak untuk efisiensi
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # Akumulasi statistik per channel menggunakan Welford's online algorithm
    # (numerik stabil, tidak perlu simpan semua data di memori)
    channels_sum = torch.zeros(3)
    channels_sq_sum = torch.zeros(3)
    num_pixels = 0

    for idx in indices:
        img, _ = dataset[idx]  # img shape: [3, H, W]
        channels_sum += img.sum(dim=[1, 2])      # Sum per channel
        channels_sq_sum += (img ** 2).sum(dim=[1, 2])  # Sum of squares per channel
        num_pixels += img.shape[1] * img.shape[2]  # H * W

    # Hitung mean dan std
    mean = channels_sum / num_pixels
    std = torch.sqrt(channels_sq_sum / num_pixels - mean ** 2)

    print(f"\n  Dataset Mean (RGB): [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  Dataset Std  (RGB): [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print(f"  ImageNet Mean (RGB): {IMAGENET_MEAN}")
    print(f"  ImageNet Std  (RGB): {IMAGENET_STD}")

    return mean.tolist(), std.tolist()


# ==============================================================================
# MAIN EXECUTION — Demo & Verifikasi
# ==============================================================================
if __name__ == "__main__":
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  TAHAP 2 — PREPROCESSING & AUGMENTASI DATA                        ║")
    print("║  Pipeline: RandAugment + MixUp/CutMix + WeightedRandomSampler     ║")
    print("╚" + "═" * 68 + "╝")
    print()

    # --- Cek apakah data sudah ada ---
    train_exists = TRAIN_DIR.exists() and any(TRAIN_DIR.glob("*/*.jpg")) or any(TRAIN_DIR.glob("*/*.png"))

    if not train_exists:
        print("  ⚠️ Data training belum ada di data/train/")
        print("  Jalankan 01_dataset_prep.py terlebih dahulu setelah mengisi data/raw/")
        print()
        print("  Meskipun begitu, pipeline preprocessing sudah siap digunakan.")
        print("  Berikut contoh penggunaan di kode training nanti:")
        print()
        print('  from preprocessing_02 import create_dataloaders, MixUpCutMix')
        print()
        print('  # Buat DataLoader untuk EfficientNetV2-S')
        print('  dataloaders = create_dataloaders(model_key="efficientnetv2_s")')
        print('  train_loader = dataloaders["train"]')
        print()
        print('  # Buat MixUp/CutMix augmenter')
        print('  mixup = MixUpCutMix()')
        print()
        print('  # Training loop')
        print('  for images, labels in train_loader:')
        print('      images, labels = mixup(images, labels)  # Apply MixUp/CutMix')
        print('      outputs = model(images)')
        print('      loss = criterion(outputs, labels)')
        print()
        sys.exit(0)

    # --- Jika data sudah ada, jalankan demo ---
    print("=" * 70)
    print("  DEMO: Membuat DataLoader untuk ketiga arsitektur")
    print("=" * 70)
    print()

    for model_key, model_cfg in MODEL_CONFIGS.items():
        print(f"  📦 {model_cfg['paper_name']} (input: {model_cfg['input_size']}×{model_cfg['input_size']})")
        dataloaders = create_dataloaders(
            model_key=model_key,
            batch_size=BATCH_SIZE
        )
        print(f"     Train batches: {len(dataloaders['train'])}")
        print(f"     Val batches:   {len(dataloaders['val'])}")
        print(f"     Test batches:  {len(dataloaders['test'])}")
        print()

    # --- Visualisasi augmentasi ---
    print("=" * 70)
    print("  DEMO: Visualisasi Contoh Augmentasi")
    print("=" * 70)

    # Cari kelas yang punya data untuk divisualisasi
    for class_name in CLASS_NAMES:
        class_folder = TRAIN_DIR / class_name
        if class_folder.exists() and any(class_folder.glob("*")):
            print(f"\n  Membuat visualisasi augmentasi untuk kelas: {class_name}")
            for model_key, model_cfg in MODEL_CONFIGS.items():
                save_path = FIGURES_DIR / f"augmentasi_{class_name}_{model_key}.png"
                visualize_augmentations(
                    dataset_dir=TRAIN_DIR,
                    class_name=class_name,
                    model_key=model_key,
                    num_augmented=7,
                    save_path=save_path
                )
            break  # Cukup 1 kelas untuk demo
    else:
        print("\n  [SKIP] Tidak ada kelas dengan data untuk divisualisasi.")

    # --- Hitung mean/std dataset sendiri ---
    print()
    print("=" * 70)
    print("  DEMO: Hitung Mean & Std Dataset (vs ImageNet)")
    print("=" * 70)
    compute_dataset_mean_std()

    print()
    print("  ✅ Pipeline preprocessing siap digunakan untuk training!")
    print()
