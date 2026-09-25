# ==============================================================================
# 04_model.py — PEMBANGUNAN ARSITEKTUR SOTA & STRATEGI FINE-TUNING (TAHAP 4)
# ==============================================================================
# Skrip ini mengimplementasikan 3 arsitektur CNN State-of-the-Art (SOTA)
# menggunakan library resmi PyTorch Image Models (timm):
#   1. EfficientNetV2-S (Tan & Le, ICML 2021)
#   2. ConvNeXt V2-Nano (Woo et al., CVPR 2023)
#   3. MobileNetV4-Conv-Medium (Qin et al., ECCV 2024 / Google Research)
#
# Fitur Utama:
#   - Factory pattern untuk membuat model dengan 100 kelas makanan Indonesia
#   - Pilihan classifier head: Standard Linear Head vs Enhanced Multi-Layer Head
#   - Strategi Fine-Tuning 3 Fase:
#       Fase 1: Feature Extraction (Freeze backbone, latih head saja)
#       Fase 2: Gradual Unfreezing (Buka blok layer teratas)
#       Fase 3: Full Fine-Tuning dengan Discriminative Learning Rates (Layer-wise LR)
#   - Profiling model: Jumlah parameter, estimasi ukuran memori (MB), estimasi FLOPs,
#     dan benchmarking latensi inferensi (ms/gambar)
#
# Argumen Ilmiah untuk Sidang Sarjana (S1 Informatika):
#   - Mengapa tidak CNN buatan sendiri (custom CNN sederhana)?
#     CNN custom 3-5 layer tidak memiliki receptive field yang cukup luas dan
#     menderita masalah vanishing gradient jika diperdalam tanpa residual connections.
#     Melatih dari nol (from scratch) pada 100 kelas membutuhkan jutaan gambar agar
#     tidak overfit. Transfer learning dari backbone SOTA memanfaatkan inductive bias
#     dan fitur visual hirarkis (tepi, tekstur, bentuk pangan) dari ImageNet.
#   - Mengapa 3 arsitektur ini dipilih sebagai komparasi?
#     1. EfficientNetV2: Merepresentasikan puncak optimalisasi Neural Architecture Search (NAS)
#        dan Fused-MBConv.
#     2. ConvNeXt V2: Merepresentasikan arsitektur modern pure-convolutional yang mengadopsi
#        filosofi Vision Transformer (7x7 depthwise conv, inverted bottleneck, GRN).
#     3. MobileNetV4: Arsitektur mobile generasi terbaru (2024) dengan Universal Inverted
#        Bottleneck (UIB) yang dirancang khusus untuk hardware-aware latency pada NPU/DSP mobile.
# ==============================================================================

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn

# Coba import timm
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARNING] timm belum terinstall. Silakan jalankan: pip install timm>=1.0.0")

# Import konfigurasi global
from config import (
    BASE_DIR, SAVED_MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
    NUM_CLASSES, MODEL_CONFIGS, LEARNING_RATE
)


# ==============================================================================
# 1. MODULAR CLASSIFIER HEAD
# ==============================================================================
class EnhancedFoodClassifierHead(nn.Module):
    """
    Classifier Head multi-layer yang ditingkatkan untuk klasifikasi 100 kelas makanan.

    Struktur:
        Input Features -> Dropout -> Linear(in_features, hidden_dim)
                       -> LayerNorm -> GELU -> Dropout -> Linear(hidden_dim, num_classes)

    Keunggulan dibanding single linear layer:
        - LayerNorm & GELU memberikan regularisasi non-linear yang lebih ekspresif
          untuk memisahkan 100 kelas kuliner Indonesia yang fine-grained (kemiripan visual tinggi).
        - Dropout ganda (p=0.3 dan p=0.2) mencegah overfitting pada kelas dengan data terbatas.
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 512,
        drop_rate: float = 0.3
    ):
        super().__init__()
        self.drop1 = nn.Dropout(p=drop_rate)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop2 = nn.Dropout(p=drop_rate * 0.7)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x


# ==============================================================================
# 2. INDONESIAN FOOD CLASSIFIER MODEL WRAPPER
# ==============================================================================
class IndonesianFoodClassifier(nn.Module):
    """
    Wrapper terpadu untuk model SOTA Image Classification makanan Indonesia.

    Parameter:
        model_key: Kunci model dalam MODEL_CONFIGS ('efficientnetv2_s', 'convnextv2_nano', 'mobilenetv4_conv_medium')
        num_classes: Jumlah kelas klasifikasi (default 100)
        pretrained: Memuat bobot pre-trained ImageNet (harus True untuk transfer learning)
        drop_rate: Dropout rate pada classifier head
        use_enhanced_head: Jika True gunakan 2-layer MLP head; jika False gunakan standard linear head timm
    """
    def __init__(
        self,
        model_key: str,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        drop_rate: float = 0.3,
        use_enhanced_head: bool = False
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("Library 'timm' belum terinstall. Jalankan: pip install timm>=1.0.0")

        if model_key not in MODEL_CONFIGS:
            raise ValueError(
                f"Model key '{model_key}' tidak ditemukan di MODEL_CONFIGS. "
                f"Pilihan valid: {list(MODEL_CONFIGS.keys())}"
            )

        self.model_key = model_key
        self.config = MODEL_CONFIGS[model_key]
        self.timm_name = self.config["timm_name"]
        self.paper_name = self.config["paper_name"]
        self.num_classes = num_classes
        self.use_enhanced_head = use_enhanced_head

        # ----------------------------------------------------------------------
        # A. Bangun Backbone via timm
        # ----------------------------------------------------------------------
        if not use_enhanced_head:
            # Cara Standar timm: ganti classifier head langsung ke num_classes
            self.backbone = timm.create_model(
                self.timm_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate
            )
            self.custom_head = None
        else:
            # Cara Enhanced Head: buat backbone tanpa classifier (num_classes=0)
            self.backbone = timm.create_model(
                self.timm_name,
                pretrained=pretrained,
                num_classes=0,  # Melepaskan fc/classifier bawaan
                drop_rate=0.0
            )
            # Ambil dimensi output fitur backbone
            in_features = self.backbone.num_features
            self.custom_head = EnhancedFoodClassifierHead(
                in_features=in_features,
                num_classes=num_classes,
                drop_rate=drop_rate
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.custom_head is None:
            return self.backbone(x)
        else:
            features = self.backbone(x)
            return self.custom_head(features)

    def get_input_size(self) -> int:
        """Mengembalikan ukuran resolusi input standar untuk arsitektur ini."""
        return self.config["input_size"]


# ==============================================================================
# 3. STRATEGI FINE-TUNING BERTAHAP (GRADUAL UNFREEZING & DISCRIMINATIVE LR)
# ==============================================================================
def freeze_backbone(model: IndonesianFoodClassifier) -> None:
    """
    FASE 1: FEATURE EXTRACTION (FREEZE BACKBONE)
    Membekukan seluruh parameter backbone, hanya classifier head yang dapat dilatih.

    Tujuan Sidang:
        Mencegah 'catastrophic forgetting' di mana bobot pre-trained ImageNet
        rusak akibat gradien acak besar dari classifier head yang baru diinisialisasi.
    """
    # Freeze semua parameter backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Unfreeze classifier head
    if model.custom_head is not None:
        for param in model.custom_head.parameters():
            param.requires_grad = True
    else:
        # Jika pakai head bawaan timm, cari sub-modul classifier/head
        _unfreeze_timm_classifier(model.backbone)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  🔒 [FASE 1] Backbone DIBEKUKAN. Parameter terlatih: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")


def unfreeze_top_stages(model: IndonesianFoodClassifier, num_stages: int = 1) -> None:
    """
    FASE 2: GRADUAL UNFREEZING (BUKA STAGE TERATAS)
    Membuka blok-blok konvolusi tingkat tinggi (high-level feature extractors).

    Tujuan Sidang:
        Layer awal mempelajari fitur umum (garis, sudut, tekstur).
        Layer akhir mempelajari semantik spesifik objek (tekstur gulai, butiran beras,
        bentuk rendang). Membuka layer akhir bertahap membantu adaptasi domain
        secara halus tanpa merusak representasi visual dasar.
    """
    model_key = model.model_key

    if model_key == "efficientnetv2_s":
        # EfficientNetV2 memiliki modul self.backbone.blocks (berisi 7 stage: 0 sampai 6)
        # Buka 2 stage terakhir (stage 5 dan 6)
        if hasattr(model.backbone, "blocks"):
            num_blocks = len(model.backbone.blocks)
            start_unfreeze = max(0, num_blocks - num_stages - 1)
            for i in range(start_unfreeze, num_blocks):
                for param in model.backbone.blocks[i].parameters():
                    param.requires_grad = True
        if hasattr(model.backbone, "conv_head"):
            for param in model.backbone.conv_head.parameters():
                param.requires_grad = True

    elif model_key == "convnextv2_nano":
        # ConvNeXt V2 memiliki 4 stages (stages 0, 1, 2, 3)
        # Buka stage terakhir (stage 3)
        if hasattr(model.backbone, "stages"):
            num_stages_total = len(model.backbone.stages)
            start_unfreeze = max(0, num_stages_total - num_stages)
            for i in range(start_unfreeze, num_stages_total):
                for param in model.backbone.stages[i].parameters():
                    param.requires_grad = True
        if hasattr(model.backbone, "norm_pre"):
            for param in model.backbone.norm_pre.parameters():
                param.requires_grad = True

    elif model_key == "mobilenetv4_conv_medium":
        # MobileNetV4 memiliki modul blocks dan conv_head
        if hasattr(model.backbone, "blocks"):
            num_blocks = len(model.backbone.blocks)
            start_unfreeze = max(0, num_blocks - num_stages - 1)
            for i in range(start_unfreeze, num_blocks):
                for param in model.backbone.blocks[i].parameters():
                    param.requires_grad = True
        if hasattr(model.backbone, "conv_head"):
            for param in model.backbone.conv_head.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  🔓 [FASE 2] Stage Teratas DIBUKA. Parameter terlatih: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")


def unfreeze_all(model: IndonesianFoodClassifier) -> None:
    """
    FASE 3: FULL FINE-TUNING
    Membuka seluruh parameter model untuk dilatih bersamaan.
    Harus dipadukan dengan learning rate kecil (misal 1e-5) agar bobot stabil.
    """
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  🌐 [FASE 3] Seluruh Layer DIBUKA. Parameter terlatih: {trainable:,} / {total:,} (100%)")


def get_discriminative_param_groups(
    model: IndonesianFoodClassifier,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-4
) -> List[Dict]:
    """
    DISCRIMINATIVE LEARNING RATES (LAYER-WISE LR DECAY)
    Membagi parameter ke dalam dua grup optimasi:
    1. Backbone Parameters: Diberi LR kecil (backbone_lr, misal 1e-5) agar fitur pre-trained
       hanya bergeser sedikit menyesuaikan dataset makanan.
    2. Classifier Head Parameters: Diberi LR lebih besar (head_lr, misal 1e-4) agar
       cepat mempelajari pemetaan 100 kelas baru.

    Format output kompatibel langsung dengan optimizer PyTorch:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    """
    head_params = []
    backbone_params = []

    if model.custom_head is not None:
        head_params = list(model.custom_head.parameters())
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    else:
        # Pisahkan parameter classifier bawaan timm dari backbone
        classifier_names = ["classifier", "head", "fc"]
        for name, param in model.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if any(cls_name in name.lower() for cls_name in classifier_names):
                head_params.append(param)
            else:
                backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
    ]


def _unfreeze_timm_classifier(timm_model: nn.Module) -> None:
    """Helper untuk mengaktifkan gradien pada classifier head timm bawaan."""
    head_found = False
    for attr in ["classifier", "head", "fc"]:
        if hasattr(timm_model, attr):
            submodule = getattr(timm_model, attr)
            for param in submodule.parameters():
                param.requires_grad = True
            head_found = True
            break
    if not head_found:
        # Fallback jika nama modul berbeda: aktifkan layer terakhir
        for param in list(timm_model.parameters())[-4:]:
            param.requires_grad = True


# ==============================================================================
# 4. PROFILING & BENCHMARKING MODEL (PARAMETER, UKURAN, LATENSI)
# ==============================================================================
def profile_model(
    model: IndonesianFoodClassifier,
    device: str = "cpu",
    num_warmup: int = 10,
    num_runs: int = 50
) -> Dict[str, Union[float, int, str]]:
    """
    Mengukur metrik teknis model untuk tabel komparasi Bab IV:
    - Jumlah parameter total, trainable, dan non-trainable
    - Estimasi ukuran bobot FP32 dalam Megabyte (MB)
    - Latensi inferensi rata-rata per gambar (milidetik)
    - Throughput inferensi (gambar per detik / FPS)
    """
    model.eval()
    model.to(device)

    input_size = model.get_input_size()
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    # 1. Hitung Parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weight_size_mb = (total_params * 4) / (1024 * 1024)  # 4 byte per float32

    # 2. Warmup Benchmark
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # 3. Pengukuran Waktu Inferensi
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # Konversi ke ms

    avg_latency_ms = float(torch.tensor(timings).mean().item())
    std_latency_ms = float(torch.tensor(timings).std().item())
    throughput_fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "model_key": model.model_key,
        "paper_name": model.paper_name,
        "input_resolution": f"{input_size}x{input_size}",
        "total_params": total_params,
        "total_params_million": round(total_params / 1e6, 2),
        "trainable_params": trainable_params,
        "weight_size_mb": round(weight_size_mb, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "std_latency_ms": round(std_latency_ms, 2),
        "throughput_fps": round(throughput_fps, 1),
        "device": device
    }


# ==============================================================================
# 5. MODEL FACTORY BUILDER (INTERFACE UTAMA UNTUK TAHAP 5)
# ==============================================================================
def build_model(
    model_key: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    drop_rate: float = 0.3,
    use_enhanced_head: bool = False
) -> IndonesianFoodClassifier:
    """
    Fungsi builder standar untuk menginstansiasi model klasifikasi makanan Indonesia.

    Contoh Pemakaian:
        model = build_model("convnextv2_nano", pretrained=True)
    """
    return IndonesianFoodClassifier(
        model_key=model_key,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
        use_enhanced_head=use_enhanced_head
    )


# ==============================================================================
# MAIN EXECUTION (DEMO & VERIFIKASI TAHAP 4)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("  TAHAP 4: PEMBANGUNAN MODEL SOTA & STRATEGI FINE-TUNING")
    print("=" * 80)
    print(f"  Target Klasifikasi : {NUM_CLASSES} Kelas Makanan Indonesia")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Perangkat Eksekusi : {device.upper()}")
    print("=" * 80)

    if not TIMM_AVAILABLE:
        print("\n  [INFO] timm belum terinstall di environment Python Anda.")
        print("  Install terlebih dahulu via terminal:")
        print("      pip install timm>=1.0.0 torch torchvision")
        sys.exit(0)

    benchmark_results = []

    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"\n[{cfg['paper_name'].upper()}]")
        print(f"  Arsitektur : {cfg['paper_name']} ({cfg['year']})")
        print(f"  timm Model : {cfg['timm_name']}")
        print(f"  Resolusi   : {cfg['input_size']}x{cfg['input_size']}")

        # 1. Instansiasi Model
        print("  -> Membangun model & menginisialisasi 100 output logits...")
        model = build_model(model_key=model_key, num_classes=NUM_CLASSES, pretrained=True)

        # 2. Uji Forward Pass Shape
        dummy_in = torch.randn(2, 3, cfg["input_size"], cfg["input_size"])
        out = model(dummy_in)
        assert out.shape == (2, NUM_CLASSES), f"Output shape salah: {out.shape}"
        print(f"  ✅ Forward Pass Berhasil! Input: {dummy_in.shape} -> Output Logits: {out.shape}")

        # 3. Uji Strategi 3 Fase Unfreezing
        print("  -> Menguji strategi gradual unfreezing...")
        freeze_backbone(model)
        unfreeze_top_stages(model, num_stages=1)
        unfreeze_all(model)

        # 4. Profiling Teknis
        print("  -> Melakukan profiling latensi & memori...")
        metrics = profile_model(model, device=device, num_warmup=3, num_runs=10)
        benchmark_results.append(metrics)
        print(f"     Parameters : {metrics['total_params_million']} Juta")
        print(f"     Ukuran FP32: {metrics['weight_size_mb']} MB")
        print(f"     Latensi    : {metrics['avg_latency_ms']} ms/gambar ({metrics['throughput_fps']} FPS di {device.upper()})")

    # Tampilkan Tabel Rekapitulasi untuk Bab IV
    print("\n" + "=" * 90)
    print("  TABEL KOMPARASI TEKNIS KETIGA ARSITEKTUR SOTA (BAHAN BAB IV SKRIPSI)")
    print("=" * 90)
    print(f"{'Arsitektur':<26} | {'Input Size':<10} | {'Params (M)':<10} | {'Ukuran (MB)':<11} | {'Latensi (ms)':<12} | {'FPS':<6}")
    print("-" * 90)
    for b in benchmark_results:
        print(f"{b['paper_name']:<26} | {b['input_resolution']:<10} | {b['total_params_million']:<10} | {b['weight_size_mb']:<11} | {b['avg_latency_ms']:<12} | {b['throughput_fps']:<6}")
    print("=" * 90)
    print("  ✅ TAHAP 4 SELESAI: Ketiga model SOTA siap dilatih pada Tahap 5!")
