# ==============================================================================
# 05_training.py — PIPELINE TRAINING & HYPERPARAMETER TUNING (TAHAP 5)
# ==============================================================================
# Skrip ini mengimplementasikan loop pelatihan lengkap untuk membandingkan
# model CNN SOTA (ConvNeXt V2-Nano, MobileNetV4, EfficientNetV2) secara
# head-to-head pada 100 kelas makanan Indonesia.
#
# Fitur Utama:
#   1. Pelatihan 3 Fase (Warmup -> Top Stage Unfreeze -> Full Fine-Tuning)
#   2. Loss Function: CrossEntropyLoss dengan Label Smoothing (0.1) untuk
#      mencegah overconfidence pada 100 kelas yang berdekatan
#   3. Mixed Precision Training (torch.cuda.amp) untuk efisiensi VRAM & kecepatan
#   4. Callbacks Lengkap: EarlyStopping, ModelCheckpoint, Learning Rate Scheduler
#   5. Logging ke TensorBoard dan ekspor riwayat metrik (JSON & visualisasi kurva loss/akurasi)
#   6. Mode Dry-Run Otomatis: dapat menguji seluruh alur training bahkan jika data
#      foto mentah belum diunduh (menggunakan synthetic data generator)
# ==============================================================================

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Coba import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[INFO] TensorBoard belum terpasang. Logging metrik akan tetap dicatat ke JSON.")

# Import konfigurasi global dan modul tahap sebelumnya
from config import (
    BASE_DIR, TRAIN_DIR, VAL_DIR, SAVED_MODELS_DIR,
    REPORTS_DIR, FIGURES_DIR, LOGS_DIR,
    NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    MODEL_CONFIGS, RANDOM_SEED
)
import importlib
_prep = importlib.import_module("02_preprocessing")
create_dataloaders = _prep.create_dataloaders

_model = importlib.import_module("04_model")
build_model = _model.build_model
freeze_backbone = _model.freeze_backbone
unfreeze_top_stages = _model.unfreeze_top_stages
unfreeze_all = _model.unfreeze_all
get_discriminative_param_groups = _model.get_discriminative_param_groups


# ==============================================================================
# 1. EARLY STOPPING CALLBACK
# ==============================================================================
class EarlyStopping:
    """
    Menghentikan proses pelatihan lebih awal jika validation loss tidak membaik
    setelah sejumlah epoch (patience), untuk menghindari overfitting dan pemborosan komputasi.
    """
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_metric: float) -> bool:
        score = -val_metric if self.mode == "min" else val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  [EarlyStopping] Peringatan: Tidak ada perbaikan ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


# ==============================================================================
# 2. MODEL CHECKPOINT MANAGER
# ==============================================================================
class ModelCheckpoint:
    """
    Menyimpan bobot model terbaik (.pth) berdasarkan performa validasi (val_acc atau val_loss).
    """
    def __init__(
        self,
        save_dir: Path,
        model_name: str,
        mode: str = "max",
        metric_name: str = "val_acc"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.mode = mode
        self.metric_name = metric_name
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.best_path = self.save_dir / f"best_{model_name}.pth"
        self.latest_path = self.save_dir / f"latest_{model_name}.pth"

    def step(self, current_metric: float, model: nn.Module, epoch: int, optimizer: torch.optim.Optimizer) -> bool:
        improved = False
        if self.mode == "max" and current_metric > self.best_metric:
            improved = True
        elif self.mode == "min" and current_metric < self.best_metric:
            improved = True

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": current_metric,
            "metric_name": self.metric_name
        }

        # Simpan checkpoint terbaru
        torch.save(checkpoint, self.latest_path)

        if improved:
            self.best_metric = current_metric
            torch.save(checkpoint, self.best_path)
            print(f"  ⭐ [Checkpoint] Model Terbaik Disimpan ({self.metric_name}: {current_metric:.4f}) -> {self.best_path.name}")

        return improved


# ==============================================================================
# 3. TRAINING ENGINE KELAS TUNGGAL & MULTI-FASE
# ==============================================================================
class ModelTrainer:
    """
    Engine utama yang mengontrol seluruh siklus pelatihan:
    - Fase 1 (Epoch 1-5): Warmup (Backbone beku)
    - Fase 2 (Epoch 6-12): Gradual Unfreezing
    - Fase 3 (Epoch 13-End): Full Fine-Tuning dengan Discriminative Learning Rate
    """
    def __init__(
        self,
        model_key: str,
        dataloaders: Dict[str, DataLoader],
        device: Optional[str] = None,
        use_amp: bool = True,
        label_smoothing: float = 0.1
    ):
        self.model_key = model_key
        self.dataloaders = dataloaders
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and (self.device == "cuda")

        # Inisialisasi Model
        print(f"\n[BUILD MODEL] Menginstansiasi arsitektur: {model_key}...")
        self.model = build_model(model_key=model_key, num_classes=NUM_CLASSES, pretrained=True)
        self.model.to(self.device)

        # Loss function dengan label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Logger
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": []
        }

        # Path log & checkpoint
        self.checkpoint = ModelCheckpoint(
            save_dir=SAVED_MODELS_DIR,
            model_name=model_key,
            mode="max",
            metric_name="val_acc"
        )
        self.early_stopping = EarlyStopping(patience=8, mode="min")

        # TensorBoard Writer
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            log_dir = LOGS_DIR / f"{model_key}_{int(time.time())}"
            self.writer = SummaryWriter(log_dir=str(log_dir))

    def _setup_optimizer_and_scheduler(
        self,
        phase: int,
        total_epochs: int,
        current_epoch: int
    ) -> Tuple[torch.optim.Optimizer, Any]:
        """Menyiapkan optimizer dan scheduler sesuai tahapan fase pelatihan."""
        remaining_epochs = max(1, total_epochs - current_epoch)

        if phase == 1:
            freeze_backbone(self.model)
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=5e-4,
                weight_decay=1e-2
            )
        elif phase == 2:
            unfreeze_top_stages(self.model, num_stages=1)
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=1e-4,
                weight_decay=1e-2
            )
        else:  # Phase 3
            unfreeze_all(self.model)
            param_groups = get_discriminative_param_groups(
                self.model,
                backbone_lr=1e-5,
                head_lr=1e-4,
                weight_decay=1e-2
            )
            optimizer = torch.optim.AdamW(param_groups)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=remaining_epochs,
            eta_min=1e-6
        )
        return optimizer, scheduler

    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """Melatih model selama 1 epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            # Gradient clipping untuk kestabilan konvergensi
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total_samples += images.size(0)

        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = (correct / max(1, total_samples)) * 100.0
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Mengevaluasi model pada validation set (Loss, Top-1 Acc, Top-5 Acc)."""
        self.model.eval()
        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total_samples = 0

        for images, targets in dataloader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)

            # Top-1 Accuracy
            preds_top1 = outputs.argmax(dim=-1)
            correct_top1 += (preds_top1 == targets).sum().item()

            # Top-5 Accuracy
            _, preds_top5 = outputs.topk(min(5, outputs.size(1)), dim=-1)
            correct_top5 += preds_top5.eq(targets.view(-1, 1).expand_as(preds_top5)).sum().item()

            total_samples += images.size(0)

        val_loss = total_loss / max(1, total_samples)
        val_top1 = (correct_top1 / max(1, total_samples)) * 100.0
        val_top5 = (correct_top5 / max(1, total_samples)) * 100.0
        return val_loss, val_top1, val_top5

    def fit(
        self,
        epochs: int = NUM_EPOCHS,
        warmup_epochs: int = 5,
        stage_unfreeze_epochs: int = 12
    ) -> Dict[str, List[float]]:
        """
        Menjalankan loop pelatihan lengkap multi-fase.
        """
        print("=" * 80)
        print(f"  MEMULAI PELATIHAN: {self.model.paper_name}")
        print(f"  Target Epochs    : {epochs}")
        print(f"  Perangkat        : {self.device.upper()} (AMP: {'Aktif' if self.use_amp else 'Nonaktif'})")
        print("=" * 80)

        current_phase = 1
        optimizer, scheduler = self._setup_optimizer_and_scheduler(current_phase, epochs, 0)

        train_loader = self.dataloaders["train"]
        val_loader = self.dataloaders["val"]

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Transisi Fase
            if epoch == warmup_epochs + 1 and current_phase == 1:
                current_phase = 2
                print(f"\n>>> [TRANSISI] Memulai Fase 2: Unfreeze Top Stages (Epoch {epoch})")
                optimizer, scheduler = self._setup_optimizer_and_scheduler(current_phase, epochs, epoch)

            elif epoch == stage_unfreeze_epochs + 1 and current_phase == 2:
                current_phase = 3
                print(f"\n>>> [TRANSISI] Memulai Fase 3: Full Fine-Tuning dengan Discriminative LR (Epoch {epoch})")
                optimizer, scheduler = self._setup_optimizer_and_scheduler(current_phase, epochs, epoch)

            # Train 1 epoch
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)

            # Evaluasi pada data validasi
            val_loss, val_top1, val_top5 = self.evaluate(val_loader)

            # Update scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            # Catat riwayat
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["train_acc"].append(round(train_acc, 2))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_acc"].append(round(val_top1, 2))
            self.history["lr"].append(current_lr)

            # Tensorboard
            if self.writer:
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("Acc/Train", train_acc, epoch)
                self.writer.add_scalar("Acc/Val_Top1", val_top1, epoch)
                self.writer.add_scalar("Acc/Val_Top5", val_top5, epoch)
                self.writer.add_scalar("LearningRate", current_lr, epoch)

            epoch_duration = time.time() - epoch_start
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}% | "
                f"Time: {epoch_duration:.1f}s"
            )

            # Simpan checkpoint jika membaik
            self.checkpoint.step(val_top1, self.model, epoch, optimizer)

            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"\n🛑 [Early Stopping] Pelatihan dihentikan pada epoch ke-{epoch} karena konvergensi.")
                break

        total_duration = (time.time() - start_time) / 60.0
        print(f"\n✅ Pelatihan {self.model.paper_name} Selesai dalam {total_duration:.2f} menit.")

        # Simpan riwayat ke JSON
        history_path = REPORTS_DIR / f"history_{self.model_key}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        print(f"  [OK] Riwayat metrik disimpan: {history_path}")

        # Buat kurva grafik untuk Bab IV
        self._plot_training_curves()

        return self.history

    def _plot_training_curves(self) -> None:
        """Menghasilkan grafik kurva Loss dan Akurasi untuk Bab IV Laporan TA."""
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / f"kurva_training_{self.model_key}.png"

        epochs_range = range(1, len(self.history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Kurva Loss
        ax1.plot(epochs_range, self.history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        ax1.plot(epochs_range, self.history["val_loss"], label="Val Loss", color="#d62728", lw=2, linestyle="--")
        ax1.set_title(f"Kurva Loss ({self.model.paper_name})", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross Entropy Loss")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()

        # Kurva Akurasi
        ax2.plot(epochs_range, self.history["train_acc"], label="Train Top-1 Acc", color="#2ca02c", lw=2)
        ax2.plot(epochs_range, self.history["val_acc"], label="Val Top-1 Acc", color="#ff7f0e", lw=2, linestyle="--")
        ax2.set_title(f"Kurva Akurasi ({self.model.paper_name})", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Akurasi (%)")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  📊 [Grafik] Kurva latihan disimpan untuk Bab IV: {save_path}")


# ==============================================================================
# 4. EKSPERIMEN HEAD-TO-HEAD KOMPARASI ARSITEKTUR SOTA
# ==============================================================================
def create_synthetic_dataloaders(input_size: int = 224, batch_size: int = 8) -> Dict[str, DataLoader]:
    """
    Membuat DataLoader sintetis untuk dry-run / unit testing bila gambar fisik
    belum sepenuhnya diunduh ke folder data/raw/.
    """
    print("  [DRY-RUN] Menggunakan Synthetic DataLoader untuk memverifikasi loop pelatihan...")
    dummy_x = torch.randn(32, 3, input_size, input_size)
    dummy_y = torch.randint(0, NUM_CLASSES, (32,))
    dataset = TensorDataset(dummy_x, dummy_y)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def run_head_to_head_experiment(
    models_to_compare: List[str] = ["convnextv2_nano", "mobilenetv4_conv_medium"],
    epochs_per_model: int = NUM_EPOCHS
) -> Dict[str, Dict]:
    """
    Menjalankan pelatihan dan komparasi head-to-head untuk minimal 2 arsitektur SOTA.
    """
    print("\n" + "=" * 80)
    print("  EKSPERIMEN HEAD-TO-HEAD KOMPARASI ARSITEKTUR SOTA")
    print(f"  Model yang diuji: {models_to_compare}")
    print("=" * 80)

    comparison_results = {}

    for model_key in models_to_compare:
        print(f"\n>>> Memulai Eksperimen Model: {model_key.upper()} <<<")
        input_size = MODEL_CONFIGS[model_key]["input_size"]

        # Cek apakah folder data/train sudah ada gambar
        has_real_data = False
        train_folder = TRAIN_DIR
        if train_folder.exists() and any(train_folder.glob("*/*.*")):
            has_real_data = True

        if has_real_data:
            print(f"  Memuat dataset nyata (Input Size: {input_size}x{input_size})...")
            dataloaders = create_dataloaders(model_key=model_key, batch_size=BATCH_SIZE)
        else:
            print(f"  [CATATAN] Dataset fisik belum tersedia di folder data/. Menggunakan synthetic loader...")
            dataloaders = create_synthetic_dataloaders(input_size=input_size, batch_size=BATCH_SIZE)

        trainer = ModelTrainer(model_key=model_key, dataloaders=dataloaders)
        history = trainer.fit(
            epochs=min(epochs_per_model, 2 if not has_real_data else epochs_per_model),
            warmup_epochs=1 if not has_real_data else 5,
            stage_unfreeze_epochs=1 if not has_real_data else 12
        )

        comparison_results[model_key] = {
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
            "best_val_acc": max(history["val_acc"]) if history["val_acc"] else 0,
        }

    # Simpan hasil komparasi
    comp_file = REPORTS_DIR / "head_to_head_summary.json"
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    print("\n" + "=" * 80)
    print("  HASIL SEMENTARA KOMPARASI HEAD-TO-HEAD")
    print("=" * 80)
    for mk, res in comparison_results.items():
        print(f"  - {MODEL_CONFIGS[mk]['paper_name']:<25}: Best Val Acc = {res['best_val_acc']:.2f}% | Val Loss = {res['final_val_loss']:.4f}")
    print("=" * 80)

    return comparison_results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Jalankan eksperimen head-to-head untuk 2 model utama
    run_head_to_head_experiment(
        models_to_compare=["convnextv2_nano", "mobilenetv4_conv_medium"],
        epochs_per_model=NUM_EPOCHS
    )
