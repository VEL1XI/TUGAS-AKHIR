# ==============================================================================
# 06_evaluation.py — EVALUASI KOMPREHENSIF MODEL KLASIFIKASI (TAHAP 6)
# ==============================================================================
# Skrip ini mengimplementasikan evaluasi mendalam pada data pengujian (Test Set):
#   1. Metrik Global: Top-1 Accuracy, Top-5 Accuracy, Macro F1, Weighted F1
#   2. Classification Report Per Kelas (Precision, Recall, F1 untuk 100 kelas)
#      yang diekspor ke CSV untuk lampiran laporan Bab IV
#   3. Visualisasi Confusion Matrix yang Disesuaikan untuk 100 Kelas:
#      - Matriks Kategori Makro (8x8) dengan persentase kebingungan antar kategori
#      - Tabel Pasangan Paling Bingung (Top-15 Most Confused Pairs)
#   4. Tabel Perbandingan Head-to-Head Antar Arsitektur SOTA
#      (Akurasi, Ukuran File MB, FLOPs, Latensi Inferensi ms, dan Throughput FPS)
# ==============================================================================

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)

# Import konfigurasi global
from config import (
    BASE_DIR, TEST_DIR, SAVED_MODELS_DIR,
    REPORTS_DIR, FIGURES_DIR,
    CLASS_NAMES, NUM_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS,
    CATEGORY_MAP, CLASS_TO_CATEGORY, MODEL_CONFIGS
)
import importlib
_prep = importlib.import_module("02_preprocessing")
create_dataloaders = _prep.create_dataloaders

_model = importlib.import_module("04_model")
build_model = _model.build_model
profile_model = _model.profile_model


# ==============================================================================
# 1. EVALUATION ENGINE: PREDIKSI TEST SET
# ==============================================================================
class ModelEvaluator:
    """
    Kelas evaluator untuk menghitung semua metrik performa model pada test set.
    """
    def __init__(
        self,
        model_key: str,
        checkpoint_path: Optional[Path] = None,
        dataloader: Optional[DataLoader] = None,
        device: Optional[str] = None
    ):
        self.model_key = model_key
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Muat model
        self.model = build_model(model_key=model_key, num_classes=NUM_CLASSES, pretrained=False)
        self.checkpoint_path = checkpoint_path or (SAVED_MODELS_DIR / f"best_{model_key}.pth")

        if self.checkpoint_path.exists():
            print(f"  Memuat bobot terlatih dari: {self.checkpoint_path.name}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print(f"  [CATATAN] Checkpoint {self.checkpoint_path.name} belum ditemukan. Menggunakan bobot inisialisasi untuk evaluasi demo.")

        self.model.to(self.device)
        self.model.eval()

        self.dataloader = dataloader
        self.y_true: np.ndarray = np.array([])
        self.y_pred: np.ndarray = np.array([])
        self.y_probs: np.ndarray = np.array([])

    @torch.no_grad()
    def run_inference(self) -> None:
        """Menjalankan inferensi pada seluruh test set dan mengumpulkan logit prediksi."""
        if self.dataloader is None:
            has_real_test = TEST_DIR.exists() and any(TEST_DIR.glob("*/*.*"))
            if has_real_test:
                dataloaders = create_dataloaders(model_key=self.model_key, batch_size=32)
                self.dataloader = dataloaders["test"]
            else:
                print(f"  [CATATAN] Folder test ({TEST_DIR}) belum terisi gambar fisik.")
                print(f"            Menggunakan synthetic test loader untuk demo evaluasi...")
                from torch.utils.data import TensorDataset
                input_size = MODEL_CONFIGS[self.model_key]["input_size"]
                dummy_x = torch.randn(64, 3, input_size, input_size)
                dummy_y = torch.randint(0, NUM_CLASSES, (64,))
                self.dataloader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=16, shuffle=False)

        all_preds = []
        all_targets = []
        all_probs = []

        print(f"  Menjalankan inferensi evaluasi pada Test Set ({len(self.dataloader)} batch)...")

        for images, targets in self.dataloader:
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

        self.y_true = np.array(all_targets)
        self.y_pred = np.array(all_preds)
        self.y_probs = np.array(all_probs)

    def compute_global_metrics(self) -> Dict[str, float]:
        """Menghitung akurasi Top-1, Top-5, Macro F1, Weighted F1."""
        top1_acc = accuracy_score(self.y_true, self.y_pred) * 100.0

        # Top-5 Accuracy
        top5_correct = 0
        for i in range(len(self.y_true)):
            top5_indices = np.argsort(self.y_probs[i])[-5:]
            if self.y_true[i] in top5_indices:
                top5_correct += 1
        top5_acc = (top5_correct / max(1, len(self.y_true))) * 100.0

        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average="macro", zero_division=0
        )
        w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average="weighted", zero_division=0
        )

        return {
            "top1_accuracy": round(top1_acc, 2),
            "top5_accuracy": round(top5_acc, 2),
            "macro_precision": round(precision * 100.0, 2),
            "macro_recall": round(recall * 100.0, 2),
            "macro_f1": round(f1 * 100.0, 2),
            "weighted_f1": round(w_f1 * 100.0, 2)
        }

    def generate_per_class_report(self) -> pd.DataFrame:
        """
        Menghasilkan tabel classification report lengkap 100 kelas dan menyimpannya ke CSV.
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
        )

        records = []
        for idx in range(NUM_CLASSES):
            class_name = IDX_TO_CLASS[idx]
            category = CLASS_TO_CATEGORY.get(class_name, "Lainnya")
            records.append({
                "class_index": idx,
                "class_name": class_name,
                "category": category,
                "precision": round(precision[idx] * 100.0, 2),
                "recall": round(recall[idx] * 100.0, 2),
                "f1_score": round(f1[idx] * 100.0, 2),
                "support": int(support[idx])
            })

        df = pd.DataFrame(records)
        csv_path = REPORTS_DIR / f"per_class_report_{self.model_key}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  [OK] Saved per-class classification report: {csv_path}")
        return df

    def analyze_most_confused_pairs(self, top_n: int = 15) -> pd.DataFrame:
        """
        Menganalisis 15 pasangan kelas yang paling sering salah diprediksi.
        Analisis ini sangat krusial saat sidang Tugas Akhir untuk menjelaskan
        karakteristik visual kuliner yang mirip (e.g. Soto Ayam vs Soto Lamongan).
        """
        cm = confusion_matrix(self.y_true, self.y_pred, labels=list(range(NUM_CLASSES)))
        np.fill_diagonal(cm, 0)  # Abaikan prediksi benar

        confused_pairs = []
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if cm[i, j] > 0:
                    confused_pairs.append({
                        "True Class": IDX_TO_CLASS[i],
                        "Predicted As": IDX_TO_CLASS[j],
                        "Error Count": int(cm[i, j]),
                        "True Category": CLASS_TO_CATEGORY.get(IDX_TO_CLASS[i], "-"),
                        "Pred Category": CLASS_TO_CATEGORY.get(IDX_TO_CLASS[j], "-")
                    })

        df_confused = pd.DataFrame(confused_pairs)
        if not df_confused.empty:
            df_confused = df_confused.sort_values(by="Error Count", ascending=False).head(top_n).reset_index(drop=True)

        csv_path = REPORTS_DIR / f"top_confused_pairs_{self.model_key}.csv"
        df_confused.to_csv(csv_path, index=False)
        print(f"  [OK] Saved top confused pairs table: {csv_path}")
        return df_confused

    def plot_category_confusion_matrix(self) -> None:
        """
        Visualisasi Confusion Matrix Makro (8x8 Kategori) yang mudah dibaca untuk Bab IV.
        Matriks 100x100 terlalu padat; representasi 8 kategori kuliner memberikan
        kejelasan visual tinggi untuk dokumen skripsi.
        """
        categories = list(CATEGORY_MAP.keys())
        cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

        # Petakan label kelas ke indeks kategori
        y_true_cat = [cat_to_idx[CLASS_TO_CATEGORY[IDX_TO_CLASS[c]]] for c in self.y_true]
        y_pred_cat = [cat_to_idx[CLASS_TO_CATEGORY[IDX_TO_CLASS[c]]] for c in self.y_pred]

        cm_cat = confusion_matrix(y_true_cat, y_pred_cat, labels=list(range(len(categories))))
        # Normalisasi ke persentase baris
        cm_norm = cm_cat.astype("float") / np.maximum(cm_cat.sum(axis=1, keepdims=True), 1) * 100.0

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            xticklabels=[c.replace("&", "&\n") for c in categories],
            yticklabels=categories,
            cbar_kws={"label": "Akurasi / Kesalahan (%)"}
        )
        plt.title(f"Confusion Matrix Antar Kategori Makanan\n({MODEL_CONFIGS[self.model_key]['paper_name']})", fontsize=13, fontweight="bold")
        plt.xlabel("Kategori Prediksi", fontsize=11, fontweight="bold")
        plt.ylabel("Kategori Sebenarnya (Ground Truth)", fontsize=11, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        save_path = FIGURES_DIR / f"confusion_matrix_category_{self.model_key}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  📊 [Grafik] Confusion matrix kategori disimpan: {save_path}")


# ==============================================================================
# 2. KOMPARASI HEAD-TO-HEAD ARSITEKTUR SOTA (TABEL BAB IV)
# ==============================================================================
def compare_all_architectures(models: List[str] = ["convnextv2_nano", "mobilenetv4_conv_medium", "efficientnetv2_s"]) -> pd.DataFrame:
    """
    Menghasilkan tabel komparasi komprehensif antar arsitektur SOTA
    untuk dicantumkan langsung pada Bab IV Laporan Tugas Akhir.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    print("\n" + "=" * 90)
    print("  MEMBUAT TABEL KOMPARASI LENGKAP SEMUA ARSITEKTUR SOTA (BAB IV)")
    print("=" * 90)

    for m_key in models:
        cfg = MODEL_CONFIGS[m_key]
        model = build_model(m_key, num_classes=NUM_CLASSES, pretrained=False)
        profile = profile_model(model, device="cpu")

        # Cek apakah ada riwayat training
        hist_path = REPORTS_DIR / f"history_{m_key}.json"
        best_val_acc = 0.0
        if hist_path.exists():
            with open(hist_path, "r", encoding="utf-8") as f:
                h = json.load(f)
                best_val_acc = max(h.get("val_acc", [0.0]))

        summary_rows.append({
            "Arsitektur": cfg["paper_name"],
            "Tahun Rilis": cfg["year"],
            "Resolusi Input": f"{cfg['input_size']}x{cfg['input_size']}",
            "Parameter (Juta)": profile["total_params_million"],
            "Ukuran FP32 (MB)": profile["weight_size_mb"],
            "Estimasi INT8 (MB)": round(profile["weight_size_mb"] / 4.0, 2),
            "Latensi CPU (ms)": profile["avg_latency_ms"],
            "Throughput (FPS)": profile["throughput_fps"],
            "Best Top-1 Acc (%)": best_val_acc
        })

    df_comp = pd.DataFrame(summary_rows)
    csv_file = REPORTS_DIR / "komparasi_arsitektur_sota_bab4.csv"
    df_comp.to_csv(csv_file, index=False)
    print(df_comp.to_string(index=False))
    print("=" * 90)
    print(f"  [OK] Tabel komparasi disimpan untuk laporan: {csv_file}")
    return df_comp


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("  TAHAP 6: EVALUASI LENGKAP MODEL KLASIFIKASI PADA 100 KELAS")
    print("=" * 80)

    # 1. Tabel Komparasi Semua Arsitektur SOTA
    compare_all_architectures()

    # 2. Demo Evaluasi Model Utama (ConvNeXt V2-Nano)
    print("\n[EVALUASI TEST SET: ConvNeXt V2-Nano]")
    evaluator = ModelEvaluator(model_key="convnextv2_nano")
    evaluator.run_inference()

    # Hitung metrik
    metrics = evaluator.compute_global_metrics()
    print("\n--- Metrik Global ---")
    for k, v in metrics.items():
        print(f"  {k:<18}: {v}")

    # Report per kelas
    df_class = evaluator.generate_per_class_report()

    # Analisis pasangan bingung
    df_conf = evaluator.analyze_most_confused_pairs(top_n=10)
    print("\n--- 10 Pasangan Kelas Paling Sering Tertukar (Bahan Pembahasan Bab IV) ---")
    print(df_conf.to_string(index=False))

    # Matriks kategori
    evaluator.plot_category_confusion_matrix()

    print("\n  ✅ TAHAP 6 SELESAI: Evaluasi klasifikasi lengkap & visualisasi siap!")
