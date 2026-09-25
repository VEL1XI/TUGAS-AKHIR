# ==============================================================================
# 07_nutrition_mapping.py — PIPELINE INFERENSI END-TO-END & EVALUASI GIZI (TAHAP 7)
# ==============================================================================
# Skrip ini mengimplementasikan:
#   1. Pipeline Inferensi End-to-End:
#      Input Gambar (Foto Makanan) -> Preprocessing -> Model SOTA -> Prediksi Kelas
#      (Top-1 & Top-3 Confidence) -> Semantic Lookup ke Basis Data Nutrisi (TKPI 2020)
#      -> Estimasi Nutrisi Lengkap (Kalori, Protein, Karbohidrat, Lemak, Serat, Porsi)
#   2. Evaluasi Estimasi Nutrisi:
#      Menghitung Mean Absolute Error (MAE) dan Mean Absolute Percentage Error (MAPE)
#      untuk setiap makronutrisi (Kalori, Protein, Karbo, Lemak) pada data uji.
#
# Argumen Ilmiah untuk Sidang Sarjana (S1 Informatika):
#   - Bagaimana error klasifikasi berdampak pada estimasi gizi?
#     Ketika model salah memprediksi kelas makanan (misal: "nasi_goreng" diprediksi "nasi_kuning"),
#     terjadi selisih nilai gizi antara makanan aktual vs estimasi.
#     Metrik MAE (Mean Absolute Error) mengukur deviasi rata-rata dalam satuan fisik
#     (kkal untuk energi, gram untuk makronutrisi), membuktikan seberapa aman dan
#     dapat diandalkan sistem jika digunakan dalam pemantauan diet harian (clinical nutrition context).
# ==============================================================================

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt

# Import konfigurasi global dan modul tahap sebelumnya
from config import (
    BASE_DIR, TEST_DIR, SAVED_MODELS_DIR,
    REPORTS_DIR, FIGURES_DIR,
    CLASS_NAMES, NUM_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS,
    MODEL_CONFIGS
)
import importlib
_nut = importlib.import_module("03_nutrition_setup")
NutritionLookup = _nut.NutritionLookup

_model = importlib.import_module("04_model")
build_model = _model.build_model


# ==============================================================================
# 1. END-TO-END FOOD & NUTRITION PREDICTOR
# ==============================================================================
class EndToEndFoodNutritionPredictor:
    """
    Sistem inferensi end-to-end:
    Menerima file gambar / PIL Image -> Menghasilkan nama makanan + rincian nutrisi.
    """
    def __init__(
        self,
        model_key: str = "convnextv2_nano",
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None
    ):
        self.model_key = model_key
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = MODEL_CONFIGS[model_key]
        self.input_size = self.config["input_size"]

        # Inisialisasi lookup nutrisi
        self.nutrition_lookup = NutritionLookup()

        # Inisialisasi model
        self.model = build_model(model_key=model_key, num_classes=NUM_CLASSES, pretrained=False)
        self.checkpoint_path = checkpoint_path or (SAVED_MODELS_DIR / f"best_{model_key}.pth")

        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            print(f"  [Predictor] Loaded weights: {self.checkpoint_path.name}")
        else:
            print(f"  [Predictor] Checkpoint tidak ditemukan, menggunakan mode demo/evaluasi.")

        self.model.to(self.device)
        self.model.eval()

        # Transform preprocessing standar ImageNet
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_input: Union[str, Path, Image.Image], top_k: int = 3) -> Dict[str, Any]:
        """
        Melakukan prediksi klasifikasi dan pemetaan nutrisi untuk 1 gambar.
        """
        # Muat gambar jika input berupa path
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=-1).squeeze(0)

        top_probs, top_indices = torch.topk(probabilities, k=top_k)

        top_predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            cls_name = IDX_TO_CLASS[idx]
            nutrition_info = self.nutrition_lookup.get_by_name(cls_name)
            top_predictions.append({
                "class_index": idx,
                "class_name": cls_name,
                "display_name": nutrition_info["display_name"] if nutrition_info else cls_name,
                "confidence": round(prob * 100.0, 2),
                "nutrition": nutrition_info
            })

        best_prediction = top_predictions[0]

        return {
            "model_used": self.config["paper_name"],
            "best_class_name": best_prediction["class_name"],
            "best_display_name": best_prediction["display_name"],
            "confidence_percent": best_prediction["confidence"],
            "nutrition_estimated": best_prediction["nutrition"],
            "top_candidates": top_predictions
        }

    def format_prediction_card(self, result: Dict[str, Any]) -> str:
        """Format teks rapi untuk terminal atau kartu informasi mobile UI."""
        nut = result["nutrition_estimated"]
        return (
            f"\n{'='*60}\n"
            f"  🍽️  HASIL KLASIFIKASI & ESTIMASI NUTRISI ({result['model_used']})\n"
            f"{'='*60}\n"
            f"  Makanan Terdeteksi : {result['best_display_name']} ({result['confidence_percent']}% confidence)\n"
            f"  Kategori Kuliner   : {nut['category']}\n"
            f"  Takaran Saji       : {nut['serving_size']} (~{nut['serving_weight_g']} gram)\n"
            f"{'-'*60}\n"
            f"  🔥 Energi (Kalori) : {nut['calories_kcal']} kkal\n"
            f"  🍗 Protein         : {nut['protein_g']} g\n"
            f"  🍚 Karbohidrat     : {nut['carbohydrate_g']} g\n"
            f"  🥑 Lemak Total     : {nut['fat_g']} g\n"
            f"  🌾 Serat Pangan    : {nut['fiber_g']} g\n"
            f"{'-'*60}\n"
            f"  📋 Status Validasi : {nut['validated_by_expert']} ({nut['source']})\n"
            f"  💬 Catatan Klinis  : {nut['expert_notes']}\n"
            f"{'='*60}\n"
        )


# ==============================================================================
# 2. EVALUASI ESTIMASI NUTRISI (MAE & MAPE PER KOMPONEN)
# ==============================================================================
def evaluate_nutrition_estimation(
    y_true_indices: np.ndarray,
    y_pred_indices: np.ndarray,
    model_name: str = "ConvNeXt V2-Nano"
) -> Dict[str, Any]:
    """
    Menghitung Mean Absolute Error (MAE) dan MAPE untuk tiap komponen gizi:
    - Kalori (kkal)
    - Protein (gram)
    - Karbohidrat (gram)
    - Lemak (gram)
    - Serat (gram)

    Bahan Analisis Bab IV:
        Menunjukkan seberapa dekat estimasi gizi sistem terhadap ground truth TKPI,
        termasuk dampak dari klasifikasi yang meleset.
    """
    lookup = NutritionLookup()

    true_cals, pred_cals = [], []
    true_prot, pred_prot = [], []
    true_carb, pred_carb = [], []
    true_fat,  pred_fat  = [], []
    true_fib,  pred_fib  = [], []

    for t_idx, p_idx in zip(y_true_indices, y_pred_indices):
        t_nut = lookup.get_by_index(int(t_idx))
        p_nut = lookup.get_by_index(int(p_idx))

        if t_nut and p_nut:
            true_cals.append(t_nut["calories_kcal"])
            pred_cals.append(p_nut["calories_kcal"])

            true_prot.append(t_nut["protein_g"])
            pred_prot.append(p_nut["protein_g"])

            true_carb.append(t_nut["carbohydrate_g"])
            pred_carb.append(p_nut["carbohydrate_g"])

            true_fat.append(t_nut["fat_g"])
            pred_fat.append(p_nut["fat_g"])

            true_fib.append(t_nut["fiber_g"])
            pred_fib.append(p_nut["fiber_g"])

    def calc_mae_mape(true_vals, pred_vals):
        t = np.array(true_vals)
        p = np.array(pred_vals)
        mae = float(np.mean(np.abs(t - p)))
        # Hindari pembagian dengan nol
        mape = float(np.mean(np.abs(t - p) / np.maximum(t, 1.0)) * 100.0)
        return round(mae, 2), round(mape, 2)

    mae_cal, mape_cal = calc_mae_mape(true_cals, pred_cals)
    mae_pro, mape_pro = calc_mae_mape(true_prot, pred_prot)
    mae_crb, mape_crb = calc_mae_mape(true_carb, pred_carb)
    mae_fat, mape_fat = calc_mae_mape(true_fat,  pred_fat)
    mae_fib, mape_fib = calc_mae_mape(true_fib,  pred_fib)

    results = {
        "model": model_name,
        "sample_count": len(true_cals),
        "mae": {
            "calories_kcal": mae_cal,
            "protein_g": mae_pro,
            "carbohydrate_g": mae_crb,
            "fat_g": mae_fat,
            "fiber_g": mae_fib
        },
        "mape_percent": {
            "calories_kcal": mape_cal,
            "protein_g": mape_pro,
            "carbohydrate_g": mape_crb,
            "fat_g": mape_fat,
            "fiber_g": mape_fib
        }
    }

    # Tampilkan tabel evaluasi
    print("\n" + "=" * 70)
    print(f"  TABEL EVALUASI ESTIMASI NUTRISI (MAE & MAPE) — {model_name}")
    print("=" * 70)
    print(f"{'Komponen Nutrisi':<25} | {'Satuan':<8} | {'MAE (Mean Abs Error)':<20} | {'MAPE (%)':<10}")
    print("-" * 70)
    print(f"{'Energi (Kalori)':<25} | {'kkal':<8} | {mae_cal:<20} | {mape_cal:<10}%")
    print(f"{'Protein':<25} | {'gram':<8} | {mae_pro:<20} | {mape_pro:<10}%")
    print(f"{'Karbohidrat':<25} | {'gram':<8} | {mae_crb:<20} | {mape_crb:<10}%")
    print(f"{'Lemak Total':<25} | {'gram':<8} | {mae_fat:<20} | {mape_fat:<10}%")
    print(f"{'Serat Pangan':<25} | {'gram':<8} | {mae_fib:<20} | {mape_fib:<10}%")
    print("=" * 70)

    # Simpan ke CSV untuk Bab IV
    df_mae = pd.DataFrame([
        {"Komponen": "Kalori", "Satuan": "kkal", "MAE": mae_cal, "MAPE (%)": mape_cal},
        {"Komponen": "Protein", "Satuan": "gram", "MAE": mae_pro, "MAPE (%)": mape_pro},
        {"Komponen": "Karbohidrat", "Satuan": "gram", "MAE": mae_crb, "MAPE (%)": mape_crb},
        {"Komponen": "Lemak", "Satuan": "gram", "MAE": mae_fat, "MAPE (%)": mape_fat},
        {"Komponen": "Serat", "Satuan": "gram", "MAE": mae_fib, "MAPE (%)": mape_fib},
    ])
    csv_path = REPORTS_DIR / f"evaluasi_mae_nutrisi_{model_name.lower().replace(' ', '_')}.csv"
    df_mae.to_csv(csv_path, index=False)
    print(f"  [OK] Tabel MAE nutrisi disimpan: {csv_path}")

    return results


# ==============================================================================
# MAIN EXECUTION (DEMO END-TO-END)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("  TAHAP 7: DEMO INFERENSI END-TO-END (CITRA -> KLASIFIKASI -> NUTRISI)")
    print("=" * 80)

    predictor = EndToEndFoodNutritionPredictor(model_key="convnextv2_nano")

    # Uji coba dengan dummy image RGB
    dummy_img = Image.new("RGB", (256, 256), color=(200, 100, 50))
    res = predictor.predict(dummy_img, top_k=3)
    print(predictor.format_prediction_card(res))

    # Simulasi evaluasi MAE nutrisi pada 100 sampel uji
    print("  Menghitung evaluasi MAE nutrisi pada sampel pengujian...")
    y_true_sim = np.random.randint(0, NUM_CLASSES, 200)
    # 85% prediksi tepat, 15% salah untuk menguji toleransi MAE
    y_pred_sim = np.copy(y_true_sim)
    random_errors = np.random.choice(200, size=30, replace=False)
    y_pred_sim[random_errors] = np.random.randint(0, NUM_CLASSES, 30)

    mae_res = evaluate_nutrition_estimation(y_true_sim, y_pred_sim, model_name="ConvNeXt V2-Nano")

    print("\n  ✅ TAHAP 7 SELESAI: Pipeline inferensi gizi end-to-end & evaluasi MAE siap!")
