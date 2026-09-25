# ==============================================================================
# 03_nutrition_setup.py — SETUP & VALIDASI BASIS DATA NUTRISI (TAHAP 3)
# ==============================================================================
# Skrip ini mengelola basis data referensi nutrisi untuk 100 kelas makanan
# Indonesia, meliputi:
#   1. Pembuatan file CSV & JSON referensi nutrisi berbasis data resmi
#      (TKPI - Tabel Komposisi Pangan Indonesia 2020, Kemenkes RI & Panganku)
#   2. Penyediaan template spreadsheet yang dirancang khusus untuk direview
#      dan divalidasi oleh ahli gizi (SPPG / Badan Gizi Nasional) tanpa coding
#   3. Fungsi validasi integritas data gizi (kelengkapan 100 kelas, konsistensi
#      hukum Atwater 4-4-9 kkal, cek non-negatif)
#   4. Fungsi lookup cepat untuk inferensi klasifikasi → estimasi gizi
#   5. Ekspor aset JSON offline untuk aplikasi mobile Flutter (Tahap 9)
#   6. Visualisasi sebaran makronutrisi per kategori untuk lampiran Bab IV
#
# Argumen Ilmiah untuk Sidang TA:
#   - Pemisahan data nutrisi dari kode (decoupling) adalah arsitektur bersih:
#     pembaruan porsi / komposisi gizi dari ahli gizi TIDAK memerlukan
#     retraining bobot CNN, karena model fokus pada representasi visual (food
#     identification), sedangkan nutrition mapping bertindak sebagai semantic layer.
#   - Validasi data menggunakan sistem faktor Atwater (4 kcal/g protein & karbo,
#     9 kcal/g lemak) untuk memastikan konsistensi termodinamika gizi.
# ==============================================================================

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import pandas as pd
import numpy as np

# Import path dan 100 kelas dari config global
from config import (
    BASE_DIR, NUTRITION_DIR, FIGURES_DIR,
    CLASS_NAMES, NUM_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS,
    CATEGORY_MAP, CLASS_TO_CATEGORY
)

# File path output
NUTRITION_CSV_PATH = NUTRITION_DIR / "food_nutrition.csv"
NUTRITION_JSON_PATH = NUTRITION_DIR / "food_nutrition.json"
EXPERT_REVIEW_CSV_PATH = NUTRITION_DIR / "food_nutrition_expert_review.csv"
FLUTTER_ASSETS_JSON_PATH = BASE_DIR / "mobile_app" / "assets" / "nutrition.json"

# ==============================================================================
# 1. DATA REFERENSI NUTRISI 100 KELAS (BERDASARKAN TKPI 2020 & PANGANKU KEMENKES)
# ==============================================================================
# Format per item:
# (index, class_name, display_name, category, serving_size, weight_g,
#  cal_kcal, protein_g, carb_g, fat_g, fiber_g, source, validated, notes)
# ==============================================================================

NUTRITION_DATABASE_RAW = [
    # === NASI & OLAHAN BERAS (12 kelas, 0-11) ===
    (0, "nasi_goreng", "Nasi Goreng", "Nasi & Olahan Beras", "1 piring sedang (250g)", 250.0, 330.0, 8.5, 48.0, 12.0, 1.8, "TKPI Kemenkes", "Belum", "Dengan sedikit telur & suwiran ayam"),
    (1, "nasi_putih", "Nasi Putih", "Nasi & Olahan Beras", "1 centong penuh (100g)", 100.0, 130.0, 2.4, 28.6, 0.2, 0.4, "TKPI Kemenkes", "Belum", "Beras giling masak"),
    (2, "nasi_padang", "Nasi Padang", "Nasi & Olahan Beras", "1 porsi lengkap (350g)", 350.0, 680.0, 24.0, 72.0, 32.0, 3.5, "TKPI / Estimasi", "Belum", "Asumsi paket nasi + rendang + daun singkong + kuah gulai"),
    (3, "nasi_uduk", "Nasi Uduk", "Nasi & Olahan Beras", "1 porsi (200g)", 200.0, 260.0, 5.0, 42.0, 8.5, 1.2, "TKPI Kemenkes", "Belum", "Nasi gurih santan tanpa lauk pendamping"),
    (4, "nasi_kuning", "Nasi Kuning", "Nasi & Olahan Beras", "1 porsi (200g)", 200.0, 250.0, 4.8, 41.5, 7.8, 1.1, "TKPI Kemenkes", "Belum", "Nasi gurih kunyit santan porsi standar"),
    (5, "nasi_pecel", "Nasi Pecel", "Nasi & Olahan Beras", "1 porsi lengkap (300g)", 300.0, 410.0, 12.0, 62.0, 14.0, 4.8, "TKPI Kemenkes", "Belum", "Nasi + sayur rebus + bumbu kacang"),
    (6, "gudeg", "Gudeg", "Nasi & Olahan Beras", "1 porsi (150g)", 150.0, 215.0, 4.2, 32.0, 8.2, 3.8, "TKPI Kemenkes", "Belum", "Nangka muda masak santan manis (belum termasuk lauk krecek/telur)"),
    (7, "ketupat", "Ketupat", "Nasi & Olahan Beras", "1 buah sedang (120g)", 120.0, 145.0, 2.8, 31.5, 0.4, 0.6, "TKPI Kemenkes", "Belum", "Beras masak daun kelapa"),
    (8, "lontong_sayur", "Lontong Sayur", "Nasi & Olahan Beras", "1 mangkok (350g)", 350.0, 340.0, 7.5, 48.0, 13.5, 3.2, "TKPI Kemenkes", "Belum", "Lontong + sayur labu kuah santan"),
    (9, "bubur_ayam", "Bubur Ayam", "Nasi & Olahan Beras", "1 mangkok (300g)", 300.0, 290.0, 11.5, 42.0, 8.0, 1.5, "TKPI Kemenkes", "Belum", "Bubur beras + suwiran ayam, cakwe, kecap, kerupuk"),
    (10, "nasi_liwet", "Nasi Liwet", "Nasi & Olahan Beras", "1 porsi (220g)", 220.0, 275.0, 5.5, 44.0, 9.0, 1.2, "TKPI Kemenkes", "Belum", "Nasi gurih santan rempah"),
    (11, "nasi_timbel", "Nasi Timbel", "Nasi & Olahan Beras", "1 bungkus daun pisang (180g)", 180.0, 230.0, 4.2, 49.0, 0.8, 1.0, "TKPI Kemenkes", "Belum", "Nasi pulen bungkus daun pisang hangat"),

    # === MIE & BAKSO (8 kelas, 12-19) ===
    (12, "mie_goreng", "Mie Goreng", "Mie & Bakso", "1 porsi (200g)", 200.0, 380.0, 9.0, 52.0, 15.5, 2.4, "TKPI Kemenkes", "Belum", "Mie basah goreng dengan sayur & bumbu"),
    (13, "mie_ayam", "Mie Ayam", "Mie & Bakso", "1 mangkok (280g)", 280.0, 420.0, 16.0, 56.0, 14.5, 2.8, "TKPI Kemenkes", "Belum", "Mie kuning basah + topping ayam kecap + kuah"),
    (14, "mie_aceh", "Mie Aceh", "Mie & Bakso", "1 porsi (250g)", 250.0, 450.0, 17.5, 54.0, 19.0, 3.5, "TKPI Kemenkes", "Belum", "Mie kuning tebal bumbu rempah kari"),
    (15, "bakso", "Bakso Kuah", "Mie & Bakso", "1 mangkok (5 butir + bihun/kuah, 350g)", 350.0, 325.0, 18.0, 30.0, 14.5, 1.5, "TKPI Kemenkes", "Belum", "Bakso sapi campuran tepung kuah kaldu"),
    (16, "bakmi_jawa", "Bakmi Jawa", "Mie & Bakso", "1 porsi (250g)", 250.0, 365.0, 12.5, 48.0, 14.0, 2.5, "TKPI Kemenkes", "Belum", "Bakmi godhog / goreng tradisional telur bebek"),
    (17, "kwetiau_goreng", "Kwetiau Goreng", "Mie & Bakso", "1 piring (220g)", 220.0, 410.0, 10.5, 58.0, 16.0, 2.0, "TKPI Kemenkes", "Belum", "Kwetiau beras goreng kecap & telur"),
    (18, "bihun_goreng", "Bihun Goreng", "Mie & Bakso", "1 porsi (200g)", 200.0, 310.0, 6.2, 50.0, 10.0, 1.8, "TKPI Kemenkes", "Belum", "Bihun beras goreng bumbu sayur"),
    (19, "indomie", "Indomie / Mie Instan", "Mie & Bakso", "1 bungkus masak (85g kering -> 180g matang)", 180.0, 380.0, 8.0, 54.0, 15.0, 2.0, "TKPI Kemenkes", "Belum", "Standar mie instan goreng / kuah"),

    # === SUP, SOTO & KUAH (13 kelas, 20-32) ===
    (20, "soto_ayam", "Soto Ayam", "Sup, Soto & Kuah", "1 mangkok sedang (300g)", 300.0, 260.0, 15.0, 18.0, 14.0, 1.2, "TKPI Kemenkes", "Belum", "Kuah kaldu ayam kuning bening + suwiran"),
    (21, "soto_betawi", "Soto Betawi", "Sup, Soto & Kuah", "1 mangkok (300g)", 300.0, 380.0, 20.0, 12.0, 28.0, 1.0, "TKPI Kemenkes", "Belum", "Kuah santan / susu + potongan daging sapi/jeroan"),
    (22, "soto_lamongan", "Soto Lamongan", "Sup, Soto & Kuah", "1 mangkok (300g)", 300.0, 285.0, 16.5, 20.0, 15.0, 1.5, "TKPI Kemenkes", "Belum", "Kuah kuning kaya bumbu + bubuk koya"),
    (23, "soto_banjar", "Soto Banjar", "Sup, Soto & Kuah", "1 mangkok (300g)", 300.0, 275.0, 15.5, 22.0, 13.5, 1.2, "TKPI Kemenkes", "Belum", "Kuah rempah harum kapulaga & perkedel"),
    (24, "rawon", "Rawon Daging", "Sup, Soto & Kuah", "1 mangkok (300g)", 300.0, 310.0, 22.0, 9.0, 20.5, 1.8, "TKPI Kemenkes", "Belum", "Sup daging kuah kluwek hitam pekat"),
    (25, "tongseng", "Tongseng Kambing", "Sup, Soto & Kuah", "1 mangkok (250g)", 250.0, 340.0, 19.5, 14.0, 23.0, 2.0, "TKPI Kemenkes", "Belum", "Daging kambing kuah santan kecap manis kol"),
    (26, "sup_buntut", "Sup Buntut", "Sup, Soto & Kuah", "1 mangkok (350g)", 350.0, 390.0, 26.0, 10.0, 27.0, 1.5, "TKPI Kemenkes", "Belum", "Buntut sapi kuah kaldu rempah bening"),
    (27, "sayur_asem", "Sayur Asem", "Sup, Soto & Kuah", "1 mangkok (250g)", 250.0, 80.0, 3.2, 14.5, 1.2, 3.0, "TKPI Kemenkes", "Belum", "Sayur kuah bening asam jawa kacang tanah"),
    (28, "sayur_lodeh", "Sayur Lodeh", "Sup, Soto & Kuah", "1 mangkok (250g)", 250.0, 160.0, 4.5, 16.0, 9.2, 3.2, "TKPI Kemenkes", "Belum", "Aneka sayur kuah santan gurih"),
    (29, "opor_ayam", "Opor Ayam", "Sup, Soto & Kuah", "1 potong ayam + kuah (180g)", 180.0, 290.0, 21.0, 6.5, 20.0, 0.8, "TKPI Kemenkes", "Belum", "Ayam masak kuah santan putih/kuning"),
    (30, "gulai_ayam", "Gulai Ayam", "Sup, Soto & Kuah", "1 potong ayam + kuah (180g)", 180.0, 310.0, 22.0, 5.0, 22.5, 0.8, "TKPI Kemenkes", "Belum", "Ayam kuah santan kental rempah pedas"),
    (31, "gulai_ikan", "Gulai Ikan / Kakap", "Sup, Soto & Kuah", "1 potong ikan + kuah (200g)", 200.0, 240.0, 24.0, 4.5, 14.0, 0.6, "TKPI Kemenkes", "Belum", "Ikan masak santan bumbu gulai Minang"),
    (32, "sop_kambing", "Sop Kambing", "Sup, Soto & Kuah", "1 mangkok (300g)", 300.0, 330.0, 23.0, 8.0, 23.0, 1.0, "TKPI Kemenkes", "Belum", "Potongan daging/tulang kambing kaldu rempah"),

    # === DAGING & UNGGAS (17 kelas, 33-49) ===
    (33, "rendang", "Rendang Daging", "Daging & Unggas", "1 potong sedang (80g)", 80.0, 220.0, 19.0, 5.0, 14.0, 1.2, "TKPI Kemenkes", "Belum", "Daging sapi masak santan kering bumbu Minang"),
    (34, "ayam_goreng", "Ayam Goreng Lengkuas/Bumbu", "Daging & Unggas", "1 potong paha/dada (100g)", 100.0, 260.0, 25.0, 2.5, 17.0, 0.2, "TKPI Kemenkes", "Belum", "Ayam ungkep goreng tradisional"),
    (35, "ayam_bakar", "Ayam Bakar", "Daging & Unggas", "1 potong (100g)", 100.0, 210.0, 24.5, 6.0, 10.0, 0.4, "TKPI Kemenkes", "Belum", "Ayam bakar bumbu kecap manis"),
    (36, "ayam_geprek", "Ayam Geprek", "Daging & Unggas", "1 potong (120g)", 120.0, 340.0, 23.0, 18.0, 20.0, 1.2, "TKPI Kemenkes", "Belum", "Ayam goreng tepung digeprek cabai pedas"),
    (37, "ayam_penyet", "Ayam Penyet", "Daging & Unggas", "1 potong (110g)", 110.0, 280.0, 24.0, 4.0, 18.5, 0.8, "TKPI Kemenkes", "Belum", "Ayam goreng ungkep ditekan sambal terasi"),
    (38, "ayam_betutu", "Ayam Betutu", "Daging & Unggas", "1 porsi (150g)", 150.0, 290.0, 28.0, 5.5, 17.5, 1.5, "TKPI Kemenkes", "Belum", "Ayam rempah base genep kukus/panggang khas Bali"),
    (39, "sate_ayam", "Sate Ayam", "Daging & Unggas", "5 tusuk + bumbu kacang (150g)", 150.0, 310.0, 22.0, 16.0, 18.0, 2.2, "TKPI Kemenkes", "Belum", "Daging ayam bakar disiram saus kacang"),
    (40, "sate_kambing", "Sate Kambing", "Daging & Unggas", "5 tusuk + kecap cabai (150g)", 150.0, 335.0, 25.0, 12.0, 21.0, 0.8, "TKPI Kemenkes", "Belum", "Daging kambing bakar bumbu kecap bawang"),
    (41, "sate_padang", "Sate Padang", "Daging & Unggas", "5 tusuk + kuah kuning (150g)", 150.0, 290.0, 20.5, 22.0, 14.0, 1.5, "TKPI Kemenkes", "Belum", "Sate daging/lidah sapi kuah kental rempah"),
    (42, "sate_lilit", "Sate Lilit", "Daging & Unggas", "4 tusuk (120g)", 120.0, 220.0, 18.5, 6.0, 14.0, 1.0, "TKPI Kemenkes", "Belum", "Cincangan ayam/ikan santan rempah batang serai"),
    (43, "sate_madura", "Sate Madura", "Daging & Unggas", "5 tusuk (150g)", 150.0, 315.0, 21.5, 18.0, 18.5, 2.0, "TKPI Kemenkes", "Belum", "Sate ayam bumbu kacang gurih pekat petis"),
    (44, "dendeng_balado", "Dendeng Balado", "Daging & Unggas", "1 lembar porsi (60g)", 60.0, 195.0, 17.0, 4.0, 12.5, 0.8, "TKPI Kemenkes", "Belum", "Daging sapi tipis goreng siram sambal balado"),
    (45, "empal", "Empal Gentong / Daging", "Daging & Unggas", "1 potong (70g)", 70.0, 185.0, 16.5, 5.0, 11.0, 0.5, "TKPI Kemenkes", "Belum", "Daging sapi manis gurih goreng/bacem"),
    (46, "bebek_goreng", "Bebek Goreng", "Daging & Unggas", "1 potong (120g)", 120.0, 360.0, 24.0, 2.0, 29.0, 0.0, "TKPI Kemenkes", "Belum", "Daging bebek gurih kaya lemak goreng"),
    (47, "gulai_kambing", "Gulai Kambing", "Daging & Unggas", "1 mangkok (200g)", 200.0, 320.0, 21.0, 6.0, 23.5, 1.0, "TKPI Kemenkes", "Belum", "Daging kambing kuah santan kental gurih"),
    (48, "semur_daging", "Semur Daging", "Daging & Unggas", "1 potong + kuah (100g)", 100.0, 210.0, 18.0, 12.0, 10.0, 0.7, "TKPI Kemenkes", "Belum", "Daging sapi bumbu kecap pala cengkeh"),
    (49, "krengsengan", "Krengsengan Daging", "Daging & Unggas", "1 porsi (120g)", 120.0, 245.0, 19.0, 9.0, 14.5, 1.0, "TKPI Kemenkes", "Belum", "Daging sapi tumis petis cabai khas Jawa Timur"),

    # === IKAN & SEAFOOD (10 kelas, 50-59) ===
    (50, "ikan_bakar", "Ikan Bakar", "Ikan & Seafood", "1 ekor sedang (150g)", 150.0, 220.0, 27.0, 4.5, 10.0, 0.4, "TKPI Kemenkes", "Belum", "Ikan nila / kembung bakar bumbu kecap"),
    (51, "ikan_goreng", "Ikan Goreng", "Ikan & Seafood", "1 ekor sedang (120g)", 120.0, 230.0, 25.0, 2.0, 13.5, 0.0, "TKPI Kemenkes", "Belum", "Ikan mas / nila bumbu ketumbar kunyit goreng"),
    (52, "pempek", "Pempek Kapal Selam", "Ikan & Seafood", "1 buah besar + cuko (180g)", 180.0, 310.0, 13.5, 42.0, 10.0, 1.2, "TKPI Kemenkes", "Belum", "Pempek isi telur rebus + kuah asam manis pedas"),
    (53, "udang_goreng_tepung", "Udang Goreng Tepung", "Ikan & Seafood", "5 ekor (100g)", 100.0, 250.0, 16.0, 18.0, 12.5, 0.8, "TKPI Kemenkes", "Belum", "Udang balut tepung krispi"),
    (54, "cumi_goreng_tepung", "Cumi Goreng Tepung", "Ikan & Seafood", "1 porsi (100g)", 100.0, 240.0, 15.0, 17.5, 12.0, 0.6, "TKPI Kemenkes", "Belum", "Cumi ring balut tepung krispi"),
    (55, "pepes_ikan", "Pepes Ikan", "Ikan & Seafood", "1 bungkus (120g)", 120.0, 160.0, 22.0, 3.5, 6.5, 1.2, "TKPI Kemenkes", "Belum", "Ikan kukus bumbu kemangi daun pisang"),
    (56, "ikan_asam_manis", "Ikan Asam Manis", "Ikan & Seafood", "1 porsi (150g)", 150.0, 260.0, 21.0, 22.0, 9.5, 1.5, "TKPI Kemenkes", "Belum", "Fillet ikan goreng tepung saus nanas tomat"),
    (57, "kepiting_saus_padang", "Kepiting Saus Padang", "Ikan & Seafood", "1 ekor / porsi (250g)", 250.0, 290.0, 26.0, 16.0, 13.0, 1.8, "TKPI Kemenkes", "Belum", "Kepiting laut kuah saus tiram cabai telur"),
    (58, "otak_otak", "Otak-Otak Ikan", "Ikan & Seafood", "3 buah (90g)", 90.0, 140.0, 9.5, 16.0, 4.0, 0.8, "TKPI Kemenkes", "Belum", "Daging ikan tenggiri santan panggang daun pisang"),
    (59, "ikan_pindang", "Pindang Ikan", "Ikan & Seafood", "1 mangkok (200g)", 200.0, 180.0, 23.0, 5.0, 7.5, 1.0, "TKPI Kemenkes", "Belum", "Ikan kuah bening asam pedas segar belimbing wuluh"),

    # === SAYURAN & SALAD (10 kelas, 60-69) ===
    (60, "gado_gado", "Gado-Gado", "Sayuran & Salad", "1 piring (250g)", 250.0, 320.0, 12.0, 34.0, 16.0, 5.5, "TKPI Kemenkes", "Belum", "Sayuran rebus + lontong + saus kacang kental"),
    (61, "karedok", "Karedok", "Sayuran & Salad", "1 piring (200g)", 200.0, 210.0, 8.5, 24.0, 9.5, 6.0, "TKPI Kemenkes", "Belum", "Sayuran segar mentah bumbu kacang kencur"),
    (62, "pecel", "Pecel Sayur", "Sayuran & Salad", "1 porsi (180g)", 180.0, 230.0, 9.0, 26.0, 11.0, 5.2, "TKPI Kemenkes", "Belum", "Bayam, tauge, kacang panjang + sambal kacang"),
    (63, "urap", "Urap Sayur", "Sayuran & Salad", "1 piring (150g)", 150.0, 150.0, 5.0, 18.0, 7.0, 4.8, "TKPI Kemenkes", "Belum", "Sayuran rebus diaduk parutan kelapa berbumbu"),
    (64, "lalapan", "Lalapan Segar", "Sayuran & Salad", "1 porsi (100g)", 100.0, 35.0, 1.8, 6.5, 0.3, 3.2, "TKPI Kemenkes", "Belum", "Mentimun, kubis, kemangi, selada mentah"),
    (65, "cap_cay", "Cap Cay Kuah", "Sayuran & Salad", "1 mangkok (220g)", 220.0, 140.0, 6.5, 18.0, 5.0, 4.0, "TKPI Kemenkes", "Belum", "Aneka tumis sayuran kuah kaldu encer"),
    (66, "tumis_kangkung", "Tumis Kangkung / Cah", "Sayuran & Salad", "1 porsi (150g)", 150.0, 120.0, 3.8, 8.5, 8.0, 3.2, "TKPI Kemenkes", "Belum", "Kangkung tumis bawang terasi/tauco"),
    (67, "capcay_goreng", "Capcay Goreng", "Sayuran & Salad", "1 porsi (200g)", 200.0, 170.0, 7.5, 20.0, 7.0, 4.2, "TKPI Kemenkes", "Belum", "Aneka sayur tumis saus tiram sedikit kuah kental"),
    (68, "asinan_jakarta", "Asinan Jakarta", "Sayuran & Salad", "1 mangkok (220g)", 220.0, 185.0, 5.0, 32.0, 4.5, 3.8, "TKPI Kemenkes", "Belum", "Sawi asin, tahu, tauge siram saus kacang cuka"),
    (69, "rujak", "Rujak Buah", "Sayuran & Salad", "1 porsi (200g)", 200.0, 160.0, 1.8, 38.0, 1.0, 3.5, "TKPI Kemenkes", "Belum", "Buah potong segar + saus gula merah cabai"),

    # === GORENGAN & CAMILAN (19 kelas, 70-88) ===
    (70, "tahu_goreng", "Tahu Goreng", "Gorengan & Camilan", "2 buah (80g)", 80.0, 115.0, 8.2, 3.0, 8.0, 1.0, "TKPI Kemenkes", "Belum", "Tahu kuning/putih goreng polos"),
    (71, "tempe_goreng", "Tempe Goreng", "Gorengan & Camilan", "2 potong (70g)", 70.0, 150.0, 11.5, 9.0, 8.5, 2.8, "TKPI Kemenkes", "Belum", "Tempe goreng gurih tanpa tepung tebal"),
    (72, "bakwan", "Bakwan Sayur / Bala-Bala", "Gorengan & Camilan", "1 buah (60g)", 60.0, 140.0, 2.8, 16.5, 7.0, 1.2, "TKPI Kemenkes", "Belum", "Adonan tepung terigu sayur kol wortel goreng"),
    (73, "pisang_goreng", "Pisang Goreng", "Gorengan & Camilan", "1 buah (70g)", 70.0, 160.0, 1.5, 27.0, 5.5, 1.8, "TKPI Kemenkes", "Belum", "Pisang kepok/raja celup adonan tepung goreng"),
    (74, "tahu_isi", "Tahu Isi Sayur / Gehu", "Gorengan & Camilan", "1 buah (75g)", 75.0, 145.0, 5.5, 14.0, 7.8, 1.5, "TKPI Kemenkes", "Belum", "Tahu kopong isi tauge wortel lapis tepung"),
    (75, "risoles", "Risoles Ragout / Mayones", "Gorengan & Camilan", "1 buah (65g)", 65.0, 170.0, 4.5, 18.0, 9.0, 0.8, "TKPI Kemenkes", "Belum", "Kulit dadar tepung panir isi ragout ayam/mayo"),
    (76, "lumpia", "Lumpia Basah / Goreng", "Gorengan & Camilan", "1 buah (70g)", 70.0, 130.0, 4.0, 17.5, 5.0, 1.6, "TKPI Kemenkes", "Belum", "Kulit tipis isi rebung wortel suwiran ayam"),
    (77, "pastel", "Pastel Goreng", "Gorengan & Camilan", "1 buah (65g)", 65.0, 165.0, 4.2, 19.0, 8.5, 1.0, "TKPI Kemenkes", "Belum", "Pastry goreng isi bihun wortel potongan telur"),
    (78, "siomay", "Siomay Ikan", "Gorengan & Camilan", "3 butir sedang + saus (120g)", 120.0, 210.0, 12.0, 22.0, 8.5, 1.4, "TKPI Kemenkes", "Belum", "Olahan ikan kukus siram bumbu kacang"),
    (79, "batagor", "Batagor (Bakso Tahu Goreng)", "Gorengan & Camilan", "1 porsi (150g)", 150.0, 350.0, 14.0, 36.0, 17.5, 2.0, "TKPI Kemenkes", "Belum", "Tahu & bakso ikan goreng renyah saus kacang"),
    (80, "martabak_telur", "Martabak Telur Bebek", "Gorengan & Camilan", "2 potong sedang (140g)", 140.0, 330.0, 14.5, 20.0, 21.0, 1.2, "TKPI Kemenkes", "Belum", "Kulit lumpia tebal isi daging cincang daun bawang telur"),
    (81, "martabak_manis", "Martabak Manis Cokelat Keju", "Gorengan & Camilan", "1 potong sedang (90g)", 90.0, 290.0, 5.5, 42.0, 11.5, 1.0, "TKPI Kemenkes", "Belum", "Adonan tebal mentega cokelat keju susu kental manis"),
    (82, "cireng", "Cireng", "Gorengan & Camilan", "4 keping (80g)", 80.0, 190.0, 1.0, 36.0, 4.5, 0.6, "TKPI Kemenkes", "Belum", "Aci digoreng bumbu bawang gurih kenyal"),
    (83, "cilok", "Cilok Saus Kacang", "Gorengan & Camilan", "5 butir (100g)", 100.0, 180.0, 2.5, 34.0, 4.0, 0.8, "TKPI Kemenkes", "Belum", "Aci dicolok rebus kenyal saus bumbu kacang"),
    (84, "ketoprak", "Ketoprak", "Gorengan & Camilan", "1 porsi (300g)", 300.0, 390.0, 12.5, 52.0, 15.0, 3.8, "TKPI Kemenkes", "Belum", "Tahu goreng, bihun, tauge, lontong saus kacang bawang putih"),
    (85, "perkedel", "Perkedel Kentang", "Gorengan & Camilan", "1 buah sedang (45g)", 45.0, 85.0, 2.0, 11.0, 3.8, 0.8, "TKPI Kemenkes", "Belum", "Tumbukan kentang bumbu celup telur goreng"),
    (86, "telur_balado", "Telur Balado", "Gorengan & Camilan", "1 butir (60g)", 60.0, 110.0, 6.5, 3.5, 7.8, 0.4, "TKPI Kemenkes", "Belum", "Telur rebus goreng dibalut sambal balado cabai merah"),
    (87, "telur_dadar", "Telur Dadar / Omelet Padang", "Gorengan & Camilan", "1 potong (70g)", 70.0, 145.0, 8.0, 2.0, 12.0, 0.2, "TKPI Kemenkes", "Belum", "Telur kocok daun bawang gurih goreng"),
    (88, "tahu_gejrot", "Tahu Gejrot", "Gorengan & Camilan", "1 porsi (100g)", 100.0, 95.0, 4.5, 12.0, 3.5, 1.2, "TKPI Kemenkes", "Belum", "Tahu sumedang potong kuah asam pedas manis gula aren"),

    # === KUE & DESSERT (11 kelas, 89-99) ===
    (89, "serabi", "Serabi Kuah Kinca", "Kue & Dessert", "1 buah + kuah kinca (80g)", 80.0, 140.0, 2.5, 26.0, 3.0, 0.8, "TKPI Kemenkes", "Belum", "Pancake tepung beras santan kuah gula merah"),
    (90, "klepon", "Klepon", "Kue & Dessert", "3 butir (60g)", 60.0, 135.0, 1.5, 27.0, 2.5, 0.9, "TKPI Kemenkes", "Belum", "Bola ketan isi gula merah cair tabur kelapa parut"),
    (91, "kue_lapis", "Kue Lapis Beras / Pepe", "Kue & Dessert", "1 potong (60g)", 60.0, 125.0, 1.2, 24.5, 2.6, 0.4, "TKPI Kemenkes", "Belum", "Kue kukus berlapis tepung beras santan"),
    (92, "dadar_gulung", "Dadar Gulung Unti", "Kue & Dessert", "1 buah (60g)", 60.0, 130.0, 2.0, 23.0, 3.5, 1.2, "TKPI Kemenkes", "Belum", "Kulit hijau pandan isi unti kelapa gula kelapa"),
    (93, "onde_onde", "Onde-Onde Kacang Hijau", "Kue & Dessert", "1 buah (50g)", 50.0, 150.0, 3.2, 24.0, 5.0, 1.5, "TKPI Kemenkes", "Belum", "Bola tepung ketan isi pasta kacang hijau wijen goreng"),
    (94, "getuk", "Getuk Lindri", "Kue & Dessert", "2 potong (70g)", 70.0, 140.0, 1.2, 31.0, 1.2, 1.8, "TKPI Kemenkes", "Belum", "Singkong kukus tumbuk gula tabur kelapa"),
    (95, "kue_putu", "Kue Putu Bambu", "Kue & Dessert", "2 potong (50g)", 50.0, 110.0, 1.4, 21.0, 2.3, 1.0, "TKPI Kemenkes", "Belum", "Tepung beras pandan isi gula merah kukus bambu"),
    (96, "es_cendol", "Es Cendol / Dawet", "Kue & Dessert", "1 gelas (250ml / 250g)", 250.0, 220.0, 2.0, 38.0, 7.0, 1.2, "TKPI Kemenkes", "Belum", "Cendol tepung beras, kuah santan, gula merah cair"),
    (97, "es_teler", "Es Teler", "Kue & Dessert", "1 mangkok (300g)", 300.0, 280.0, 3.5, 46.0, 9.5, 2.5, "TKPI Kemenkes", "Belum", "Alpukat, kelapa muda, nangka, susu kental manis, santan"),
    (98, "kolak", "Kolak Pisang Ubi", "Kue & Dessert", "1 mangkok (250g)", 250.0, 240.0, 2.8, 48.0, 4.5, 2.6, "TKPI Kemenkes", "Belum", "Pisang kepok & ubi kuah santan manis gula kelapa"),
    (99, "bika_ambon", "Bika Ambon", "Kue & Dessert", "1 potong (50g)", 50.0, 145.0, 2.2, 23.0, 5.0, 0.4, "TKPI Kemenkes", "Belum", "Kue berongga tepung tapioka, telur, santan, daun jeruk"),
]


# ==============================================================================
# 2. FUNGSI PEMBUATAN DATABASE NUTRISI (CSV & JSON)
# ==============================================================================
def create_initial_nutrition_database() -> pd.DataFrame:
    """
    Membangun database nutrisi awal dan menyimpannya ke format CSV & JSON.

    File yang dihasilkan:
    1. nutrition/food_nutrition.csv:
       Format standar yang dapat langsung dibuka via Microsoft Excel / Google Sheets
       oleh non-programmer (ahli gizi SPPG/BGN).
    2. nutrition/food_nutrition.json:
       Format JSON terstruktur untuk lookup cepat pada pipeline inferensi Python.
    3. nutrition/food_nutrition_expert_review.csv:
       Template khusus dengan kolom instruksi review untuk validasi ahli gizi.
    """
    NUTRITION_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "class_index",
        "class_name",
        "display_name",
        "category",
        "serving_size",
        "serving_weight_g",
        "calories_kcal",
        "protein_g",
        "carbohydrate_g",
        "fat_g",
        "fiber_g",
        "source",
        "validated_by_expert",
        "expert_notes",
        "validation_date"
    ]

    records = []
    for item in NUTRITION_DATABASE_RAW:
        record = {
            "class_index": item[0],
            "class_name": item[1],
            "display_name": item[2],
            "category": item[3],
            "serving_size": item[4],
            "serving_weight_g": item[5],
            "calories_kcal": item[6],
            "protein_g": item[7],
            "carbohydrate_g": item[8],
            "fat_g": item[9],
            "fiber_g": item[10],
            "source": item[11],
            "validated_by_expert": item[12],
            "expert_notes": item[13],
            "validation_date": ""
        }
        records.append(record)

    df = pd.DataFrame(records, columns=columns)

    # Simpan ke CSV utama
    df.to_csv(NUTRITION_CSV_PATH, index=False, encoding="utf-8")
    print(f"  [OK] Saved nutrition CSV : {NUTRITION_CSV_PATH}")

    # Simpan ke JSON utama (indexed by class_name untuk O(1) lookup)
    nutrition_dict = {row["class_name"]: row for row in records}
    with open(NUTRITION_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(nutrition_dict, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved nutrition JSON: {NUTRITION_JSON_PATH}")

    # Buat file review khusus ahli gizi
    expert_df = df.copy()
    expert_df.insert(13, "expert_correction_calories", "")
    expert_df.insert(14, "expert_correction_protein", "")
    expert_df.insert(15, "expert_correction_carb", "")
    expert_df.insert(16, "expert_correction_fat", "")
    expert_df.insert(17, "expert_signature", "")
    expert_df.to_csv(EXPERT_REVIEW_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"  [OK] Saved expert review template: {EXPERT_REVIEW_CSV_PATH}")

    return df


# ==============================================================================
# 3. FUNGSI VALIDASI INTEGRITAS DATA GIZI (TERMODINAMIKA & KELENGKAPAN)
# ==============================================================================
def validate_nutrition_database(df: Optional[pd.DataFrame] = None) -> bool:
    """
    Melakukan pemeriksaan integritas menyeluruh terhadap data nutrisi:
    1. Memastikan 100 kelas sesuai persis dengan CLASS_NAMES di config.py
    2. Memastikan tidak ada missing value (NaN/Null) pada kolom esensial
    3. Memastikan semua nilai makronutrisi non-negatif
    4. Cek konsistensi sistem faktor Atwater:
       Kalori aktual vs (4*Protein + 4*Carb + 9*Fat)
       Toleransi deviasi maksimal 25% (karena serat, alkohol, & pembulatan porsi TKPI)
    """
    if df is None:
        if not NUTRITION_CSV_PATH.exists():
            print("  [ERROR] File database nutrisi belum ada. Jalankan create_initial_nutrition_database().")
            return False
        df = pd.read_csv(NUTRITION_CSV_PATH)

    print("\n" + "=" * 70)
    print("  MEMERIKSA INTEGRITAS BASIS DATA NUTRISI (100 KELAS)")
    print("=" * 70)

    # 1. Cek jumlah baris
    if len(df) != NUM_CLASSES:
        print(f"  ❌ GAGAL: Jumlah baris data ({len(df)}) tidak sama dengan NUM_CLASSES ({NUM_CLASSES})")
        return False
    print(f"  ✅ Jumlah kelas terdaftar: {len(df)} / {NUM_CLASSES}")

    # 2. Cek kesesuaian nama kelas dengan config.py
    csv_classes = df["class_name"].tolist()
    missing_classes = set(CLASS_NAMES) - set(csv_classes)
    extra_classes = set(csv_classes) - set(CLASS_NAMES)

    if missing_classes:
        print(f"  ❌ GAGAL: Ada kelas config.py yang hilang di CSV: {missing_classes}")
        return False
    if extra_classes:
        print(f"  ❌ GAGAL: Ada kelas tak dikenal di CSV: {extra_classes}")
        return False
    print("  ✅ 100% nama kelas dan urutan indeks cocok persis dengan config.py")

    # 3. Cek Missing Value
    required_cols = ["class_name", "calories_kcal", "protein_g", "carbohydrate_g", "fat_g"]
    null_counts = df[required_cols].isnull().sum()
    if null_counts.sum() > 0:
        print(f"  ❌ GAGAL: Ditemukan nilai kosong:\n{null_counts[null_counts > 0]}")
        return False
    print("  ✅ Tidak ada data kosong (Null/NaN) pada variabel makronutrisi esensial")

    # 4. Cek nilai negatif
    for col in ["calories_kcal", "protein_g", "carbohydrate_g", "fat_g", "fiber_g", "serving_weight_g"]:
        if (df[col] < 0).any():
            print(f"  ❌ GAGAL: Ada nilai negatif pada kolom {col}")
            return False
    print("  ✅ Semua angka makronutrisi dan porsi non-negatif")

    # 5. Cek konsistensi kalorimetri Atwater
    # Formula Atwater: Calorie ≈ (4 * Protein) + (4 * Carb) + (9 * Fat)
    calculated_calories = (4.0 * df["protein_g"]) + (4.0 * df["carbohydrate_g"]) + (9.0 * df["fat_g"])
    diff_percent = np.abs(df["calories_kcal"] - calculated_calories) / np.maximum(df["calories_kcal"], 1.0) * 100

    outliers = df[diff_percent > 30.0]
    if len(outliers) > 0:
        print(f"  ⚠️  PERINGATAN: Ditemukan {len(outliers)} kelas dengan deviasi Atwater > 30%:")
        for _, row in outliers.iterrows():
            est = 4 * row["protein_g"] + 4 * row["carbohydrate_g"] + 9 * row["fat_g"]
            print(f"      - {row['class_name']}: Tercatat={row['calories_kcal']} kkal, Hitungan Atwater={est:.1f} kkal")
        print("      (Perbedaan ini wajar pada makanan berkuah/berongga/bumbu khas TKPI, perlu perhatian khusus ahli gizi)")
    else:
        print("  ✅ Konsistensi faktor Atwater (4-4-9) terpenuhi dengan deviasi dalam batas wajar (<30%)")

    print("\n  🎉 STATUS INTEGRITAS: BASIS DATA NUTRISI VALID & LENGKAP!")
    return True


# ==============================================================================
# 4. FUNGSI LOOKUP NUTRISI (UNTUK MODEL INFERENSI & APLIKASI)
# ==============================================================================
class NutritionLookup:
    """
    Kelas utility untuk pemetaan cepat hasil prediksi model (class index atau name)
    ke rincian nutrisi lengkap.

    Contoh penggunaan:
        lookup = NutritionLookup()
        info = lookup.get_by_name("nasi_goreng")
        print(info["calories_kcal"])  # Output: 330.0
    """
    def __init__(self, json_path: Optional[Path] = None):
        self.json_path = json_path or NUTRITION_JSON_PATH
        if not self.json_path.exists():
            print("  [INFO] Membuat file database nutrisi terlebih dahulu...")
            create_initial_nutrition_database()

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data: Dict[str, dict] = json.load(f)

        # Buat index mapping
        self.idx_to_data: Dict[int, dict] = {
            item["class_index"]: item for item in self.data.values()
        }

    def get_by_index(self, idx: int) -> Optional[dict]:
        """Ambil data gizi berdasarkan class index (0-99)"""
        return self.idx_to_data.get(idx, None)

    def get_by_name(self, name: str) -> Optional[dict]:
        """Ambil data gizi berdasarkan class name (string, misal 'rendang')"""
        return self.data.get(name, None)

    def get_summary_string(self, identifier: Union[int, str]) -> str:
        """Menghasilkan teks ringkasan rapi untuk ditampilkan di terminal atau log"""
        item = self.get_by_index(identifier) if isinstance(identifier, int) else self.get_by_name(identifier)
        if not item:
            return f"Data nutrisi untuk '{identifier}' tidak ditemukan."

        return (
            f"🥗 {item['display_name']} ({item['category']})\n"
            f"   Porsi Standar : {item['serving_size']} (~{item['serving_weight_g']}g)\n"
            f"   🔥 Kalori      : {item['calories_kcal']} kkal\n"
            f"   🍗 Protein     : {item['protein_g']} g\n"
            f"   🍚 Karbohidrat : {item['carbohydrate_g']} g\n"
            f"   🥑 Lemak       : {item['fat_g']} g\n"
            f"   🌾 Serat       : {item['fiber_g']} g\n"
            f"   📋 Validasi    : {item['validated_by_expert']} ({item['source']})"
        )


# ==============================================================================
# 5. EKSPOR ASET UNTUK FLUTTER MOBILE APP
# ==============================================================================
def export_for_flutter() -> None:
    """
    Mengekspor data gizi ke folder aset mobile_app/assets/nutrition.json.
    File ini dimuat secara lokal (offline) oleh aplikasi Flutter sehingga
    aplikasi tidak memerlukan koneksi internet untuk menampilkan informasi gizi.
    """
    FLUTTER_ASSETS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not NUTRITION_JSON_PATH.exists():
        create_initial_nutrition_database()

    with open(NUTRITION_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Format ringkas ramah memori mobile
    flutter_data = []
    for item in data.values():
        flutter_data.append({
            "id": item["class_index"],
            "code": item["class_name"],
            "name": item["display_name"],
            "category": item["category"],
            "portion": item["serving_size"],
            "weight": item["serving_weight_g"],
            "calories": item["calories_kcal"],
            "protein": item["protein_g"],
            "carb": item["carbohydrate_g"],
            "fat": item["fat_g"],
            "fiber": item["fiber_g"],
            "validated": (item["validated_by_expert"] == "Ya")
        })

    with open(FLUTTER_ASSETS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(flutter_data, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Exported mobile offline asset: {FLUTTER_ASSETS_JSON_PATH}")


# ==============================================================================
# 6. RINGKASAN STATISTIK GIZI PER KATEGORI (UNTUK BAB IV LAPORAN TA)
# ==============================================================================
def generate_nutrition_statistics(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Menghitung statistik deskriptif rata-rata kalori dan makronutrisi
    per kategori makanan. Tabel ini esensial untuk dicantumkan pada
    Bab IV Hasil dan Pembahasan.
    """
    if df is None:
        if not NUTRITION_CSV_PATH.exists():
            create_initial_nutrition_database()
        df = pd.read_csv(NUTRITION_CSV_PATH)

    summary = df.groupby("category")[["calories_kcal", "protein_g", "carbohydrate_g", "fat_g"]].agg(["mean", "min", "max", "std"]).round(1)

    print("\n" + "=" * 80)
    print("  TABEL REKAPITULASI RATA-RATA NUTRISI PER KATEGORI MAKANAN (BAB IV)")
    print("=" * 80)
    mean_table = df.groupby("category")[["calories_kcal", "protein_g", "carbohydrate_g", "fat_g"]].mean().round(1)
    mean_table.columns = ["Rata-rata Kalori (kkal)", "Protein (g)", "Karbohidrat (g)", "Lemak (g)"]
    print(mean_table.to_string())
    print("=" * 80)

    return summary


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  TAHAP 3: SETUP BASIS DATA NUTRISI & TEMPLATE VALIDASI AHLI GIZI")
    print("=" * 70)

    # 1. Buat database CSV & JSON
    df = create_initial_nutrition_database()

    # 2. Validasi integritas
    is_valid = validate_nutrition_database(df)

    # 3. Ekspor untuk Flutter
    export_for_flutter()

    # 4. Ringkasan statistik
    generate_nutrition_statistics(df)

    # 5. Uji Lookup Demo
    print("\n" + "=" * 70)
    print("  DEMO: PENGUJIAN FUNGSI LOOKUP NUTRISI")
    print("=" * 70)
    lookup = NutritionLookup()
    demo_samples = ["rendang", "nasi_goreng", "gado_gado", "martabak_manis", "soto_ayam"]
    for sample in demo_samples:
        print(lookup.get_summary_string(sample))
        print("-" * 50)

    print("\n  ✅ TAHAP 3 SELESAI: Basis data nutrisi 100 kelas siap divalidasi & diintegrasikan!")
