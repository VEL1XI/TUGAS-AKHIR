// ==============================================================================
// result_screen.dart — TAMPILAN RINCIAN KLASIFIKASI & ESTIMASI GIZI LENGKAP
// ==============================================================================
// Menampilkan hasil inferensi model ConvNeXt V2-Nano secara komprehensif:
//   1. Preview foto makanan dengan overlay confidence badge
//   2. Detail nama makanan, kategori, dan level keyakinan AI
//   3. Grid 4 makronutrisi utama (Kalori, Protein, Karbo, Lemak)
//   4. Grafik pie chart distribusi makronutrisi (via fl_chart)
//   5. Progress bar kontribusi terhadap AKG harian (2.150 kkal)
//   6. Top-3 kandidat alternatif (untuk fine-grained correction)
//   7. Status validasi ahli gizi (SPPG / BGN)
//   8. Tombol berbagi hasil analisis
//
// Argumen Sidang:
//   - Pie chart distribusi makro menunjukkan komposisi energi seimbang/tidak
//   - AKG bar memberikan konteks persentase kontribusi per porsi terhadap kebutuhan harian
//   - Top-3 candidates penting karena fine-grained visual similarity antar kelas kuliner
// ==============================================================================

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:percent_indicator/linear_percent_indicator.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:share_plus/share_plus.dart';
import '../models/food_item.dart';
import '../services/classifier_service.dart';

class ResultScreen extends StatelessWidget {
  final File imageFile;
  final PredictionResult result;

  const ResultScreen({
    Key? key,
    required this.imageFile,
    required this.result,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final food = result.food;

    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      body: CustomScrollView(
        physics: const BouncingScrollPhysics(),
        slivers: [
          // === 1. SLIVER APP BAR DENGAN FOTO MAKANAN ===
          SliverAppBar(
            expandedHeight: 280,
            pinned: true,
            stretch: true,
            backgroundColor: const Color(0xFF0F172A),
            foregroundColor: Colors.white,
            leading: IconButton(
              icon: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.3),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.arrow_back_rounded, size: 20),
              ),
              onPressed: () => Navigator.pop(context),
            ),
            actions: [
              // Tombol share
              IconButton(
                icon: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.3),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(Icons.share_rounded, size: 20),
                ),
                onPressed: () => _shareResult(context, food),
              ),
            ],
            flexibleSpace: FlexibleSpaceBar(
              background: Stack(
                fit: StackFit.expand,
                children: [
                  // Foto makanan
                  Image.file(imageFile, fit: BoxFit.cover),

                  // Gradient overlay
                  Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [
                          Colors.transparent,
                          Colors.black.withOpacity(0.2),
                          Colors.black.withOpacity(0.8),
                        ],
                        stops: const [0.3, 0.6, 1.0],
                      ),
                    ),
                  ),

                  // Info makanan di bawah foto
                  Positioned(
                    bottom: 16,
                    left: 16,
                    right: 16,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Badge kategori
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: const Color(0xFF0EA5E9),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(
                            food.category,
                            style: GoogleFonts.inter(
                              color: Colors.white,
                              fontSize: 11,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                        const SizedBox(height: 8),

                        // Nama makanan
                        Text(
                          food.name,
                          style: GoogleFonts.poppins(
                            color: Colors.white,
                            fontSize: 28,
                            fontWeight: FontWeight.bold,
                            height: 1.1,
                          ),
                        ),
                        const SizedBox(height: 6),

                        // Badge confidence & latency
                        Row(
                          children: [
                            _buildInfoBadge(
                              icon: Icons.psychology_rounded,
                              text:
                                  "${result.confidencePercent.toStringAsFixed(1)}% Confidence",
                              color: _getConfidenceColor(
                                  result.confidencePercent),
                            ),
                            const SizedBox(width: 8),
                            _buildInfoBadge(
                              icon: Icons.speed_rounded,
                              text:
                                  "${result.inferenceTimeMs.toStringAsFixed(0)} ms",
                              color: const Color(0xFF10B981),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),

          // === BODY CONTENT ===
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // === 2. KARTU TAKARAN PORSI STANDAR ===
                  _buildPortionCard(food),
                  const SizedBox(height: 20),

                  // === 3. GRID 4 MAKRONUTRISI UTAMA ===
                  _buildSectionTitle("Kandungan Nutrisi Utama",
                      Icons.science_rounded),
                  const SizedBox(height: 12),
                  _buildMacroGrid(food),
                  const SizedBox(height: 24),

                  // === 4. PIE CHART DISTRIBUSI MAKRONUTRISI ===
                  _buildSectionTitle(
                      "Distribusi Energi Makronutrien", Icons.pie_chart_rounded),
                  const SizedBox(height: 12),
                  _buildNutritionPieChart(food),
                  const SizedBox(height: 24),

                  // === 5. KONTRIBUSI TERHADAP AKG HARIAN ===
                  _buildAkgCard(food),
                  const SizedBox(height: 24),

                  // === 6. TOP-3 KANDIDAT ALTERNATIF ===
                  if (result.topCandidates.length > 1) ...[
                    _buildSectionTitle("Kandidat Alternatif (Top-3)",
                        Icons.format_list_numbered_rounded),
                    const SizedBox(height: 8),
                    Text(
                      "Hasil prediksi kelas lain dengan probabilitas tertinggi — berguna jika AI salah mengidentifikasi pada kelas yang mirip secara visual.",
                      style: GoogleFonts.inter(
                        fontSize: 12,
                        color: const Color(0xFF94A3B8),
                        height: 1.4,
                      ),
                    ),
                    const SizedBox(height: 12),
                    ...result.topCandidates.asMap().entries.map(
                        (entry) => _buildCandidateCard(entry.key, entry.value)),
                    const SizedBox(height: 24),
                  ],

                  // === 7. STATUS VALIDASI AHLI GIZI ===
                  _buildValidationCard(food),
                  const SizedBox(height: 30),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ============================================================================
  // WIDGET BUILDERS
  // ============================================================================

  Widget _buildInfoBadge({
    required IconData icon,
    required String text,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.2),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 13, color: color),
          const SizedBox(width: 4),
          Text(
            text,
            style: GoogleFonts.inter(
              color: Colors.white,
              fontSize: 11,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionTitle(String title, IconData icon) {
    return Row(
      children: [
        Icon(icon, size: 20, color: const Color(0xFF0284C7)),
        const SizedBox(width: 8),
        Text(
          title,
          style: GoogleFonts.poppins(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: const Color(0xFF1E293B),
          ),
        ),
      ],
    );
  }

  Widget _buildPortionCard(FoodNutritionItem food) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [Color(0xFFFEF3C7), Color(0xFFFDE68A)],
              ),
              borderRadius: BorderRadius.circular(14),
            ),
            child: const Icon(Icons.restaurant_rounded,
                color: Color(0xFFD97706), size: 24),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Takaran Porsi Standar",
                  style: GoogleFonts.inter(
                    color: const Color(0xFF64748B),
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  "${food.portion} (~${food.weight.toInt()}g)",
                  style: GoogleFonts.poppins(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                    color: const Color(0xFF1E293B),
                  ),
                ),
              ],
            ),
          ),
          // Kalori besar
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                "${food.calories.toInt()}",
                style: GoogleFonts.poppins(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: const Color(0xFFEF4444),
                ),
              ),
              Text(
                "kkal",
                style: GoogleFonts.inter(
                  fontSize: 12,
                  color: const Color(0xFF94A3B8),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMacroGrid(FoodNutritionItem food) {
    return Column(
      children: [
        Row(
          children: [
            _buildMacroCard(
              title: "Kalori",
              value: "${food.calories.toInt()}",
              unit: "kkal",
              color: const Color(0xFFEF4444),
              bgColor: const Color(0xFFFEE2E2),
              icon: Icons.local_fire_department_rounded,
            ),
            const SizedBox(width: 12),
            _buildMacroCard(
              title: "Protein",
              value: food.protein.toStringAsFixed(1),
              unit: "gram",
              color: const Color(0xFF3B82F6),
              bgColor: const Color(0xFFDBEAFE),
              icon: Icons.fitness_center_rounded,
            ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            _buildMacroCard(
              title: "Karbohidrat",
              value: food.carb.toStringAsFixed(1),
              unit: "gram",
              color: const Color(0xFFF59E0B),
              bgColor: const Color(0xFFFEF3C7),
              icon: Icons.grain_rounded,
            ),
            const SizedBox(width: 12),
            _buildMacroCard(
              title: "Lemak",
              value: food.fat.toStringAsFixed(1),
              unit: "gram",
              color: const Color(0xFF10B981),
              bgColor: const Color(0xFFD1FAE5),
              icon: Icons.water_drop_rounded,
            ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            _buildMacroCard(
              title: "Serat",
              value: food.fiber.toStringAsFixed(1),
              unit: "gram",
              color: const Color(0xFF8B5CF6),
              bgColor: const Color(0xFFEDE9FE),
              icon: Icons.eco_rounded,
            ),
            const SizedBox(width: 12),
            // Berat porsi
            Expanded(
              child: Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.04),
                      blurRadius: 8,
                      offset: const Offset(0, 3),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text("Berat",
                            style: GoogleFonts.inter(
                                color: Colors.grey, fontSize: 12)),
                        Container(
                          padding: const EdgeInsets.all(6),
                          decoration: const BoxDecoration(
                            color: Color(0xFFF1F5F9),
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(Icons.scale_rounded,
                              color: Color(0xFF64748B), size: 16),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.baseline,
                      textBaseline: TextBaseline.alphabetic,
                      children: [
                        Text(
                          "${food.weight.toInt()}",
                          style: GoogleFonts.poppins(
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                            color: const Color(0xFF475569),
                          ),
                        ),
                        const SizedBox(width: 4),
                        Text("gram",
                            style: GoogleFonts.inter(
                                color: Colors.grey, fontSize: 11)),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildMacroCard({
    required String title,
    required String value,
    required String unit,
    required Color color,
    required Color bgColor,
    required IconData icon,
  }) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 8,
              offset: const Offset(0, 3),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(title,
                    style: GoogleFonts.inter(color: Colors.grey, fontSize: 12)),
                Container(
                  padding: const EdgeInsets.all(6),
                  decoration: BoxDecoration(color: bgColor, shape: BoxShape.circle),
                  child: Icon(icon, color: color, size: 16),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              crossAxisAlignment: CrossAxisAlignment.baseline,
              textBaseline: TextBaseline.alphabetic,
              children: [
                Text(
                  value,
                  style: GoogleFonts.poppins(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: color,
                  ),
                ),
                const SizedBox(width: 4),
                Text(unit,
                    style:
                        GoogleFonts.inter(color: Colors.grey, fontSize: 11)),
              ],
            ),
          ],
        ),
      ),
    );
  }

  /// Pie chart distribusi makronutrisi (protein, karbo, lemak)
  Widget _buildNutritionPieChart(FoodNutritionItem food) {
    // Kalori dari makronutrisi: Protein=4kkal/g, Karbo=4kkal/g, Lemak=9kkal/g
    final proteinCal = food.protein * 4.0;
    final carbCal = food.carb * 4.0;
    final fatCal = food.fat * 9.0;
    final totalMacroCal = proteinCal + carbCal + fatCal;

    final proteinPct =
        totalMacroCal > 0 ? (proteinCal / totalMacroCal * 100) : 0.0;
    final carbPct =
        totalMacroCal > 0 ? (carbCal / totalMacroCal * 100) : 0.0;
    final fatPct =
        totalMacroCal > 0 ? (fatCal / totalMacroCal * 100) : 0.0;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          SizedBox(
            height: 180,
            child: PieChart(
              PieChartData(
                sectionsSpace: 3,
                centerSpaceRadius: 36,
                sections: [
                  PieChartSectionData(
                    value: proteinPct,
                    color: const Color(0xFF3B82F6),
                    title: "${proteinPct.toStringAsFixed(0)}%",
                    radius: 48,
                    titleStyle: GoogleFonts.inter(
                      color: Colors.white,
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  PieChartSectionData(
                    value: carbPct,
                    color: const Color(0xFFF59E0B),
                    title: "${carbPct.toStringAsFixed(0)}%",
                    radius: 48,
                    titleStyle: GoogleFonts.inter(
                      color: Colors.white,
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  PieChartSectionData(
                    value: fatPct,
                    color: const Color(0xFF10B981),
                    title: "${fatPct.toStringAsFixed(0)}%",
                    radius: 48,
                    titleStyle: GoogleFonts.inter(
                      color: Colors.white,
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          // Legend
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _buildLegendItem("Protein", const Color(0xFF3B82F6),
                  "${proteinCal.toInt()} kkal"),
              _buildLegendItem("Karbohidrat", const Color(0xFFF59E0B),
                  "${carbCal.toInt()} kkal"),
              _buildLegendItem("Lemak", const Color(0xFF10B981),
                  "${fatCal.toInt()} kkal"),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            "Kontribusi energi: Protein 4 kkal/g · Karbohidrat 4 kkal/g · Lemak 9 kkal/g",
            textAlign: TextAlign.center,
            style: GoogleFonts.inter(
              fontSize: 10,
              color: const Color(0xFFCBD5E1),
              fontStyle: FontStyle.italic,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLegendItem(String label, Color color, String value) {
    return Column(
      children: [
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 10,
              height: 10,
              decoration:
                  BoxDecoration(color: color, borderRadius: BorderRadius.circular(3)),
            ),
            const SizedBox(width: 4),
            Text(label,
                style: GoogleFonts.inter(
                    fontSize: 11, color: const Color(0xFF64748B))),
          ],
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: GoogleFonts.inter(
            fontSize: 12,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ],
    );
  }

  Widget _buildAkgCard(FoodNutritionItem food) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.assessment_rounded,
                  size: 20, color: Color(0xFF0284C7)),
              const SizedBox(width: 8),
              Text(
                "Kontribusi AKG Harian",
                style: GoogleFonts.poppins(
                    fontWeight: FontWeight.bold, fontSize: 15),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            "Persentase terhadap Angka Kecukupan Gizi harian (AKG 2.150 kkal)",
            style: GoogleFonts.inter(
              fontSize: 11,
              color: const Color(0xFF94A3B8),
            ),
          ),
          const SizedBox(height: 16),
          _buildAkgBar("Energi (Kalori)", food.akgCaloriesPercent,
              const Color(0xFFEF4444), "${food.calories.toInt()} / 2.150 kkal"),
          _buildAkgBar("Protein", food.akgProteinPercent,
              const Color(0xFF3B82F6), "${food.protein.toStringAsFixed(1)} / 60g"),
          _buildAkgBar("Karbohidrat", food.akgCarbPercent,
              const Color(0xFFF59E0B), "${food.carb.toStringAsFixed(1)} / 340g"),
          _buildAkgBar("Lemak", food.akgFatPercent,
              const Color(0xFF10B981), "${food.fat.toStringAsFixed(1)} / 67g"),
        ],
      ),
    );
  }

  Widget _buildAkgBar(
      String label, double percent, Color color, String detail) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(label,
                  style: GoogleFonts.inter(
                      fontSize: 12, color: const Color(0xFF475569))),
              Row(
                children: [
                  Text(
                    detail,
                    style: GoogleFonts.inter(
                        fontSize: 10, color: const Color(0xFF94A3B8)),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: color.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Text(
                      "${(percent * 100).toInt()}%",
                      style: GoogleFonts.inter(
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                        color: color,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 6),
          LinearPercentIndicator(
            lineHeight: 8.0,
            percent: percent.clamp(0.0, 1.0),
            progressColor: color,
            backgroundColor: const Color(0xFFF1F5F9),
            barRadius: const Radius.circular(4),
            padding: EdgeInsets.zero,
            animation: true,
            animationDuration: 800,
          ),
        ],
      ),
    );
  }

  Widget _buildCandidateCard(int rank, FoodNutritionItem item) {
    final Color rankColor = rank == 0
        ? const Color(0xFFF59E0B)
        : rank == 1
            ? const Color(0xFF94A3B8)
            : const Color(0xFFCD7F32);
    final String medalEmoji = rank == 0 ? "🥇" : rank == 1 ? "🥈" : "🥉";

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: rank == 0 ? const Color(0xFFFEFCE8) : Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: rank == 0
            ? Border.all(color: const Color(0xFFFDE68A), width: 1.5)
            : Border.all(color: const Color(0xFFF1F5F9)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.02),
            blurRadius: 6,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          Text(medalEmoji, style: const TextStyle(fontSize: 24)),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  item.name,
                  style: GoogleFonts.poppins(
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                    color: const Color(0xFF1E293B),
                  ),
                ),
                Text(
                  "${item.category} · ${item.calories.toInt()} kkal",
                  style: GoogleFonts.inter(
                    fontSize: 11,
                    color: const Color(0xFF94A3B8),
                  ),
                ),
              ],
            ),
          ),
          // Confidence bar
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: rankColor.withOpacity(0.1),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              "${(item.confidence ?? 0).toStringAsFixed(1)}%",
              style: GoogleFonts.inter(
                fontWeight: FontWeight.bold,
                fontSize: 12,
                color: rankColor,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildValidationCard(FoodNutritionItem food) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: food.validated
            ? const Color(0xFFF0FDF4)
            : const Color(0xFFFFFBEB),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: food.validated
              ? const Color(0xFFBBF7D0)
              : const Color(0xFFFDE68A),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            food.validated
                ? Icons.verified_rounded
                : Icons.pending_rounded,
            color: food.validated
                ? const Color(0xFF16A34A)
                : const Color(0xFFD97706),
            size: 28,
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  food.validated
                      ? "Data Gizi Tervalidasi"
                      : "Menunggu Validasi Ahli Gizi",
                  style: GoogleFonts.poppins(
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                    color: food.validated
                        ? const Color(0xFF166534)
                        : const Color(0xFF92400E),
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  food.validated
                      ? "Nilai nutrisi telah diverifikasi oleh Ahli Gizi SPPG / Badan Gizi Nasional sesuai referensi TKPI 2020 Kemenkes RI."
                      : "Nilai nutrisi bersumber dari TKPI 2020 dan sedang dalam proses verifikasi oleh tim ahli gizi mitra.",
                  style: GoogleFonts.inter(
                    fontSize: 12,
                    height: 1.4,
                    color: food.validated
                        ? const Color(0xFF15803D)
                        : const Color(0xFFB45309),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 80) return const Color(0xFF10B981);
    if (confidence >= 60) return const Color(0xFFF59E0B);
    return const Color(0xFFEF4444);
  }

  void _shareResult(BuildContext context, FoodNutritionItem food) {
    final shareText = "🍽️ Hasil Analisis Nutrisi Nusantara AI\n\n"
        "Makanan: ${food.name}\n"
        "Kategori: ${food.category}\n"
        "Porsi: ${food.portion} (~${food.weight.toInt()}g)\n\n"
        "📊 Kandungan Nutrisi:\n"
        "🔥 Kalori: ${food.calories.toInt()} kkal\n"
        "💪 Protein: ${food.protein.toStringAsFixed(1)}g\n"
        "🌾 Karbohidrat: ${food.carb.toStringAsFixed(1)}g\n"
        "💧 Lemak: ${food.fat.toStringAsFixed(1)}g\n"
        "🌿 Serat: ${food.fiber.toStringAsFixed(1)}g\n\n"
        "Confidence: ${result.confidencePercent.toStringAsFixed(1)}%\n"
        "Model: ConvNeXt V2-Nano SOTA CNN\n"
        "Sumber: TKPI 2020 Kemenkes RI";

    Share.share(shareText, subject: "Analisis Nutrisi: ${food.name}");
  }
}
