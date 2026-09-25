// ==============================================================================
// splash_screen.dart — LAYAR SPLASH ANIMASI INISIALISASI AI ENGINE
// ==============================================================================
// Splash screen ini menjalankan inisialisasi berat secara asinkron:
//   1. Memuat basis data nutrisi 100 kelas dari JSON asset
//   2. Memuat model ONNX ConvNeXt V2-Nano ke memori GPU/NPU
//   3. Menampilkan progress indikator dan logo aplikasi
//
// Durasi splash ditentukan oleh waktu loading model sesungguhnya,
// bukan timer artifisial — ini penting untuk UX yang jujur.
// ==============================================================================

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/classifier_service.dart';
import '../services/nutrition_service.dart';
import 'home_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({Key? key}) : super(key: key);

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  late Animation<double> _scaleAnimation;

  String _statusText = "Menginisialisasi sistem...";
  double _progress = 0.0;

  @override
  void initState() {
    super.initState();

    // Animasi fade-in dan scale untuk logo
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    );

    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );

    _scaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.elasticOut),
    );

    _controller.forward();

    // Mulai loading asinkron setelah frame pertama selesai render
    WidgetsBinding.instance.addPostFrameCallback((_) => _initializeApp());
  }

  Future<void> _initializeApp() async {
    try {
      // Fase 1: Muat basis data nutrisi
      setState(() {
        _statusText = "Memuat basis data 100 kelas nutrisi...";
        _progress = 0.3;
      });
      await NutritionService().loadDatabase();
      await Future.delayed(const Duration(milliseconds: 300));

      // Fase 2: Muat model AI
      setState(() {
        _statusText = "Memuat model ConvNeXt V2-Nano ONNX...";
        _progress = 0.7;
      });
      await ClassifierService().initializeModel();
      await Future.delayed(const Duration(milliseconds: 300));

      // Fase 3: Finalisasi
      setState(() {
        _statusText = "Sistem siap digunakan ✓";
        _progress = 1.0;
      });
      await Future.delayed(const Duration(milliseconds: 500));

      // Navigasi ke HomeScreen
      if (mounted) {
        Navigator.of(context).pushReplacement(
          PageRouteBuilder(
            pageBuilder: (context, animation, secondaryAnimation) =>
                const HomeScreen(),
            transitionsBuilder:
                (context, animation, secondaryAnimation, child) {
              return FadeTransition(opacity: animation, child: child);
            },
            transitionDuration: const Duration(milliseconds: 600),
          ),
        );
      }
    } catch (e) {
      setState(() {
        _statusText = "Peringatan: $e\nMelanjutkan dalam mode terbatas...";
        _progress = 1.0;
      });
      await Future.delayed(const Duration(seconds: 2));
      if (mounted) {
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(builder: (context) => const HomeScreen()),
        );
      }
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          // Gradient gelap premium untuk splash
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF0F172A), // Slate 900
              Color(0xFF1E293B), // Slate 800
              Color(0xFF0C4A6E), // Sky 900
            ],
            stops: [0.0, 0.5, 1.0],
          ),
        ),
        child: SafeArea(
          child: FadeTransition(
            opacity: _fadeAnimation,
            child: ScaleTransition(
              scale: _scaleAnimation,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Spacer(flex: 3),

                  // === LOGO IKON UTAMA ===
                  Container(
                    width: 100,
                    height: 100,
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                        colors: [Color(0xFF0EA5E9), Color(0xFF0284C7)],
                      ),
                      borderRadius: BorderRadius.circular(28),
                      boxShadow: [
                        BoxShadow(
                          color: const Color(0xFF0EA5E9).withOpacity(0.4),
                          blurRadius: 30,
                          offset: const Offset(0, 10),
                        ),
                      ],
                    ),
                    child: const Icon(
                      Icons.lunch_dining_rounded,
                      size: 52,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 24),

                  // === NAMA APLIKASI ===
                  Text(
                    "Nutrisi Nusantara",
                    style: GoogleFonts.poppins(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                      letterSpacing: -0.5,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
                    decoration: BoxDecoration(
                      border: Border.all(
                          color: const Color(0xFF0EA5E9).withOpacity(0.5)),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      "⚡ Powered by ConvNeXt V2 SOTA CNN",
                      style: GoogleFonts.inter(
                        fontSize: 11,
                        color: const Color(0xFF38BDF8),
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),

                  const Spacer(flex: 2),

                  // === PROGRESS BAR ===
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 48),
                    child: Column(
                      children: [
                        ClipRRect(
                          borderRadius: BorderRadius.circular(4),
                          child: LinearProgressIndicator(
                            value: _progress,
                            minHeight: 4,
                            backgroundColor: Colors.white.withOpacity(0.1),
                            valueColor: const AlwaysStoppedAnimation<Color>(
                              Color(0xFF0EA5E9),
                            ),
                          ),
                        ),
                        const SizedBox(height: 16),
                        Text(
                          _statusText,
                          textAlign: TextAlign.center,
                          style: GoogleFonts.inter(
                            fontSize: 12,
                            color: Colors.white54,
                            height: 1.4,
                          ),
                        ),
                      ],
                    ),
                  ),

                  const Spacer(flex: 1),

                  // === FOOTER: INFO TUGAS AKHIR ===
                  Padding(
                    padding: const EdgeInsets.only(bottom: 24),
                    child: Column(
                      children: [
                        Text(
                          "Tugas Akhir S1 Informatika",
                          style: GoogleFonts.inter(
                            fontSize: 11,
                            color: Colors.white30,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          "Klasifikasi & Estimasi Nutrisi Berbasis CNN",
                          style: GoogleFonts.inter(
                            fontSize: 10,
                            color: Colors.white20,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
