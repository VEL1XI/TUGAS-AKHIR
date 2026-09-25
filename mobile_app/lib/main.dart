// ==============================================================================
// main.dart — ENTRY POINT APLIKASI FLUTTER NUTRISI NUSANTARA AI
// ==============================================================================
// Titik masuk utama aplikasi. Menginisialisasi:
//   1. Orientasi layar terkunci ke portrait (optimal untuk scanning kamera)
//   2. Status bar transparan untuk estetika modern
//   3. Tema Material Design 3 dengan palet warna kustom
//   4. Routing ke SplashScreen → HomeScreen → ResultScreen
//
// Argumen Sidang:
//   Mengapa Flutter dipilih untuk aplikasi mobile?
//   - Single codebase untuk Android & iOS (efisiensi waktu pengembangan TA)
//   - Widget rendering engine Skia memungkinkan animasi UI 60fps
//   - ONNX Runtime Flutter plugin mendukung inferensi model SOTA (ConvNeXt V2)
//     tanpa perlu menulis bridge code native di Java/Swift
// ==============================================================================

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screens/splash_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Kunci orientasi portrait untuk kenyamanan pemindaian kamera
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  // Styling status bar sistem — transparan untuk immersive experience
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: Color(0xFF0F172A),
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );

  runApp(const IndonesianFoodNutritionApp());
}

/// Widget root aplikasi — mendefinisikan MaterialApp dengan tema global
class IndonesianFoodNutritionApp extends StatelessWidget {
  const IndonesianFoodNutritionApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "Nutrisi Nusantara AI",
      debugShowCheckedModeBanner: false,

      // ================================================================
      // TEMA MATERIAL DESIGN 3 — PALET WARNA KUSTOM
      // ================================================================
      // Warna primer: Deep Sky Blue (#0284C7) — profesional, trust, teknologi
      // Warna sekunder: Emerald (#10B981) — kesehatan, gizi, organik
      // Background: Slate (#F8FAFC) — clean, modern, tidak melelahkan mata
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0284C7),
          primary: const Color(0xFF0284C7),
          secondary: const Color(0xFF10B981),
          surface: Colors.white,
          brightness: Brightness.light,
        ),
        textTheme: GoogleFonts.poppinsTextTheme(
          Theme.of(context).textTheme,
        ),
        scaffoldBackgroundColor: const Color(0xFFF8FAFC),
        appBarTheme: const AppBarTheme(
          surfaceTintColor: Colors.transparent,
          backgroundColor: Colors.white,
          elevation: 0,
        ),
        cardTheme: CardTheme(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 24),
          ),
        ),
      ),
      home: const SplashScreen(),
    );
  }
}
