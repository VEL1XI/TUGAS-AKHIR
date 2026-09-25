// ==============================================================================
// food_item.dart — MODEL DATA MAKANAN & RINCIAN NUTRISI LENGKAP
// ==============================================================================

class FoodNutritionItem {
  final int id;
  final String code;
  final String name;
  final String category;
  final String portion;
  final double weight;
  final double calories;
  final double protein;
  final double carb;
  final double fat;
  final double fiber;
  final bool validated;
  double? confidence;

  FoodNutritionItem({
    required this.id,
    required this.code,
    required this.name,
    required this.category,
    required this.portion,
    required this.weight,
    required this.calories,
    required this.protein,
    required this.carb,
    required this.fat,
    required this.fiber,
    required this.validated,
    this.confidence,
  });

  factory FoodNutritionItem.fromJson(Map<String, dynamic> json) {
    return FoodNutritionItem(
      id: json['id'] as int,
      code: json['code'] as String,
      name: json['name'] as String,
      category: json['category'] as String,
      portion: json['portion'] as String,
      weight: (json['weight'] as num).toDouble(),
      calories: (json['calories'] as num).toDouble(),
      protein: (json['protein'] as num).toDouble(),
      carb: (json['carb'] as num).toDouble(),
      fat: (json['fat'] as num).toDouble(),
      fiber: (json['fiber'] as num).toDouble(),
      validated: json['validated'] as bool? ?? false,
      confidence: json['confidence'] != null ? (json['confidence'] as num).toDouble() : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'code': code,
      'name': name,
      'category': category,
      'portion': portion,
      'weight': weight,
      'calories': calories,
      'protein': protein,
      'carb': carb,
      'fat': fat,
      'fiber': fiber,
      'validated': validated,
      'confidence': confidence,
    };
  }

  // Persentase kontribusi terhadap Angka Kecukupan Gizi (AKG) Harian 2150 kkal
  double get akgCaloriesPercent => (calories / 2150.0).clamp(0.0, 1.0);
  double get akgProteinPercent => (protein / 60.0).clamp(0.0, 1.0);
  double get akgCarbPercent => (carb / 340.0).clamp(0.0, 1.0);
  double get akgFatPercent => (fat / 67.0).clamp(0.0, 1.0);
}
