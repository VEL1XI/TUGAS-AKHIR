// ==============================================================================
// nutrition_service.dart — LAYANAN BASIS DATA GIZI OFFLINE
// ==============================================================================

import 'dart:convert';
import 'package:flutter/services.dart';
import '../models/food_item.dart';

class NutritionService {
  static final NutritionService _instance = NutritionService._internal();
  factory NutritionService() => _instance;
  NutritionService._internal();

  List<FoodNutritionItem> _items = [];
  Map<int, FoodNutritionItem> _itemsById = {};
  Map<String, FoodNutritionItem> _itemsByCode = {};
  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;

  /// Memuat basis data nutrisi dari file aset lokal secara offline
  Future<void> loadDatabase() async {
    if (_isLoaded) return;

    try {
      final String jsonString = await rootBundle.loadString('assets/nutrition.json');
      final List<dynamic> jsonList = json.decode(jsonString);

      _items = jsonList.map((j) => FoodNutritionItem.fromJson(j)).toList();
      _itemsById = {for (var item in _items) item.id: item};
      _itemsByCode = {for (var item in _items) item.code: item};
      _isLoaded = true;
    } catch (e) {
      print("[NutritionService] Gagal memuat database gizi: $e");
    }
  }

  FoodNutritionItem? getItemById(int id) => _itemsById[id];
  FoodNutritionItem? getItemByCode(String code) => _itemsByCode[code];
  List<FoodNutritionItem> getAllItems() => List.unmodifiable(_items);

  List<FoodNutritionItem> getItemsByCategory(String category) {
    return _items.where((i) => i.category.toLowerCase() == category.toLowerCase()).toList();
  }
}
