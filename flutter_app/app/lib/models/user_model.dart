import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class UserModel {
  final String id;
  final String name;
  final String educationLevel;
  int xp;
  int mathCoins;
  int consecutiveDays;
  List<String> achievements;

  static const String _storageKey = 'user_data';

  UserModel({
    required this.id,
    required this.name,
    required this.educationLevel,
    this.xp = 0,
    this.mathCoins = 0,
    this.consecutiveDays = 0,
    this.achievements = const [],
  });

  // Convert to Map for SharedPreferences
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'name': name,
      'educationLevel': educationLevel,
      'xp': xp,
      'mathCoins': mathCoins,
      'consecutiveDays': consecutiveDays,
      'achievements': achievements,
    };
  }

  // Convert to JSON string
  String toJson() => json.encode(toMap());

  // Create from Map
  factory UserModel.fromMap(Map<String, dynamic> map) {
    return UserModel(
      id: map['id'],
      name: map['name'],
      educationLevel: map['educationLevel'],
      xp: map['xp'] ?? 0,
      mathCoins: map['mathCoins'] ?? 0,
      consecutiveDays: map['consecutiveDays'] ?? 0,
      achievements: List<String>.from(map['achievements'] ?? []),
    );
  }

  // Create from JSON string
  factory UserModel.fromJson(String jsonString) {
    return UserModel.fromMap(json.decode(jsonString));
  }

  // Save user data to SharedPreferences
  Future<void> save() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_storageKey, toJson());
  }

  // Load user data from SharedPreferences
  static Future<UserModel?> load() async {
    final prefs = await SharedPreferences.getInstance();
    final String? jsonString = prefs.getString(_storageKey);
    if (jsonString == null) return null;
    return UserModel.fromJson(jsonString);
  }

  // Delete user data from SharedPreferences
  static Future<void> delete() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_storageKey);
  }

  // Update user data
  Future<void> update({
    int? xp,
    int? mathCoins,
    int? consecutiveDays,
    List<String>? achievements,
  }) async {
    if (xp != null) this.xp = xp;
    if (mathCoins != null) this.mathCoins = mathCoins;
    if (consecutiveDays != null) this.consecutiveDays = consecutiveDays;
    if (achievements != null) this.achievements = achievements;
    await save();
  }
}