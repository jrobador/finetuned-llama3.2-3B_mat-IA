import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/user_model.dart';
import 'dart:convert';

class UserProvider with ChangeNotifier {
  UserModel? _currentUser;
  final SharedPreferences _prefs;

  UserProvider(this._prefs);

  // Getters
  UserModel? get currentUser => _currentUser;
  bool get isUserLoggedIn => _currentUser != null;

  // Save user to SharedPreferences
  Future<void> _saveUserToPreferences(String username) async {
    if (_currentUser != null) {
      await _prefs.setString('user_$username', jsonEncode(_currentUser!.toMap()));
    }
  }

  // Create new user
  Future<void> createUser({
    required String name, 
    required String educationLevel,
  }) async {
    try {
      // Generate unique ID
      String userId = DateTime.now().millisecondsSinceEpoch.toString();

      // Create user model
      UserModel newUser = UserModel(
        id: userId,
        name: name,
        educationLevel: educationLevel,
        xp: 0,
        mathCoins: 50, // Initial bonus coins
        consecutiveDays: 0,
        achievements: [],
      );

      // Set as current user and save locally
      _currentUser = newUser;
      await _saveUserToPreferences(name);
      notifyListeners();
    } catch (e) {
      print('Error creating user: $e');
    }
  }

  // Load existing user
  Future<bool> loadUser(String username) async {
    try {
      String? userJson = _prefs.getString('user_$username');
      if (userJson == null) {
        print('No saved data found for user: $username');
        return false;
      }
  
      Map<String, dynamic> userData = jsonDecode(userJson);
      _currentUser = UserModel(
        id: userData['id'],
        name: userData['name'],
        educationLevel: userData['educationLevel'],
        xp: userData['xp'] ?? 0,
        mathCoins: userData['mathCoins'] ?? 0,
        consecutiveDays: userData['consecutiveDays'] ?? 0,
        achievements: List<String>.from(userData['achievements'] ?? []),
      );
      notifyListeners();
      return true;
    } catch (e) {
      print('Error loading user: $e');
      return false;
    }
  }

  // Update user progress
  void updateProgress({
    int xpEarned = 0, 
    int mathCoinsEarned = 0,
    bool updateConsecutiveDays = false,
  }) {
    if (_currentUser != null) {
      _currentUser!.xp += xpEarned;
      _currentUser!.mathCoins += mathCoinsEarned;

      if (updateConsecutiveDays) {
        _currentUser!.consecutiveDays++;
      }

      // Save updated user data
      _saveUserToPreferences(_currentUser!.name);
      notifyListeners();
    }
  }

  // Add achievement
  void addAchievement(String achievement) {
    if (_currentUser != null) {
      if (!_currentUser!.achievements.contains(achievement)) {
        _currentUser!.achievements.add(achievement);
        
        // Save updated achievements
        _saveUserToPreferences(_currentUser!.name);
        notifyListeners();
      }
    }
  }

  // Check and reward consecutive days
  void checkConsecutiveDaysReward() {
    if (_currentUser != null) {
      if (_currentUser!.consecutiveDays % 7 == 0) {
        updateProgress(
          mathCoinsEarned: 20, // Bonus for 7-day streak
          xpEarned: 10,
        );
        addAchievement('7-Day Learner');
      }
    }
  }

  // Logout
  void logout() {
    _currentUser = null;
    _prefs.remove('user_${_currentUser!.name}'); // Remove user data from local storage
    notifyListeners();
  }
}