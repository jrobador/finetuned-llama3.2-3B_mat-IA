import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert'; 

class ProgressProvider with ChangeNotifier {
  final SharedPreferences _prefs;

  // Global leaderboard
  List<Map<String, dynamic>> _globalLeaderboard = [];
  List<Map<String, dynamic>> get globalLeaderboard => _globalLeaderboard;

  // Achievement tracking
  final Map<String, int> _globalAchievementStats = {};
  Map<String, int> get globalAchievementStats => _globalAchievementStats;

  // Learning paths progress
  Map<String, dynamic> _learningPathsProgress = {
    'Primaria': {
      'Number Ninja': {
        'completed': 0,
        'total': 10,
        'progress': 0.0
      },
      'Geometry Basics': {
        'completed': 0,
        'total': 8,
        'progress': 0.0
      }
    },
    'Secundaria': {
      'Algebra Master': {
        'completed': 0,
        'total': 15,
        'progress': 0.0
      },
      'Trigonometry Explorer': {
        'completed': 0,
        'total': 12,
        'progress': 0.0
      }
    }
  };
  Map<String, dynamic> get learningPathsProgress => _learningPathsProgress;

  ProgressProvider(this._prefs) {
    _initializeData();
  }

  // Initialize data from SharedPreferences
Future<void> _initializeData() async {
  try {
    // Load global leaderboard
    String? leaderboardJson = _prefs.getString('globalLeaderboard');
    _globalLeaderboard = leaderboardJson != null 
      ? List<Map<String, dynamic>>.from(jsonDecode(leaderboardJson))
      : []; // Default empty list if no data
    
    // Load global achievement stats
    String? achievementStatsJson = _prefs.getString('globalAchievementStats');
    _globalAchievementStats.addAll(
      achievementStatsJson != null 
        ? Map<String, int>.from(jsonDecode(achievementStatsJson))
        : {} // Default empty map if no data
    );
    
    // Load learning paths progress
    String? learningPathsJson = _prefs.getString('learningPathsProgress');
    _learningPathsProgress = learningPathsJson != null 
      ? Map<String, dynamic>.from(jsonDecode(learningPathsJson))
      : {}; // Default empty map if no data
    
    notifyListeners();
  } catch (e) {
    print('Error initializing data: $e');
    // Initialize with default values in case of error
    _globalLeaderboard = [];
    _globalAchievementStats.clear();
    _learningPathsProgress = {};
    notifyListeners();
  }
}

  // Save data to SharedPreferences
  Future<void> _saveData() async {
    try {
      await _prefs.setString('globalLeaderboard', jsonEncode(_globalLeaderboard));
      await _prefs.setString('globalAchievementStats', jsonEncode(_globalAchievementStats));
      await _prefs.setString('learningPathsProgress', jsonEncode(_learningPathsProgress));
    } catch (e) {
      print('Error saving data: $e');
    }
  }

  // Fetch global leaderboard
  Future<void> fetchGlobalLeaderboard() async {
    try {
      // Simulate fetching data (for demonstration purposes)
      List<Map<String, dynamic>> leaderboard = [
        {'name': 'Alice', 'xp': 1500, 'educationLevel': 'Primaria'},
        {'name': 'Bob', 'xp': 1200, 'educationLevel': 'Secundaria'},
        {'name': 'Charlie', 'xp': 1000, 'educationLevel': 'Primaria'}
      ];

      _globalLeaderboard = leaderboard;
      await _saveData();
      notifyListeners();
    } catch (e) {
      print('Error fetching leaderboard: $e');
    }
  }

  // Update global achievement stats
  Future<void> updateGlobalAchievementStats() async {
    try {
      // Simulate updating achievement stats (for demonstration purposes)
      Map<String, int> achievementStats = {
        '7-Day Learner': 5,
        'Math Master': 3,
        'Geometry Guru': 2
      };

      _globalAchievementStats.addAll(achievementStats);
      await _saveData();
      notifyListeners();
    } catch (e) {
      print('Error updating achievement stats: $e');
    }
  }

  // Update learning path progress
  void updateLearningPathProgress({
    required String educationLevel,
    required String path,
    int completedItems = 1
  }) {
    if (_learningPathsProgress.containsKey(educationLevel) &&
        _learningPathsProgress[educationLevel].containsKey(path)) {
      
      var pathProgress = _learningPathsProgress[educationLevel][path];
      pathProgress['completed'] += completedItems;
      pathProgress['progress'] = pathProgress['completed'] / pathProgress['total'];

      _saveData().then((_) {
        notifyListeners();
      });
    }
  }

  // Reset progress (useful for testing or user request)
  void resetProgress() {
    _globalLeaderboard.clear();
    _globalAchievementStats.clear();
    
    // Reset learning paths
    _learningPathsProgress = {
      'Primaria': {
        'Number Ninja': {
          'completed': 0,
          'total': 10,
          'progress': 0.0
        },
        'Geometry Basics': {
          'completed': 0,
          'total': 8,
          'progress': 0.0
        }
      },
      'Secundaria': {
        'Algebra Master': {
          'completed': 0,
          'total': 15,
          'progress': 0.0
        },
        'Trigonometry Explorer': {
          'completed': 0,
          'total': 12,
          'progress': 0.0
        }
      }
    };

    _saveData().then((_) {
      notifyListeners();
    });
  }
}