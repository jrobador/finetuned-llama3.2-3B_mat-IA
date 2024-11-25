import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'screens/welcome_screen.dart';
import 'providers/user_provider.dart';
import 'providers/progress_provider.dart';
import './themes/app_theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize shared preferences for local storage
  final prefs = await SharedPreferences.getInstance();

  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (context) => UserProvider(prefs)),
        ChangeNotifierProvider(create: (context) => ProgressProvider(prefs)),
      ],
      child: MatIAApp(),
    ),
  );
}

class MatIAApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Mat-IA',
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      home: WelcomeScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}