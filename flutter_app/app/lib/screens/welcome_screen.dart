import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/user_provider.dart';
import './main_menu_screen.dart';

class WelcomeScreen extends StatefulWidget {
  const WelcomeScreen({super.key});

  @override
  WelcomeScreenState createState() => WelcomeScreenState();
}

class WelcomeScreenState extends State<WelcomeScreen> {
  final _nameController = TextEditingController();
  String _selectedEducationLevel = 'Primaria';
  bool _isNewUser = true;

  final List<String> _educationLevels = [
    'Primaria', 
    'Secundaria', 
    'Nivel Superior'
  ];

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }

  Future<void> _createNewProfile() async {
    if (_nameController.text.isNotEmpty) {
      try {
        final userProvider = Provider.of<UserProvider>(context, listen: false);
        
        await userProvider.createUser(
          name: _nameController.text,
          educationLevel: _selectedEducationLevel,
        );

        if (mounted && userProvider.currentUser != null) {
          Navigator.pushReplacement(
            context, 
            MaterialPageRoute(
              builder: (context) => MainMenuScreen(user: userProvider.currentUser!)
            )
          );
        }
      } catch (error) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Error al crear el perfil: $error'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Por favor, ingresa tu nombre'),
          backgroundColor: Colors.orange,
        ),
      );
    }
  }

  Future<void> _loadExistingProfile() async {
    if (_nameController.text.isNotEmpty) {
      try {
        final userProvider = Provider.of<UserProvider>(context, listen: false);
        final bool success = await userProvider.loadUser(_nameController.text);
        
        if (!mounted) return;
        
        if (success && userProvider.currentUser != null) {
          Navigator.pushReplacement(
            context, 
            MaterialPageRoute(
              builder: (context) => MainMenuScreen(user: userProvider.currentUser!)
            )
          );
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Usuario no encontrado'),
              backgroundColor: Colors.orange,
            ),
          );
        }
      } catch (error) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Error al cargar el perfil: $error'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Por favor, ingresa tu nombre'),
          backgroundColor: Colors.orange,
        ),
      );
    }
  }

  @override
 Widget build(BuildContext context) {
  return Scaffold(
    body: SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const SizedBox(height: 40),
            Image.asset(
              './assets/matIA_logo.png',
              height: 100,
              width: 100,
              fit: BoxFit.contain,
            ),
              const SizedBox(height: 30),
              const Text(
                "¡HOLA!",
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              const Text(
                "Soy Mat-IA, tu asistente de matemáticas.\n¿Cómo te llamas?",
                style: TextStyle(fontSize: 20),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 40),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Expanded(
                    child: TextButton(
                      onPressed: () => setState(() => _isNewUser = true),
                      style: TextButton.styleFrom(
                        backgroundColor: _isNewUser ? Theme.of(context).primaryColor : null,
                      ),
                      child: Text(
                        'Nuevo Usuario',
                        style: TextStyle(
                          color: _isNewUser ? Colors.white : Theme.of(context).primaryColor,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: TextButton(
                      onPressed: () => setState(() => _isNewUser = false),
                      style: TextButton.styleFrom(
                        backgroundColor: !_isNewUser ? Theme.of(context).primaryColor : null,
                      ),
                      child: Text(
                        'Usuario Existente',
                        style: TextStyle(
                          color: !_isNewUser ? Colors.white : Theme.of(context).primaryColor,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              TextField(
                controller: _nameController,
                decoration: InputDecoration(
                  labelText: 'Nombre',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  prefixIcon: const Icon(Icons.person),
                ),
              ),
              const SizedBox(height: 20),
              if (_isNewUser) ...[
                DropdownButtonFormField<String>(
                  value: _selectedEducationLevel,
                  decoration: InputDecoration(
                    labelText: 'Nivel Educativo',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                    prefixIcon: const Icon(Icons.school),
                  ),
                  items: _educationLevels.map((String level) {
                    return DropdownMenuItem<String>(
                      value: level,
                      child: Text(level),
                    );
                  }).toList(),
                  onChanged: (String? value) {
                    if (value != null) {
                      setState(() => _selectedEducationLevel = value);
                    }
                  },
                ),
                const SizedBox(height: 20),
              ],
              ElevatedButton(
                onPressed: _isNewUser ? _createNewProfile : _loadExistingProfile,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 15),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                child: Text(
                  _isNewUser ? 'Crear Perfil' : 'Cargar Perfil',
                  style: const TextStyle(fontSize: 18),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}