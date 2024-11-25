import 'package:flutter/material.dart';
import '../models/user_model.dart';
import '../widgets/progression_widget.dart';
import '../../../../screens/learn_mode_screen.dart';
import '../../../../screens/chat_mode_screen.dart';
import 'package:google_fonts/google_fonts.dart';

class MainMenuScreen extends StatelessWidget {
  final UserModel user;

  const MainMenuScreen({super.key, required this.user});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: _buildAppBar(),
      drawer: _buildDrawer(context),
      body: _buildGridMenu(context),
    );
  }

  PreferredSizeWidget _buildAppBar() {
    return AppBar(
      title: const Text('Mat-IA'),
      centerTitle: true,
    );
  }

  Widget _buildDrawer(BuildContext context) {
    return Drawer(
      child: ListView(
        padding: EdgeInsets.zero,
        children: [
          _buildDrawerHeader(),
          ProgressionWidget(user: user),
        ],
      ),
    );
  }

  Widget _buildDrawerHeader() {
    return UserAccountsDrawerHeader(
      decoration: BoxDecoration(
        color: Colors.blue,
      ),
      accountName: Text(
        user.name, 
        style: const TextStyle(fontWeight: FontWeight.bold),
      ),
      accountEmail: Text('${user.educationLevel}'),
      currentAccountPicture: CircleAvatar(
        backgroundColor: Colors.white,
        child: Text(
          user.name.isNotEmpty ? user.name[0].toUpperCase() : '?',
          style: const TextStyle(fontSize: 24),
        ),
      ),
    );
  }

  Widget _buildGridMenu(BuildContext context) {
    return GridView.count(
      crossAxisCount: 2,
      padding: const EdgeInsets.all(16),
      mainAxisSpacing: 16,
      crossAxisSpacing: 16,
      children: [
        _buildModeCard(
          context: context,
          title: 'Aprendizaje Personalizado',
          icon: Icons.school,
          onTap: () => _navigateToScreen(context, LearnModeScreen(user: user)),
        ),
        _buildModeCard(
          context: context,
          title: 'Conversación Interactiva',
          icon: Icons.chat,
          onTap: () => _navigateToScreen(context, ChatModeScreen(user: user)),
        ),
      ],
    );
  }

  void _navigateToScreen(BuildContext context, Widget screen) {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => screen),
    );
  }

  Widget _buildModeCard({
    required BuildContext context,
    required String title,
    required IconData icon,
    required VoidCallback onTap,
  }) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(15),
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(15),
        onTap: onTap,
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  icon, 
                  size: 160, 
                  color: Theme.of(context).primaryColor,
                ),
                const SizedBox(height: 10),
                Text(
                  title,
                  style: GoogleFonts.roboto(
                                  fontSize: 28,              // Tamaño de la fuente
                                  fontWeight: FontWeight.bold,  // Grosor de la fuente
                                  color: const Color.fromARGB(255, 255, 255, 255),     // Color de la fuente
                                  letterSpacing: 1.0,         // Espacio entre letras
                                  height: 1.5,                // Altura de la línea
                                ),
                  textAlign: TextAlign.center,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}