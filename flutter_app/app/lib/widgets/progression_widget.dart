import 'package:flutter/material.dart';
import '../../../models/user_model.dart';

class ProgressionWidget extends StatelessWidget {
  final UserModel user;

  const ProgressionWidget({super.key, required this.user});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // XP and Streak Section
        ListTile(
          leading: Icon(Icons.star, color: Colors.orange),
          title: Text('Puntos de Experiencia'),
          subtitle: Text('${user.xp} XP'),
        ),
        ListTile(
          leading: Icon(Icons.calendar_today, color: Colors.green),
          title: Text('Racha de Días'),
          subtitle: Text('${user.consecutiveDays} días consecutivos'),
        ),

        // Math Coins Section
        ListTile(
          leading: Icon(Icons.monetization_on, color: Colors.blue),
          title: Text('Monedas Mat-IA'),
          subtitle: Text('${user.mathCoins} monedas'),
        ),

        // Achievements Section
        ExpansionTile(
          leading: Icon(Icons.badge, color: Colors.purple),
          title: Text('Insignias Desbloqueadas'),
          children: user.achievements.isEmpty
            ? [Text('Aún no hay insignias')]
            : user.achievements.map((achievement) => 
                ListTile(
                  title: Text(achievement),
                  leading: Icon(Icons.check_circle, color: Colors.green),
                )
              ).toList(),
        ),
      ],
    );
  }
}
