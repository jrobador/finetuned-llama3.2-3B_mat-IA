import 'package:flutter/material.dart';

// chat_message.dart
class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;

  ChatMessage({
    required this.text,
    required this.isUser,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}


class ChatMessageWidget extends StatelessWidget {
  final ChatMessage message;

  const ChatMessageWidget({Key? key, required this.message}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.symmetric(vertical: 4, horizontal: 8),
      child: Align(
        alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
        child: Container(
          padding: EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: message.isUser ? Colors.blue[100] : Colors.green[100],
            borderRadius: BorderRadius.circular(15),
          ),
          child: Text(
            message.text,
            style: TextStyle(
              color: message.isUser ? Colors.blue[900] : Colors.green[900],
            ),
          ),
        ),
      ),
    );
  }
}