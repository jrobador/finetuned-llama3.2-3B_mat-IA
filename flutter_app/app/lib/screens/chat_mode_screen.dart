import 'package:flutter/material.dart';
import '../models/user_model.dart';
import 'package:langchain/langchain.dart';
import 'package:langchain_ollama/langchain_ollama.dart';
import 'package:flutter_tts/flutter_tts.dart';


class ChatModeScreen extends StatefulWidget {
  final UserModel user;

  const ChatModeScreen({super.key, required this.user});

  @override
  ChatModeScreenState createState() => ChatModeScreenState();
}

class ChatModeScreenState extends State<ChatModeScreen> {
  final TextEditingController _messageController = TextEditingController();
  final List<Map<String, dynamic>> _messages = [];
  final FlutterTts _flutterTts = FlutterTts(); // TTS instance

  @override
  void initState() {
    super.initState();
    // Mensaje de bienvenida personalizado
    _messages.add({
      'text': 'Hola ${widget.user.name}! Estoy listo para ayudarte con cualquier problema matemático. ¿En qué puedo ayudarte hoy?',
      'isUser': false,
      'timestamp': DateTime.now(),
    });
  }

  final memory = ConversationBufferMemory(returnMessages: true);

  void _sendMessage() {
    if (_messageController.text.isEmpty) return;

    setState(() {
      // Mensaje del usuario
      _messages.add({
        'text': _messageController.text,
        'isUser': true,
        'timestamp': DateTime.now(),
      });
    });

    // Llamado a la función de generación de respuesta de IA
    _generateAIResponse(_messageController.text);

    // Limpiar el campo de entrada
    _messageController.clear();
  }

  Future<void> _generateAIResponse(String userMessage) async {
    final promptTemplate = ChatPromptTemplate.fromPromptMessages([
      SystemChatMessagePromptTemplate.fromTemplate(
        'Eres un tutor de matematicas de un alumno de nivel educativo: ${widget.user.educationLevel}. Debes dar un paso a paso de como se debe razonar para llegar a la respuesta de forma detallada, pero adaptandote a su nivel educativo de ${widget.user.educationLevel}. Razona dos veces si no sabes la respuesta. Si no estás 100% seguro de algo, debes indicarlo. Si necesitas información adicional, debes solicitarla. Nunca inventes fórmulas o propiedades matemáticas.',
      ),
      const MessagesPlaceholder(variableName: 'history'),
      HumanChatMessagePromptTemplate.fromTemplate('{Message}'),
    ]);

    // Instancia de memoria para retener el contexto
    print(await memory.loadMemoryVariables());

    // Remoto
    //final chat = ChatOllama(
    //  baseUrl: 'http://34.80.87.250:11434/api',
    //  defaultOptions: ChatOllamaOptions(
    //    model: 'hf.co/jrobador/MatIA-Q8_0-GGUF',
    //  ),
    //);

    //Local
    final chat = ChatOllama(
      defaultOptions: ChatOllamaOptions(
        model: 'hf.co/jrobador/MatIA-Q8_0-GGUF',
      ),
    );

    final chain = Runnable.fromMap({
          'Message': Runnable.passthrough(),
          'history': Runnable.mapInput(
            (_) async {
              final m = await memory.loadMemoryVariables();
              return m['history'];
            },
          ),
        }) |
        promptTemplate |
        chat |
        StringOutputParser();

    // Agregar mensaje inicial vacío al UI
    setState(() {
      _messages.add({
        'text': '',  // este se actualizará con el stream
        'isUser': false,
        'timestamp': DateTime.now(),
      });
    });

    // Buffer para respuesta del AI y actualizar el UI en tiempo real
    final aiResponseBuffer = StringBuffer();
    final stream = chain.stream({'Message': userMessage});

    // Procesar y mostrar respuesta del stream en la app
    await for (final responseChunk in stream) {
      aiResponseBuffer.write(responseChunk);

      setState(() {
        // Actualizar el último mensaje en la lista
        _messages[_messages.length - 1]['text'] = aiResponseBuffer.toString();
      });
    }

    // Guardar el contexto del mensaje en memoria para futuras respuestas
    await memory.saveContext(
      inputValues: {'Message': userMessage},
      outputValues: {'output': aiResponseBuffer.toString()},
    );
  }

  Future<void> _playMessage(String text) async {
    await _flutterTts.setLanguage("es-ES");
    await _flutterTts.setSpeechRate(1.12);
    await _flutterTts.setVolume(1.0);
    await _flutterTts.setPitch(1.15);
    await _flutterTts.speak(text);
  }

  Widget _buildMessageBubble(Map<String, dynamic> message) {
    return Align(
      alignment: message['isUser'] ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: EdgeInsets.symmetric(vertical: 4, horizontal: 8),
        padding: EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: message['isUser'] ? Colors.blue[100] : const Color.fromARGB(255, 44, 62, 77),
          borderRadius: BorderRadius.circular(15),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Expanded(
              child: Text(
                message['text'],
                style: TextStyle(
                  color: message['isUser']
                      ? const Color.fromARGB(255, 0, 0, 0)
                      : const Color.fromARGB(255, 255, 255, 255),
                ),
              ),
            ),
            if (!message['isUser']) // Show speaker icon only for AI messages
              IconButton(
                icon: Icon(Icons.volume_up, color: Colors.white),
                onPressed: () => _playMessage(message['text']),
              ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Conversación'),
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              reverse: true,
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                return _buildMessageBubble(_messages[_messages.length - 1 - index]);
              },
            ),
          ),
          Padding(
            padding: EdgeInsets.all(8.0),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _messageController,
                    decoration: InputDecoration(
                      hintText: 'Escribe tu problema matemático...',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                    maxLines: null,
                  ),
                ),
                SizedBox(width: 8),
                CircleAvatar(
                  backgroundColor: Colors.blue,
                  child: IconButton(
                    icon: Icon(Icons.send, color: Colors.white),
                    onPressed: _sendMessage,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _messageController.dispose();
    _flutterTts.stop(); // Stop any ongoing TTS playback on dispose
    super.dispose();
  }
}
