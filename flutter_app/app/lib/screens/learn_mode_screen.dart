import 'package:flutter/material.dart';
import '../models/user_model.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:langchain/langchain.dart';
import 'package:langchain_ollama/langchain_ollama.dart';

class AIQuestionResponse {
  final String question;
  final String answer;
  final String explanation;
  final int experiencePoints;

  AIQuestionResponse({
    required this.question,
    required this.answer,
    required this.explanation,
    required this.experiencePoints,
  });

factory AIQuestionResponse.fromJson(Map<String, dynamic> json) {
    return AIQuestionResponse(
      question: json['question'].toString(),
      answer: json['answer'].toString(),
      explanation: json['explanation'].toString(),
      experiencePoints: int.parse(json['experiencePoints'].toString()),
    );
  }
}

class LearnModeScreen extends StatefulWidget {
  final UserModel user;
  const LearnModeScreen({super.key, required this.user});

  @override
  LearnModeScreenState createState() => LearnModeScreenState();
}

class LearnModeScreenState extends State<LearnModeScreen> {
  final List<Map<String, dynamic>> _learningPaths = [
    {
      'title': 'Number Ninja',
      'description': 'Domina operaciones matemáticas básicas',
      'icon': Icons.calculate,
      'color': Colors.blue,
      'levels': [
        'Suma y Resta',
        'Multiplicación',
        'División',
        'Números Decimales',
        'Fracciones'
      ]
    },
    {
      'title': 'Life Survivor',
      'description': 'Problemas prácticos en contextos cotidianos',
      'icon': Icons.real_estate_agent,
      'color': Colors.green,
      'levels': [
        'Finanzas Personales',
        'Compras y Presupuesto',
        'Tiempo y Distancia',
        'Cocina y Medidas',
        'Estadística Básica'
      ]
    },
    {
      'title': 'Fast Mind',
      'description': 'Ejercicios para mejorar agilidad en cálculos',
      'icon': Icons.speed,
      'color': Colors.red,
      'levels': [
        'Cálculo Mental Rápido',
        'Patrones Numéricos',
        'Estimación',
        'Puzzles Matemáticos',
        'Desafíos de Velocidad'
      ]
    },
    {
      'title': 'Math Killer',
      'description': 'Juegos matemáticos para aprendizaje divertido',
      'icon': Icons.games,
      'color': Colors.purple,
      'levels': [
        'Trivia Matemática',
        'Rompecabezas Numéricos',
        'Batalla de Cálculo',
        'Estrategia Matemática',
        'Quiz Interactivo'
      ]
    }
  ];

   void _startLearningPath(Map<String, dynamic> path) {
    showModalBottomSheet(
      context: context,
      builder: (BuildContext context) {
        return ListView.builder(
          itemCount: path['levels'].length,
          itemBuilder: (context, index) {
            return ListTile(
              title: Text(path['levels'][index]),
              trailing: Icon(Icons.play_arrow),
              onTap: () {
                if (path['title'] == 'Number Ninja' || path['title'] == 'Life Survivor') {
                  Navigator.pop(context); // Cerrar el modal anterior
                  _startNumberLevel(path['levels'][index], path['title']);
                } else {
                  _showLevelDialog(path['title'], path['levels'][index]);
                }
              },
            );
          },
        );
      },
    );
  }

  late PageController _pageController;

  String _feedback = '';
  bool _showFeedback = false;
  TextEditingController _answerController = TextEditingController();

  late final ChatPromptTemplate _promptTemplate_ninja;
  late final ChatPromptTemplate _promptTemplate_life;
  late final ChatOllama _chatjson;
  late final Runnable _chain_ninja;
  late final Runnable _chain_life;



  @override
  void initState() {
    super.initState();
    _pageController = PageController();
    _initializeNinjaAI();
    _initializeLifeAI();
  }

  @override
  void dispose() {
    _pageController.dispose();
    _answerController.dispose();
    super.dispose();
  }

  void _initializeNinjaAI() {
    _promptTemplate_ninja = ChatPromptTemplate.fromTemplates(const [
        (ChatMessageType.system, '''
        Eres un experto maestro de matemáticas y tienes la tarea de crear preguntas claras y precisas para estudiantes. Cada pregunta debe enfocarse exclusivamente en el tema especificado y usar un nivel de dificultad y experiencia adecuados.

        Instrucciones:
        - **"question"**: Crea una pregunta de operaciones matematicas directa y específica sobre el tema {topic}, sin introducir otros conceptos que no sean relevantes al tema.
        - **"explanation"**: Da una explicación clara y paso a paso que guíe al estudiante en la resolución de la pregunta.
        - **"answer"**: Proporciona solo la respuesta final en formato numérico, revisando que sea coherente con la explicación.
        - **"experiencePoints"**: Da un valor entre 1 y 7 para los puntos de experiencia, basándote en la dificultad de la pregunta que generaste.

        Responde exclusivamente en JSON usando este formato:
          {{
            "experiencePoints": Tipo: "int". Número entero entre 1 y 7 que refleja los puntos que el estudiante puede ganar,
            "question": "Tipo: "string". Pregunta matemática clara y específica sobre el tema {topic}.",
            "explanation": "Tipo: "string". Explicación cómo resolver la pregunta, como si tuviera 5 años y paso a paso.",
            "answer": "Tipo: "int". Respuesta correcta, usando solo números y solo numeros. Si la respuesta es otra cosa, se invalida.",
            
          }}
        '''),

        (ChatMessageType.human, 'Por favor, crea una pregunta precisa y específica sobre {topic} que se ajuste al nivel educativo del usuario que es de {nivel_educativo} (el valor maximo es 100). La pregunta debe enfocarse únicamente en {topic}, sin introducir otros temas.')
    ]);

    // Remoto
    // _chatjson = ChatOllama(
    //   baseUrl: 'https://stingray-fitting-certainly.ngrok-free.app/api',
    //   defaultOptions: ChatOllamaOptions(
    //     model: 'hf.co/jrobador/MatIA-Q8_0-GGUF',
    //     temperature: 0.3,
    //     format: OllamaResponseFormat.json,
    //   ),
    // );

   // Local
     _chatjson = ChatOllama(
       defaultOptions: ChatOllamaOptions(
         model: 'hf.co/jrobador/MatIA-Q8_0-GGUF',
         temperature: 0.3,
         format: OllamaResponseFormat.json,
       ),
     );

    _chain_ninja = _promptTemplate_ninja | _chatjson | JsonOutputParser();
  }

  void _initializeLifeAI() {
  _promptTemplate_life = ChatPromptTemplate.fromTemplates(const [
    (ChatMessageType.system, '''
    Eres un experto maestro de matemáticas y tienes la tarea de crear preguntas claras, prácticas y precisas para estudiantes. Cada pregunta debe enfocarse exclusivamente en el tema especificado, utilizando un nivel de dificultad y experiencia adecuados, y debe aplicar operaciones matemáticas en situaciones de la vida cotidiana. Si realizas una pregunta sobre realizar una operación matemática, incluye números y detalles específicos para que el estudiante pueda resolverla directamente.

    Instrucciones:
    - **"question"**: Crea una pregunta matemática clara con absolutamente todo el contexto necesario para responderla y específica sobre el tema {topic}, situándola en un contexto práctico y realista (por ejemplo, calcular el costo de varios artículos en una tienda, sumar tiempos en una actividad, o calcular distancias recorridas). Asegúrate de proporcionar los números exactos necesarios para que el estudiante realice el cálculo.
    - **"explanation"**: Da una explicación clara y detallada, paso a paso, como si le explicaras a alguien de 5 años. La explicación debe guiar al estudiante en la resolución del problema y mostrar cómo se aplican las operaciones matemáticas al contexto dado.
    - **"answer"**: Proporciona solo la respuesta final en formato numérico, revisando que sea coherente con la explicación y la operación realizada.
    - **"experiencePoints"**: Asigna un valor entre 1 y 7 para los puntos de experiencia, basándote en la dificultad y la utilidad práctica de la pregunta generada.

    Responde exclusivamente en JSON usando este formato:
      {{
        "experiencePoints": Tipo: "int". Número entero entre 1 y 7 que refleja los puntos que el estudiante puede ganar,
        "question": "Tipo: "string". Pregunta matemática clara y específica sobre el tema {topic}, con un contexto práctico y relevante que incluya números específicos.",
        "explanation": "Tipo: "string". Explicación de cómo resolver la pregunta paso a paso, enfocada en el contexto cotidiano.",
        "answer": "Tipo: "int". Respuesta correcta, usando solo números y únicamente números. Si la respuesta es otra cosa, se invalida.",
      }}
    '''),
    
    (ChatMessageType.human, 'Por favor, crea una pregunta específica que incluya todos los datos necesarios para responderla sobre {topic} que se ajuste al nivel educativo del usuario, que es de {nivel_educativo}. La pregunta debe estar en un contexto cotidiano y relevante, y enfocarse únicamente en {topic}, sin introducir otros temas.')
  ]);

    _chain_life = _promptTemplate_life | _chatjson | JsonOutputParser();
  }

  Future<AIQuestionResponse> _generateAIQuestion(String topic, String roadmap) async {
    try {
      if (roadmap == "Number Ninja"){
        final response = await _chain_ninja.invoke({
          'topic': topic,
          'nivel_educativo': widget.user.xp,
        });
        print (response);
        return AIQuestionResponse.fromJson(response as Map<String, dynamic>);}
      if (roadmap == "Life Survivor"){
        final response = await _chain_life.invoke({
          'topic': topic,
          'nivel_educativo': widget.user.xp,
        });
        print (response);
        return AIQuestionResponse.fromJson(response as Map<String, dynamic>);}
      else return AIQuestionResponse(question: "", answer: "", explanation: "", experiencePoints: 0);        
    } catch (e) {
      print('Error generating question: $e');
      // Fallback question en caso de error
      return AIQuestionResponse(
        question: e.toString(),
        answer: "8",
        explanation: "La suma de 5 y 3 es 8",
        experiencePoints: 10,
      );
    }
  }

  void _startNumberLevel(String level, String roadmap) {
    setState(() {
      _feedback = '';
      _showFeedback = false; 
    });
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (BuildContext context) {
        return Container(
          height: MediaQuery.of(context).size.height * 0.9,
          child: PageView(
            controller: _pageController,
            physics: NeverScrollableScrollPhysics(),
            children: [
              _buildIntroductionPage(level, roadmap),
              _buildPracticePage(level, roadmap),
            ],
          ),
        );
      },
    );
  }

  Widget _buildIntroductionPage(String level, String roadmap) {
    return FutureBuilder(
      future:  roadmap == 'Number Ninja' ? _generateNinjaAIIntroduction(level) 
            :  (roadmap == 'Life Survivor' ? _generateLifeAIIntroduction(level) 
            :   null),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }
        return Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                "Introducción a $level",
                style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 30),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.blue.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  snapshot.data ?? "Cargando introducción...",
                  textAlign: TextAlign.center,
                                  style: GoogleFonts.montserrat(
                                  fontSize: 14,              // Tamaño de la fuente
                                  color: const Color.fromARGB(255, 255, 255, 255),     // Color de la fuente
                                  letterSpacing: 1.0,         // Espacio entre letras
                                  height: 1.5,                // Altura de la línea
                                ),
                ),
              ),
              const SizedBox(height: 40),
              ElevatedButton.icon(
                icon: const Icon(Icons.play_arrow),
                label: const Text("Comenzar Práctica"),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                ),
                onPressed: () => _pageController.nextPage(
                  duration: const Duration(milliseconds: 500),
                  curve: Curves.easeInOut,
                ),
              ),
              const Text(
                "¡Gana experiencia respondiendo correctamente!",
                style: TextStyle(fontSize: 14, color: Colors.grey),
              ),
            ],
          ),
        );
      },
    );
  }

Widget _buildPracticePage(String level, String roadmap) {
  return FutureBuilder<AIQuestionResponse>(
    future: _generateAIQuestion(level, roadmap),
    builder: (context, snapshot) {
      if (snapshot.connectionState == ConnectionState.waiting) {
        return const Center(child: CircularProgressIndicator());
      }

      if (snapshot.hasError || !snapshot.hasData) {
        return Center(
          child: Text('Error: No se pudo cargar la pregunta.'),
        );
      }

      final question = snapshot.data!;

      return StatefulBuilder(
        builder: (BuildContext context, StateSetter setModalState) {
          return Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  question.question,
                  style: const TextStyle(fontSize: 20),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 20),
                TextField(
                  controller: _answerController,
                  decoration: const InputDecoration(
                    hintText: "Escribe tu respuesta",
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 20),
                if (_showFeedback)
                  Text(
                    _feedback,
                    style: TextStyle(
                      color: _feedback.contains("¡Correcto!") 
                        ? Colors.green 
                        : Colors.red,
                    ),
                  ),
                const SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _showFeedback ? null : () {  // Button will be disabled when _showFeedback is true
                    final userAnswer = _answerController.text.trim();
                    final correctAnswer = question.answer.trim();
                    
                    setModalState(() {
                      _showFeedback = true;
                      if (userAnswer == correctAnswer) {
                        _feedback = "¡Correcto! Has ganado ${question.experiencePoints} puntos de experiencia.";
                        widget.user.xp += question.experiencePoints;
                      } else {
                        _feedback = "Incorrecto. La respuesta correcta es $correctAnswer.\n${question.explanation}";
                      }
                    });
                  },
                  child: const Text("Verificar Respuesta"),
                ),
              ],
            ),
          );
        },
      );
    },
  );
}
  Future<String> _generateLifeAIIntroduction(String level) async {
    final promptTemplate = ChatPromptTemplate.fromTemplates([
      (ChatMessageType.system, 'Eres un tutor de matematicas de un alumno de nivel educativo: ${widget.user.educationLevel}. Debes explicar cómo se aplica la matematica en el contexto cotidiano que tu alumno te pregunte. Nunca inventes fórmulas o propiedades matemáticas. No superes las 200 palabras.'),
      (ChatMessageType.human, 'Explicame como afecta la matemática en {concept}, relacionado al contexto cotidiano de las personas. Asegurate de explicarlo de forma concisa y de explicarlo de forma que lo pueda entender fácilmente. No generes ninguna pregunta inicial ni final.'),
      ]);

      // Remoto
      // final chatModel = ChatOllama(
      //   baseUrl: 'https://stingray-fitting-certainly.ngrok-free.app/api',
      //   defaultOptions: ChatOllamaOptions(
      //   model: 'hf.co/jrobador/MatIA-Q8_0-GGUF',
      //   temperature: 0,
      //   ),
      // );

      // Local
       final chatModel = ChatOllama(
         defaultOptions: ChatOllamaOptions(
         model: 'hf.co/jrobador/MatIA-Q8_0-GGUF',
         temperature: 0,
         ),
       );

      final chain = promptTemplate | chatModel | StringOutputParser();
      
      final res = await chain.invoke({'concept': level});
      return res.toString();

  }

  Future<String> _generateNinjaAIIntroduction(String level) async {
    final promptTemplate = ChatPromptTemplate.fromTemplates([
      (ChatMessageType.system, 'Eres un tutor de matematicas de un alumno de nivel educativo: ${widget.user.educationLevel}. Explica el concepto que tu alumno te pregunta de la forma mas clara y precisa como lo haría el mejor tutor de matemáticas sin que quede nada ambiguo ya que solo realizará una sola pregunta. Nunca inventes fórmulas o propiedades matemáticas.'),
      (ChatMessageType.human, 'Explicame el concepto {concept}, asegurándote de explicarlo de forma concisa y de explicarlo de forma que lo pueda entender fácilmente. No generes ninguna pregunta inicial ni final.'),
      ]);

      // Remoto
     // final chatModel = ChatOllama(
     //   baseUrl: 'https://stingray-fitting-certainly.ngrok-free.app/api',
     //   defaultOptions: ChatOllamaOptions(
     //   model: 'hf.co/jrobador/MatIA-Q8_0-GGUF',
     //   temperature: 0,
     //   ),
     // );

      // Local
       final chatModel = ChatOllama(
         defaultOptions: ChatOllamaOptions(
         model: 'hf.co/jrobador/MatIA-Q8_0-GGUF',
         temperature: 0,
         ),
       );

      final chain = promptTemplate | chatModel | StringOutputParser();

      final res = await chain.invoke({'concept': level});
      return res.toString();

  }

  void _showLevelDialog(String pathTitle, String level) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('$pathTitle - $level'),
          content: Text('Próximamente: Ejercicios interactivos'),
          actions: [
            TextButton(
              child: Text('Cerrar'),
              onPressed: () => Navigator.of(context).pop(),
            )
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Aprender'),
        actions: [
          IconButton(
            icon: Icon(Icons.help),
            onPressed: () {
              // Add tutorial or help information
            },
          )
        ],
      ),
      body: GridView.builder(
        padding: EdgeInsets.all(16),
        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 2,
          childAspectRatio: 0.8,
          crossAxisSpacing: 16,
          mainAxisSpacing: 16,
        ),
        itemCount: _learningPaths.length,
        itemBuilder: (context, index) {
          var path = _learningPaths[index];
          return Card(
            elevation: 4,
            color: path['color'].withOpacity(0.1),
            child: InkWell(
              onTap: () => _startLearningPath(path),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(path['icon'], size: 80, color: path['color']),
                  SizedBox(height: 10),
                  Text(
                    path['title'],
                    style: TextStyle(
                      fontSize: 24, 
                      fontWeight: FontWeight.bold,
                      color: path['color']
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Text(
                      path['description'],
                      textAlign: TextAlign.center,
                          style: GoogleFonts.orbitron(
                                  fontSize: 12,              // Tamaño de la fuente
                                  fontWeight: FontWeight.bold,  // Grosor de la fuente
                                  color: Colors.blueGrey,     // Color de la fuente
                                  letterSpacing: 1.0,         // Espacio entre letras
                                  height: 1.5,                // Altura de la línea
                                ),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

 
}