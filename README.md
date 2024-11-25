# MatIA: Democratizando la Educación Matemática

MatIA es un tutor de matemáticas impulsado por el modelo Llama 3.2 3B, finetuneado con datasets especializados en matemáticas para proporcionar una experiencia de aprendizaje interactiva y personalizada. Diseñado para funcionar en dispositivos móviles y en entornos sin conexión a internet, MatIA adapta el nivel de dificultad y los conceptos según el nivel educativo y la experiencia del estudiante, acercando así la educación matemática a más personas.

## Preprocesamiento de Datos

Para el entrenamiento, utilizamos los siguientes datasets:

- [MathInstruct](https://huggingface.co/datasets/agicorp/MathInstruct)
- [GSM8k](https://huggingface.co/datasets/openai/gsm8k)

Ambos datasets fueron traducidos al español utilizando los scripts en `preprocess/multicore_translate_data.py`, los cuales permiten procesamiento en modo multicore o en modo simple. Los datasets traducidos se encuentran disponibles en Hugging Face:

- [GSM8k en español](https://huggingface.co/jrobador/gsm8k_es)
- [MathInstruct en español](https://huggingface.co/jrobador/mathinstruct_es)
- [MathSet_spanish (dataset combinado)](https://huggingface.co/jrobador/MathSet_spanish)

Agradecemos a los creadores originales de estos datasets en inglés.

## Entrenamiento

El entrenamiento se realizó en Hugging Face Space utilizando el script `train/run_train.ipynb`. Durante el proceso:

1. Se empleó `chat_template.py` en `train/script` para procesar los datos y formatearlos de acuerdo con el estándar de LLaMa 3, empleando sus tokens especiales. Más detalles sobre el formato en la [documentación de LLaMa 3](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#-special-tokens-).
2. Se utilizó `SFTTrainer` para optimizar el modelo, ajustando los parámetros específicos para el contexto matemático. Ver más en la [documentación de SFTTrainer](https://huggingface.co/docs/trl/sft_trainer).

El entrenamiento se realizó en una instancia con GPU A10G y tuvo una duración de 10 horas.

### Modelos Cuantizados

Para maximizar la eficiencia, el modelo fue cuantizado usando `gguf` en colaboración con ggml-org. Las versiones cuantizadas disponibles son:

- [MatIA-Q4_K_M-GGUF](https://huggingface.co/jrobador/MatIA-Q4_K_M-GGUF)
- [MatIA-Q8_0-GGUF](https://huggingface.co/jrobador/MatIA-Q8_0-GGUF)

## Ejecutar MatIA Localmente

Para correr MatIA de manera local:

1. **Instalación**:
   - Instala [Ollama](https://ollama.com/download).
   - Instala los recursos de Flutter para VS Code.

2. **Correr el servidor**:
   - Ejecuta uno de los siguientes comandos para inicializar el modelo en Ollama:
     ```bash
     ollama run hf.co/jrobador/MatIA-Q8_0-GGUF   # Para el modelo Q8
     ollama run hf.co/jrobador/MatIA-Q4_K_M-GGUF # Para el modelo Q4_K_M
     ```

3. **Configurar Flutter**:
   ```bash
   git clone https://github.com/jrobador/finetuned-llama3.2-3B_mat-IA.git
   cd finetuned-llama3.2-3B_mat-IA/flutter_app/app
   flutter clean
   flutter pub get
   flutter pub upgrade
   flutter run -d chrome
    ```

La aplicación interactúa con el servidor de Ollama en `localhost:11434`, gestionando las peticiones de manera local.

## MatIA en la Web

También hemos desplegado una versión de MatIA en Vercel: [https://flutter-app-mat-ia.vercel.app/](https://flutter-app-mat-ia.vercel.app/). Esta versión depende de un servidor temporal conectado a una instancia de Google Cloud con GPU Tesla T4 mediante Ngrok. 

> Nota: Por limitaciones de conexión, es posible que algunas secciones de Aprendizaje Personalizado o la pestaña de Práctica presenten errores de carga. Se recomienda utilizar la versión local para una experiencia óptima.

## Video de Presentación

Para una explicación completa del proyecto y una demostración en video, visita el siguiente enlace:

[Ver presentación en YouTube](https://www.youtube.com/watch?v=57-zDScQV80&ab_channel=JoaquinRobador)