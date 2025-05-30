# Proyecto de Detección de Emociones con Python y TensorFlow

Este proyecto utiliza un modelo de redes neuronales entrenado con Keras/TensorFlow para detectar emociones faciales en tiempo real a través de la cámara web. Se emplea OpenCV para la captura de video y detección de rostros usando Haar Cascade.

![grafico_entrenamineto emociones](https://github.com/user-attachments/assets/27a61107-aaee-44aa-8657-1a1fc8ae3707)
---

## Características

- Captura de video en tiempo real desde la webcam.
- Detección de rostros usando Haar Cascade.
- Clasificación de emociones faciales en las siguientes clases: `enojado`, `asco`, `miedo`, `feliz`, `triste`, `sorprendido`, `neutral`.
- Visualización en tiempo real de la emoción detectada con un recuadro alrededor del rostro.
- Medición de FPS para monitorear el rendimiento.

---

## Requisitos

- Python 3.8 o superior
- TensorFlow 2.19.0
- OpenCV
- NumPy


## Recomendaciones:

Un **entorno virtual** es un espacio aislado que contiene su propia instalación de Python y sus propias librerías, separadas del resto del sistema operativo y de otros proyectos. Así puedes:

- Instalar las dependencias específicas que tu proyecto necesita sin afectar otros proyectos.
- Evitar conflictos de versiones entre librerías.
- Facilitar el manejo y despliegue del proyecto, ya que el entorno contiene solo lo que el proyecto necesita.
- Mejorar la reproducibilidad del proyecto: otro desarrollador puede crear un entorno idéntico usando el archivo `requirements.txt`.

### Crear el entorno:

1. **Crear el entorno virtual:**

   ```bash
   python -m venv env

2. **instalar dependencias**
   ```bash
   pip install -r requirements.txt
