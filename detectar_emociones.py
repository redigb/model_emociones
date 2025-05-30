import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# 📥 Cargar modelo
try:
    modelo = load_model("modelo_emociones_clases_detectadas.h5")
except Exception as e:
    print("❌ Error al cargar el modelo:", e)
    exit()

# 📚 Emociones en inglés y su traducción al español
clases = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
traduccion_emociones = {
    'angry': 'Enojo',
    'disgust': 'Asco',
    'fear': 'Miedo',
    'happy': 'Felicidad',
    'sad': 'Tristeza',
    'surprise': 'Sorpresa',
    'neutral': 'Neutral',
    'desconocido': 'Desconocido',
    'error': 'Error'
}

# 📸 Inicializar webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 🔁 Loop principal
fps_start = time.time()
fps_count = 0
fps = 0

print("Presiona 'q' para salir...")

while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ No se pudo leer el frame.")
        break

    fps_count += 1
    if fps_count >= 10:
        fps_end = time.time()
        fps = int(fps_count / (fps_end - fps_start))
        fps_start = time.time()
        fps_count = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]

        if roi.size == 0:
            continue  # ignorar si no hay imagen

        try:
            roi_resized = cv2.resize(roi, (48, 48))
        except:
            continue

        roi_normalized = roi_resized.astype("float32") / 255.0
        roi_reshape = np.expand_dims(roi_normalized, axis=(0, -1))

        try:
            pred = modelo.predict(roi_reshape, verbose=0)
            emotion_index = np.argmax(pred)
            emotion = clases[emotion_index] if emotion_index < len(clases) else "desconocido"
        except:
            emotion = "error"

        # Traducir al español
        emotion_es = traduccion_emociones.get(emotion, "Desconocido")

        # Cuadro en rostro + emoción traducida
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_es, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar FPS
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Mostrar frame
    cv2.imshow("🎭 Detección de Emociones", frame)

    # Salida con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🧹 Limpiar
cam.release()
cv2.destroyAllWindows()
