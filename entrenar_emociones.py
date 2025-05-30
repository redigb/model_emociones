import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# 📁 Directorios (ajusta estas rutas si tus carpetas son distintas)
ruta_train = 'dataset/train'
ruta_test = 'dataset/test'

# 🧪 Parámetros
img_size = (48, 48)
batch_size = 32

# Detectar automáticamente las clases (carpetas) en train y test
clases_train = sorted(os.listdir(ruta_train))
clases_test = sorted(os.listdir(ruta_test))

if clases_train != clases_test:
    print(f"⚠️ Advertencia: Las clases en train y test no coinciden:")
    print(f"Train: {clases_train}")
    print(f"Test: {clases_test}")

clases = clases_train
print(f"Clases detectadas: {clases}")

# 🔄 Preprocesamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 📥 Carga de datos
train_data = train_datagen.flow_from_directory(
    ruta_train,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    classes=clases
)

test_data = test_datagen.flow_from_directory(
    ruta_test,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    classes=clases
)

# 🧠 Modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(clases), activation='softmax')  # La salida siempre coincide con las clases detectadas
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 🏋️ Entrenamiento
history = model.fit(
    train_data,
    epochs=20,
    validation_data=test_data
)

# 💾 Guardar modelo
model.save("modelo_emociones_clases_detectadas.h5")

# 📊 Visualizar entrenamiento
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()
