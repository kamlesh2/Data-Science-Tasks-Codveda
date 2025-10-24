import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def load_images(path):
    with open(path, 'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, 28*28)

def load_labels(path):
    with open(path, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)

train_images_path = r"C:\Users\sharm\OneDrive\Desktop\22-10-2025_codeveda\Fashion\train-images-idx3-ubyte\train-images-idx3-ubyte"
train_labels_path = r"C:\Users\sharm\OneDrive\Desktop\22-10-2025_codeveda\Fashion\train-labels-idx1-ubyte\train-labels-idx1-ubyte"
test_images_path  = r"C:\Users\sharm\OneDrive\Desktop\22-10-2025_codeveda\Fashion\t10k-images-idx3-ubyte\t10k-images-idx3-ubyte"
test_labels_path  = r"C:\Users\sharm\OneDrive\Desktop\22-10-2025_codeveda\Fashion\t10k-labels-idx1-ubyte\t10k-labels-idx1-ubyte"

X_train = load_images(train_images_path)
y_train = load_labels(train_labels_path)
X_test = load_images(test_images_path)
y_test = load_labels(test_labels_path)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.1, verbose=2)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()
