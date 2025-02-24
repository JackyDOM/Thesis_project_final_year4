import os
import cv2
import numpy as np
import requests
import json
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Preprocess data
data_dir = "train_image"
class_names = ["jump", "left", "right", "roll"]
img_size = (64, 64)

images = []
labels = []
for label, class_name in enumerate(class_names):
    folder_path = os.path.join(data_dir, class_name)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size) / 255.0
            images.append(img)
            labels.append(label)

X = np.array(images)
y = to_categorical(np.array(labels), num_classes=len(class_names))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)
datagen.fit(X_train)

# Step 3: Define CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=4),
    epochs=20,
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // 4
)
model.save("subway_gesture_model.h5")

# Step 5: Evaluate
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Step 6: Test with API using an image from train_image
api_key = "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"

# Dynamically select a test image (e.g., first image from 'jump' folder)
test_folder = os.path.join(data_dir, "jump")  # You can change to "left", "right", or "roll"
test_img_name = os.listdir(test_folder)[0]  # Pick the first image in the folder
test_img_path = os.path.join(test_folder, test_img_name)
print(f"Testing with image: {test_img_path}")

test_img = cv2.imread(test_img_path)
if test_img is not None:
    test_img = cv2.resize(test_img, img_size) / 255.0
    test_img = np.expand_dims(test_img, axis=0)
    prediction = model.predict(test_img)
    predicted_gesture = class_names[np.argmax(prediction)]
    print(f"Predicted gesture: {predicted_gesture}")

    # Query the API
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps({
            "model": "deepseek/deepseek-r1:free",
            "messages": [{"role": "user", "content": f"What does the subway gesture '{predicted_gesture}' mean?"}]
        })
    )
    if response.status_code == 200:
        print(response.json()['choices'][0]['message']['content'])
    else:
        print(f"API Error: {response.status_code} - {response.text}")
else:
    print("Test image not found!")