import numpy as np
import pandas as pd
import os
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from sklearn.metrics import f1_score, mean_squared_error

# Define directory (Your dataset root path)
data_dir = "D:/DA_SATYA/mlfow soil/Soil types/"

# Image preprocessing
image_size = 220
batch_size = 10

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Apply train-validation split (80% train, 20% validation)
train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir,  
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset="training"  # 80% for training
)

test_generator = train_datagen.flow_from_directory(
    data_dir,  
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset="validation",  # 20% for validation
    shuffle=False  # Keep order for evaluation
)

# Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(image_size, image_size, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# Start MLflow experiment
mlflow.set_experiment("Soil Classification Experiment")

with mlflow.start_run():
    total_sample = train_generator.n
    n_epochs = 30
    
    # Train model
    history = model.fit(
        train_generator, 
        steps_per_epoch=int(total_sample/batch_size),  
        epochs=n_epochs,
        verbose=1,
        validation_data=test_generator  # Validate on 20% data
    )

    # Log training parameters
    mlflow.log_param("image_size", image_size)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", n_epochs)
    
    # Log final training accuracy
    final_train_accuracy = history.history['accuracy'][-1]
    mlflow.log_metric("train_accuracy", final_train_accuracy)

    # Convert model to TFLite
    model.save("my_model.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open("soil.tflite", "wb") as f:
        f.write(tflite_model)

    # Log model & TFLite file
    mlflow.tensorflow.log_model(model, "soil_classification_model")
    mlflow.log_artifact("soil.tflite")

    # ---- TESTING PHASE ---- #
    print("Evaluating on Test Data...")

    y_true = test_generator.classes  
    y_pred_probs = model.predict(test_generator) 
    y_pred = np.argmax(y_pred_probs, axis=1) 

    # Compute test accuracy
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    f1 = f1_score(y_true, y_pred, average="weighted")
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Log test metrics
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
