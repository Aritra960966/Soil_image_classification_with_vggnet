import tensorflow as tf
import numpy as np

def predict(image):
    model = tf.keras.models.load_model("my_model.h5")
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

if __name__ == "__main__":
    sample_image = np.random.rand(220, 220, 3)  
    result = predict(sample_image)
    print("Predicted class probabilities:", result)
