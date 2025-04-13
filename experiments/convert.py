import tensorflow as tf

def convert_to_tflite():
    model = tf.keras.models.load_model("my_model.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("soil.tflite", "wb").write(tflite_model)
    print("Model converted to TFLite.")

if __name__ == "__main__":
    convert_to_tflite()
