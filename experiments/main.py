
import train
import explain
import convert
import inference
import data_loader
import numpy as np

def run_training():
    print("Starting model training...")
    train.train()
    print("Training complete.\n")

def run_explainability():
    print("Generating SHAP explanations...")
    explain.explain_model()
    print("SHAP explanation saved as shap_explanation.png.\n")

def run_conversion():
    print("Converting model to TFLite format...")
    convert.convert_to_tflite()
    print("TFLite conversion complete.\n")

def run_inference():
    print("Running inference on a sample image...")
    train_gen = data_loader.load_data()
    sample_images, _ = next(iter(train_gen))
    sample_image = sample_images[0] 
    prediction = inference.predict(sample_image)
    print("Predicted class probabilities:", prediction)

if __name__ == "__main__":
    run_training()
    run_explainability()
    run_conversion()
    run_inference()
