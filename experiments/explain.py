import shap
import matplotlib.pyplot as plt
import tensorflow as tf
import data_loader

def explain_model():
    model = tf.keras.models.load_model('my_model.h5')
    train_generator = data_loader.load_data()
    sample_images, _ = next(iter(train_generator))
    sample_images = sample_images[:5]

    explainer = shap.GradientExplainer(model, sample_images)
    shap_values = explainer.shap_values(sample_images)

    shap.image_plot(shap_values, sample_images)
    plt.savefig("shap_explanation.png")

if __name__ == "__main__":
    explain_model()
