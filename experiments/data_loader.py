from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config

def load_data():
    train_datagen = ImageDataGenerator(rescale=1/255)
    train_generator = train_datagen.flow_from_directory(
        config.DATASET_PATH,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        classes=config.CLASSES,
        class_mode='categorical'
    )
    return train_generator
