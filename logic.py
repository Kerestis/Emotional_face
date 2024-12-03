from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from livelossplot import PlotLossesKeras
from tensorflow.keras.callbacks import ModelCheckpoint
import os

print('Введите размер батча:', end=' ')
BATCH_SIZE = int(input())
print('Введите размер изображения:', end=' ')
IMG_SHAPE = int(input())
print("Укажите директорию с данными для обучения:", end=' ')
n = input()
dir = Path(n)
print('Введите количество эмоций:', end=' ')
NUM_CLASSES = int(input())
print('Введите количество эпох:', end=' ')
EPOCHS = int(input())


# Создание ImageDataGenerator с препроцессингом

def DataGeneration():
    image_gen = ImageDataGenerator(preprocessing_function=preprocess_input,  # препроцессинг
                                   validation_split=0.2,
                                   # размер валидационной выборки, так как всё берётся из одной папки
                                   rotation_range=40,  # максимальный угол поворота
                                   width_shift_range=0.2,  # смещение максимум на 20% ширины по горизонтали
                                   height_shift_range=0.2,  # смещение максимум на 20% высоты по вертикали
                                   zoom_range=0.2,  # картинка будет увеличена или уменьшена не более чем на 20%
                                   horizontal_flip=True,  # случайное отражение по горизонтали
                                   fill_mode="nearest")  # чем заполнять пробелы
    return image_gen


# Создание генераторов данных для тренировки и валидации
def create_generation_for_training(BATCH_SIZE, IMG_SHAPE, dir):
    train_data_gen = DataGeneration().flow_from_directory(batch_size=BATCH_SIZE,  # размер батча
                                                          directory=dir,  # директория для доступа к изображениям
                                                          shuffle=True,  # перемешивать ли данные
                                                          target_size=(IMG_SHAPE, IMG_SHAPE),  # размер изображения
                                                          class_mode="categorical",  # тип классового распределения
                                                          subset="training")  # указываем, что данная выборка тренировочная
    return train_data_gen


def validation(BATCH_SIZE, IMG_SHAPE, dir):
    val_data_gen = DataGeneration().flow_from_directory(batch_size=BATCH_SIZE,
                                                        directory=dir,
                                                        shuffle=False,
                                                        target_size=(IMG_SHAPE, IMG_SHAPE),
                                                        class_mode='categorical',
                                                        subset="validation")  # указываем, что данная выборка валидационная
    return val_data_gen


IMG_SHAPE1 = (IMG_SHAPE, IMG_SHAPE, 3)


def create_base_model(IMG_SHAPE1):
    base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE1,
                                             include_top=False,
                                             weights='imagenet')
    base_model.trainable = True
    base_model.summary()
    return base_model


def create_model(NUM_CLASSES):
    model = tf.keras.Sequential([create_base_model(IMG_SHAPE1),
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dense(512, activation="relu"),
                                 tf.keras.layers.Dropout(0.3),
                                 tf.keras.layers.Dense(NUM_CLASSES)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    return model


checkpoint_path = "trainer/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)


# Создайте колбэк ModelCheckpoint
def create_checkpoint(checkpoint_path):
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  verbose=1)
    return cp_callback


model = create_model(NUM_CLASSES)
if latest_checkpoint:
    try:
        model.load_weights(latest_checkpoint)
        print("Веса успешно загружены.")
    except Exception as e:
        print(f"Ошибка при загрузке весов: {e}")

history = model.fit(create_generation_for_training(BATCH_SIZE, IMG_SHAPE, dir),
                    epochs=EPOCHS,
                    validation_data=validation(BATCH_SIZE, IMG_SHAPE, dir),
                    callbacks=[PlotLossesKeras(), create_checkpoint(checkpoint_path)])
model.save("Logic_Model.h5")
