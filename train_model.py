from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
from keras.models import save_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import config as cfg
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import json


def save_model_ext(model, filepath, overwrite=True, meta_data=None):
    save_model(model, filepath, overwrite)
    if meta_data is not None:
        f = h5py.File(filepath, mode='a')
        f.attrs['my_meta_data'] = meta_data
        f.close()


def create_image_generators(df_4_train, df_4_test):
    if cfg.AUGMENTATION:
        datagen = ImageDataGenerator(rescale=cfg.RESCALE, validation_split=cfg.TEST_SPLIT, rotation_range=cfg.ROTATE,
                                     height_shift_range=cfg.SHIFT, width_shift_range=cfg.SHIFT, horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(rescale=cfg.RESCALE, validation_split=cfg.TEST_SPLIT)

    train_gen = datagen.flow_from_dataframe(
        dataframe=df_4_train,
        directory=cfg.IMAGES_FOLDER,
        x_col="image_name",
        y_col="label",
        subset="training",
        batch_size=cfg.BATCH_SIZE,
        seed=cfg.SEED,
        class_mode="categorical",
        target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

    valid_gen = datagen.flow_from_dataframe(
        dataframe=df_4_train,
        directory=cfg.IMAGES_FOLDER,
        x_col="image_name",
        y_col="label",
        subset="validation",
        batch_size=cfg.BATCH_SIZE,
        seed=cfg.SEED,
        class_mode="categorical",
        target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

    test_datagen = ImageDataGenerator(rescale=cfg.RESCALE)
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=df_4_test,
        directory=cfg.IMAGES_FOLDER,
        x_col="image_name",
        y_col="label",
        batch_size=1,
        seed=cfg.SEED,
        shuffle=False,
        class_mode="categorical",
        target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

    return train_gen, valid_gen, test_gen


def train_evaluate_model(train_generator, valid_generator, test_generator):
    pre_trained_model = cfg.PRETRAIN_MODEL(include_top=False, input_shape=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3))
    pre_trained_model.trainable = cfg.TRAIN_ALL

    model = tf.keras.models.Sequential([pre_trained_model,
                                        Flatten(),
                                        Dense(cfg.HEAD_UNITS, activation='relu', trainable=True),
                                        Dense(len(train_generator.class_indices), trainable=True, activation='sigmoid')
                                        ])

    opt = Adam(learning_rate=cfg.LEARNING_RATE)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[Precision(), Recall(), 'accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=cfg.PATIENCE, restore_best_weights=True)
    model.fit(train_generator, validation_data=valid_generator, epochs=cfg.EPOCHS, callbacks=es)

    IMAGE_LABELS_LIST = list(train_generator.class_indices.keys())
    n_images = len(test_generator.classes)
    n_labels = len(IMAGE_LABELS_LIST)
    encode_labels = np.zeros((n_images, n_labels)).astype(int)
    IMAGE_LABELS_LIST = list(train_generator.class_indices.keys())
    for i, labels in enumerate(test_generator.classes):
        for label_index in labels:
            encode_labels[i, label_index] = 1

    y_prob = model.predict(test_generator)
    thresh = 0.5
    y_pred = np.array([[1 if i > thresh else 0 for i in j] for j in y_prob])
    print('Threshold:', thresh)
    print(classification_report(encode_labels, y_pred, target_names=IMAGE_LABELS_LIST))

    return model


def create_df():
    df_image_comments = pd.read_csv(cfg.IMAGES_FOLDER + cfg.CAPTIONS)
    df_image_comments['label'] = df_image_comments[
        ['comment_0', 'comment_1', 'comment_2', 'comment_3', 'comment_4']].apply(find_labels, axis=1)
    df_image_comments = df_image_comments.dropna(axis=0)
    df_train, df_test = train_test_split(df_image_comments, test_size=cfg.TEST_SPLIT, random_state=cfg.SEED)
    return df_train, df_test


def invert_dict(d):
    inverse = dict()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse:
                # If not create a new list
                inverse[item] = [key]
            else:
                inverse[item].append(key)
    return inverse


def find_labels(comments):
    labels = list()
    singles = list()
    for comment in comments:
        stemmer = PorterStemmer()
        singles += [stemmer.stem(plural) for plural in word_tokenize(comment)]

    for key, value in cfg.IMAGE_LABELS.items():
        labels += list(set(value) & set(singles))

    IMAGE_LABELS_REV = invert_dict(cfg.IMAGE_LABELS)
    labels_out = list()
    for label in labels:
        labels_out += IMAGE_LABELS_REV[label]

    return list(set(labels_out))


def main():
    answer = input('Warning- Continue only if you have GPU\n Continue? (y,n)')
    if answer.lower() == 'y':
        df_train, df_test = create_df()
        train_g, valid_g, test_g = create_image_generators(df_train, df_test)
        model = train_evaluate_model(train_g, valid_g, test_g)

    # save
    answer = input('Overwrite model? (y,n)')
    if answer.lower() == 'y':
        labels = list(train_g.class_indices.keys())
        labels_string = json.dumps(labels)
        save_model_ext(model, cfg.CNN_MODEL_FILE, meta_data=labels_string)


if __name__ == '__main__':
    main()
