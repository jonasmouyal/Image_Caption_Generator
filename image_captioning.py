import h5py
import json
import cv2
import time
import gradio as gr
import numpy as np
import config as cfg
import tensorflow as tf
from collections import defaultdict
from pyinflect import getInflection


def load_model_ext(filepath):
    """
    load_model_ext() is loading keras model and the classes of the model's prediction
    :param filepath: string with the path of the model
    :return: model: keras model
    :return: meta_data: classes that the model predicts
    """
    model = tf.keras.models.load_model(filepath)
    f = h5py.File(filepath, mode='r')
    meta_data = None
    if 'my_meta_data' in f.attrs:
        meta_data = f.attrs.get('my_meta_data')
    f.close()
    return model, meta_data


def create_caption_cnn(img, thresh):
    """
    create_caption_cnn() create a caption for an image using a cnn model that was trained on 'Flickr image dataset'
    :param img: a numpy array of the image
    :param thresh: a float with a probability threshold for choosing the model labels' probabilities prediction
    :return: a string of the image caption
    """

    # preprocess image
    img = cv2.resize(img, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))[np.newaxis, :]
    img = img * cfg.RESCALE

    # predict probabilities of the labels
    pred = cnn_model.predict(img)

    # create labels prediction list by the probabilities threshold
    y_pred = np.array([[1 if i > thresh else 0 for i in j] for j in pred])
    pred_labels = cnn_labels[y_pred.astype(bool)[0]]

    # organize labels by their pos tags
    labels_pos_tag = {pred_label: cfg.IMAGE_LABELS[pred_label][-1] for pred_label in pred_labels}
    mydict = dict(labels_pos_tag)
    pos_tag_labels = defaultdict(list)
    for key, value in mydict.items():
        pos_tag_labels[value].append(key)

    # create caption
    caption = ['A']
    if 'NOUN1' in pos_tag_labels:
        for n, noun in enumerate(pos_tag_labels['NOUN1']):
            if n > 0:
                caption += ['and a']
            caption += [noun]
    if 'VERB' in pos_tag_labels:
        for n, verb in enumerate(pos_tag_labels['VERB']):
            if n > 0:
                caption += ['and']
            verb = getInflection(verb, tag='VBG')[0]
            caption += [verb]
    if 'NOUN2' in pos_tag_labels:
        for n, noun in enumerate(pos_tag_labels['NOUN2']):
            if n > 0:
                caption += ['and']
            caption += ['on the', noun]

    return ' '.join(caption)


def create_caption_gui_wrapper(img):
    """
    classify_image() is a function that used in the GUI in order to generate a caption to an image
    :param img: numpy array of the image
    :return: a string of the caption
    """

    # create caption and print the result
    pred_caption = create_caption_cnn(img, cfg.PROBA_THRESH)

    return pred_caption


def main():
    """
    main() getting the input arguments of the py file
    main() also catches exceptions
    """

    # TODO: exceptions!!!!!

    print('Welcome to image captioning')

    run = True
    while run:
        cmd = input('\nPlease choose an option:\n' +
                    '1. Image captioning in terminal\n' +
                    '2. Create GUI link for image captioning\n' +
                    '3. Exit\n')
        if cmd == '1':
            # load image
            file = input('Please enter image file name\n')
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # create caption and print the result
            pred_caption = create_caption_cnn(img, cfg.PROBA_THRESH)
            print('\nImage caption:')
            print(pred_caption)
        elif cmd == '2':
            # create GUId object and print a link to the GUI
            gr.Interface(fn=create_caption_gui_wrapper,
                         inputs=gr.inputs.Image(shape=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
                         outputs='text',
                         examples=cfg.IMAGES_EXAMPLES
                         ).launch(share=True)
        elif cmd == '3':
            run = False
        else:
            print('Wrong input')
        time.sleep(2)


# load model
cnn_model, loaded_labels_string = load_model_ext(cfg.CNN_MODEL_FILE)
cnn_labels = np.array(json.loads(loaded_labels_string))

if __name__ == '__main__':
    main()
