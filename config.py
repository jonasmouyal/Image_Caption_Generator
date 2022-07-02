import os
from keras.applications.xception import Xception

# prediction
PROBA_THRESH = 0.4
IMAGE_LABELS = {
    'person': ['man', 'men', 'woman', 'women', 'guy', 'lady', 'person', 'people', 'boy', 'girl', 'child', 'kid',
               'children', 'NOUN1'],
    'dog': ['dog', 'NOUN1'],
    'water': ['water', 'NOUN2'],
    'grass': ['grass', 'NOUN2'],
    'street': ['street', 'NOUN2'],
    'jump': ['jump', 'VERB'],
    'run': ['run', 'VERB'],
    'ride': ['ride', 'VERB'],
    'sit': ['sit', 'VERB']}

# train validation test
TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.1 / (1 - TEST_SPLIT)

# ImageDataGenerator
COMMENTS = 'image_comments.csv'
IMAGES_FOLDER = "./images/"
RESCALE = 1. / 255.
SEED = 1

# augmentation
AUGMENTATION = True
SHIFT = 0.1
ROTATE = 35

# cnn network
CNN_MODEL_FILE = 'cnn_model/cnn_model.h5'
PRETRAIN_MODEL = Xception
TRAIN_ALL = True
HEAD_UNITS = 256  # final layer before output layer
IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 3e-5
PATIENCE = 3  # EarlyStopping

# transformer
TRANSFORMER_PATH = os.getcwd() + '/transformer'
MAX_LENGTH = 15  # caption's number of tokens
CAPTIONS = 'captions.txt'

# GUI and csv
IMAGES_EXAMPLES_FOLDER = 'examples/'
images = os.listdir(IMAGES_EXAMPLES_FOLDER)
IMAGES_EXAMPLES = [IMAGES_EXAMPLES_FOLDER + img for img in images]
CSV_FILE = 'captions.csv'
