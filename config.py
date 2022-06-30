# Logging configuration
LOGFILE_NAME = "logfile.log"
LOGFILE_DEBUG = False  # When the flag is true the logging level is debug

# cnn model hyperparameters
IMAGE_LABELS = {
    'person': ['man', 'men', 'woman', 'women', 'guy', 'lady', 'person', 'people', 'boy', 'girl', 'child', 'kid',
               'children', 'NOUN1'],
    'dog': ['dog', 'dogs', 'NOUN1'],
    'water': ['water', 'NOUN2'],
    'grass': ['grass', 'NOUN2'],
    'street': ['street', 'NOUN2'],
    'jump': ['jump', 'jumping', 'VERB'],
    'run': ['run', 'runs', 'runing', 'VERB'],
    'ride': ['riding', 'ride', 'VERB'],
    'sit': ['sit', 'VERB']}
# MODE = 'multi_label'
#
# if MODE == 'multi_class':
#   LABEL_CREATION = find_class
# elif MODE == 'multi_label':
#   LABEL_CREATION = find_labels
BATCH_SIZE = 16
IMAGE_SIZE = 256
AUG = True
PROBA_THRESH = 0.4

CNN_MODEL_FILE = 'cnn_model\cnn_model.h5'
