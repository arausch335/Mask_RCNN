import os

MASK_RCNN_DIR = os.getcwd()
IMAGE_DIR = os.path.join(MASK_RCNN_DIR, 'images')
LOGS_DIR = os.path.join(MASK_RCNN_DIR, 'logs')
COCO_WEIGHTS_PATH = os.path.join(MASK_RCNN_DIR, 'mask_rcnn_coco.h5')
CUSTOM_DIR = os.path.join(MASK_RCNN_DIR, 'trainCustomModel')
PREPROCESSED_DATA_DIR = os.path.join(MASK_RCNN_DIR, 'trainingData')
DATASET_DIR = os.path.join(CUSTOM_DIR, 'dataset')
CONFIG_DIR = os.path.join(CUSTOM_DIR, 'configurations')
