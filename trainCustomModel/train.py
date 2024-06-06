import os
import sys
import json
from datetime import datetime
import imgaug.augmenters as iaa
import numpy as np
import skimage.draw
import yaml
import pandas as pd
from keras.callbacks import CSVLogger


# Root directory of the project
MASK_RCNN_DIR = os.path.abspath("../../")
try:
    assert 'Mask_RCNN' in str(MASK_RCNN_DIR)
except AssertionError:
    MASK_RCNN_DIR = os.path.abspath('./')

sys.path.append(MASK_RCNN_DIR)
from pathVariables import *
from mrcnn.config import Config
from mrcnn import model as modellib, utils

############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the custom dataset.
    Derives from the base Config class and overrides some values.
    """

    def load_from_yaml(self, filePath):
        # Make changes to base config as specified in config file
        data = self.read_yaml(filePath)
        fullData = self.read_yaml(os.path.join(CONFIG_DIR, 'baseConfig.yaml'))

        for key, value in data.items():
            fullData[key] = value

        # Make changes to class instance
        self.NAME = fullData['NAME']
        self.DATASET_DIR = DATASET_DIR if fullData['DATASET_DIR'] == 'dataset' else fullData['DATASET_DIR']
        self.WEIGHTS = fullData['WEIGHTS']
        self.EPOCHS = fullData['EPOCHS']
        self.LAYERS = fullData['LAYERS']
        self.CLASSES = fullData['CLASSES']
        self.PREPROCESSED_DATA_DIR = fullData['PREPROCESSED_DATA_DIR']
        self.AUGMENTATION = fullData['AUGMENTATION']
        self.EVALUATE = fullData['EVALUATE']

        self.GPU_COUNT = fullData['GPU_COUNT']
        self.IMAGES_PER_GPU = fullData['IMAGES_PER_GPU']
        self.STEPS_PER_EPOCH = fullData['STEPS_PER_EPOCH']
        self.VALIDATION_STEPS = fullData['VALIDATION_STEPS']
        self.BACKBONE = fullData['BACKBONE']
        self.COMPUTE_BACKBONE_SHAPE = None if fullData['COMPUTE_BACKBONE_SHAPE'] == 'None' else fullData['COMPUTE_BACKBONE_SHAPE']
        self.BACKBONE_STRIDES = fullData['BACKBONE_STRIDES']
        self.FPN_CLASSIF_FC_LAYERS_SIZE = fullData['FPN_CLASSIF_FC_LAYERS_SIZE']
        self.TOP_DOWN_PYRAMID_SIZE = fullData['TOP_DOWN_PYRAMID_SIZE']
        self.NUM_CLASSES = fullData['NUM_CLASSES']
        self.RPN_ANCHOR_SCALES = fullData['RPN_ANCHOR_SCALES']
        self.RPN_ANCHOR_RATIOS = fullData['RPN_ANCHOR_RATIOS']
        self.RPN_ANCHOR_STRIDE = fullData['RPN_ANCHOR_STRIDE']
        self.RPN_NMS_THRESHOLD = fullData['RPN_NMS_THRESHOLD']
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = fullData['RPN_TRAIN_ANCHORS_PER_IMAGE']
        self.PRE_NMS_LIMIT = fullData['PRE_NMS_LIMIT']
        self.POST_NMS_ROIS_TRAINING = fullData['POST_NMS_ROIS_TRAINING']
        self.POST_NMS_ROIS_INFERENCE = fullData['POST_NMS_ROIS_INFERENCE']
        self.USE_MINI_MASK = fullData['USE_MINI_MASK']
        self.MINI_MASK_SHAPE = fullData['MINI_MASK_SHAPE']
        self.IMAGE_RESIZE_MODE = fullData['IMAGE_RESIZE_MODE']
        self.IMAGE_MIN_DIM = fullData['IMAGE_MIN_DIM']
        self.IMAGE_MAX_DIM = fullData['IMAGE_MAX_DIM']
        self.IMAGE_MIN_SCALE = fullData['IMAGE_MIN_SCALE']
        self.IMAGE_CHANNEL_COUNT = fullData['IMAGE_CHANNEL_COUNT']
        self.MEAN_PIXEL = np.array(fullData['MEAN_PIXEL'])
        self.TRAIN_ROIS_PER_IMAGE = fullData['TRAIN_ROIS_PER_IMAGE']
        self.ROI_POSITIVE_RATIO = fullData['ROI_POSITIVE_RATIO']
        self.POOL_SIZE = fullData['POOL_SIZE']
        self.MASK_POOL_SIZE = fullData['MASK_POOL_SIZE']
        self.MASK_SHAPE = fullData['MASK_SHAPE']
        self.MAX_GT_INSTANCES = fullData['MAX_GT_INSTANCES']
        self.RPN_BBOX_STD_DEV = np.array(fullData['RPN_BBOX_STD_DEV'])
        self.BBOX_STD_DEV = np.array(fullData['BBOX_STD_DEV'])
        self.DETECTION_MAX_INSTANCES = fullData['DETECTION_MAX_INSTANCES']
        self.DETECTION_MIN_CONFIDENCE = fullData['DETECTION_MIN_CONFIDENCE']
        self.DETECTION_NMS_THRESHOLD = fullData['DETECTION_NMS_THRESHOLD']
        self.LEARNING_RATE = fullData['LEARNING_RATE']
        self.LEARNING_MOMENTUM = fullData['LEARNING_MOMENTUM']
        self.WEIGHT_DECAY = fullData['WEIGHT_DECAY']
        self.LOSS_WEIGHTS = fullData['LOSS_WEIGHTS']
        self.USE_RPN_ROIS = fullData['USE_RPN_ROIS']
        self.TRAIN_BN = fullData['TRAIN_BN']
        self.GRADIENT_CLIP_NORM = fullData['GRADIENT_CLIP_NORM']
        self.CONFIG_FILE = filePath

        self.compute_attributes()
    
    def create_dataframe(self):
        config_series = pd.Series({
            'name': self.NAME,
            'subset': None,
            'epochs': self.EPOCHS,
            'steps_per_epoch': self.STEPS_PER_EPOCH,
            'validation_steps': self.VALIDATION_STEPS,
            'augmentation': self.AUGMENTATION,
            'num_classes': self.NUM_CLASSES,
            'layers': self.LAYERS,
            'train_rois_per_image': self.TRAIN_ROIS_PER_IMAGE,
            'max_gt_instances': self.MAX_GT_INSTANCES,
            'learning_rate': self.LEARNING_RATE,
            'learning_momentum': self.LEARNING_MOMENTUM,
            'weight_decay': self.WEIGHT_DECAY,
            'rpn_class_loss': self.LOSS_WEIGHTS['rpn_class_loss'],
            'rpn_bbox_loss': self.LOSS_WEIGHTS['rpn_bbox_loss'],
            'mrcnn_class_loss': self.LOSS_WEIGHTS['mrcnn_class_loss'],
            'mrcnn_bbox_loss': self.LOSS_WEIGHTS['mrcnn_bbox_loss'],
            'mrcnn_mask_loss': self.LOSS_WEIGHTS['mrcnn_mask_loss'],

            'training_time': self.TRAIN_TIME,
            'detecting_time_per_image': None,
            'mean_average_precision': None,
            'mean_precisions': None,
            'mean_recalls': None,
            'mean_overlaps': None
        })
        self.CONFIG_DATAFRAME = config_series.to_frame().transpose()

    def read_yaml(self, filePath):
        stream = open(filePath, 'r')
        data = yaml.load(stream, yaml.Loader)
        stream.close()
        return data


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the custom dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes according to the number of classes required to detect
        for num, className in self.classes.items():
            self.add_class(self.name, num, className)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        annotations = list(annotations.values())

        # Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get polygons and image size, accounting for different annotation types
            try:  # custom annotation format, similar to VIA 2.0 and contains size data
                polygons = [r['shape_attributes'] for r in a['regions']]
                custom = [s['region_attributes'] for s in a['regions']]

                # Get size
                height, width = a['image_attributes']['height'], a['image_attributes']['width']

            except TypeError:  # VIA 1.0 format
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                custom = [s['region_attributes'] for s in a['regions'].values()]

                # Get size
                image_path = os.path.join(dataset_dir, a['filename'])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
            
            num_ids = []
            # Add the classes according to the requirement
            classNames = list(self.classes.values())
            classesByName = {v: k for k, v in self.classes.items()}
            for n in custom:
                try:
                    if n['label'] in classNames:
                        num_ids.append(classesByName[n['label']])
                except:
                    pass

            self.add_image(
                self.name,
                image_id=a['filename'],  # use file name as a unique image id
                path=os.path.join(dataset_dir, a['filename']),
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)

    def load_from_config(self, config):
        self.name = config.NAME
        self.classes = config.CLASSES

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a custom dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != self.name:
            return super(self.__class__, self).load_mask(image_id)
        num_ids = image_info['num_ids']	
        #print("Here is the numID",num_ids)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        num_ids = np.array(num_ids, dtype=np.int32)	
        return mask, num_ids#.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32), 

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == self.name:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    name = ''
    classes = {}


############################################################
#  Training
############################################################


def train(custom):
    """Train the model."""
    # Get current contents of log directory
    if os.path.exists(LOGS_DIR):
        logsContents = os.listdir(LOGS_DIR)
    else:
        logsContents = []

    # Display instance of custom class
    custom.display()

    name = custom.NAME
    classes = custom.CLASSES
    weights = custom.WEIGHTS
    dataset_dir = DATASET_DIR if custom.DATASET_DIR == 'dataset' else custom.DATASET_DIR

    # Create model based on custom config
    model = modellib.MaskRCNN(mode="training", config=custom, model_dir=LOGS_DIR)

    # Retrieve weights
    if weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = weights
    custom.WEIGHTS = weights_path

    # Load weights
    print("Loading weights ", weights_path)
    if weights.lower() == "coco":
        # Exclude the last layers because they require a matching number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_from_config(custom)
    dataset_train.load_custom(dataset_dir, "train")
    dataset_train.prepare()
    custom.TRAIN_DATASET = dataset_train

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_from_config(custom)
    dataset_val.load_custom(dataset_dir, "val")
    dataset_val.prepare()
    custom.VAL_DATASET = dataset_val

    # Image Augmentation
    augmentation = custom.AUGMENTATION
    if augmentation:
        aug = iaa.Sometimes(5 / 6, iaa.OneOf([
            iaa.Fliplr(1),
            iaa.Flipud(1),
            iaa.Affine(rotate=(-45, 45)),
            iaa.Affine(rotate=(-90, 90)),
            iaa.Affine(scale=(0.5, 1.5))
        ]))
    else:
        aug = {}

    # Get start time
    start = datetime.now()

    # Train
    print("Training model")
    model.train(dataset_train, dataset_val,
                learning_rate=custom.LEARNING_RATE,
                epochs=custom.EPOCHS,
                layers=custom.LAYERS,
                augmentation=aug,
                custom_callbacks=[CSVLogger(os.path.join(LOGS_DIR, f'log.csv'),
                                            append=True, separator=';')])

    # Get end time
    stop = datetime.now()

    # Subtract times to get time elapsed
    time_difference = stop - start
    custom.TRAIN_TIME = time_difference.total_seconds()

    # Get new log directory
    logsContents2 = os.listdir(LOGS_DIR)
    newDirectory = list(set(logsContents2) - set(logsContents))[0]

    # Rename to config.NAME
    if os.path.exists(os.path.join(LOGS_DIR, name)):
        os.rename(os.path.join(LOGS_DIR, name),
                  os.path.join(LOGS_DIR, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'))
    os.rename(os.path.join(LOGS_DIR, newDirectory),
              os.path.join(LOGS_DIR, name))

    # Move logs directory
    os.rename(os.path.join(LOGS_DIR, 'log.csv'),
              os.path.join(LOGS_DIR, f'{name}\\log.csv'))
