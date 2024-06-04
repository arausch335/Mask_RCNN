import os
import sys
import skimage.io

# Root directory of the project
MASK_RCNN_DIR = os.path.abspath("../../")
try:
    assert 'Mask_RCNN' in str(MASK_RCNN_DIR)
except AssertionError:
    MASK_RCNN_DIR = os.path.abspath('./')

sys.path.append(MASK_RCNN_DIR)
from pathVariables import *
import mrcnn.model as modellib
from mrcnn import visualize

sys.path.append(CUSTOM_DIR)
from train import CustomConfig


def detect(customConfig, weights='last'):
    class InferenceConfig(CustomConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        def load_inference(self):
            self.GPU_COUNT = 1
            self.IMAGES_PER_GPU = 1
            self.BATCH_SIZE = 1

    config = InferenceConfig()
    config.load_from_yaml(customConfig)
    config.load_inference()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=LOGS_DIR, config=config)

    # Load weights trained on MS-COCO
    if weights == 'last':
        weights = model.find_last()
    model.keras_model.load_weights(weights, by_name=True)

    # Get classes
    class_names = ['BG']
    class_names.extend(config.CLASSES.values())

    # Load a random image from the images folder
    file_names = os.listdir(IMAGE_DIR)

    for file in file_names:
        print(file)
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file))

        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        RUNS_DIR = os.path.join(LOGS_DIR, f"{config.NAME}\\runs")
        os.makedirs(RUNS_DIR, exist_ok=True)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'],
                                    display=False, savePath=os.path.join(RUNS_DIR, file))


