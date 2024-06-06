import os
import sys

import pandas as pd
import skimage.io
from matplotlib import pyplot as plt
from numpy import mean
from datetime import datetime, date

# Root directory of the project
MASK_RCNN_DIR = os.path.abspath("../../")
try:
    assert 'Mask_RCNN' in str(MASK_RCNN_DIR)
except AssertionError:
    MASK_RCNN_DIR = os.path.abspath('./')

# import files from parent directories
sys.path.append(MASK_RCNN_DIR)
from pathVariables import *
import mrcnn.model as modellib
from mrcnn.utils import compute_ap
from mrcnn import visualize

sys.path.append(CUSTOM_DIR)
from train import CustomConfig, CustomDataset


# evaluate_model is used to calculate mean Average Precision of the model
def evaluate_model(dataset, model, cfg, subset):
    APs, precisions, recalls, overlaps, times = [], [], [], [], []

    # Get start time
    start = datetime.now()

    for image_id in dataset.image_ids:
        # Get ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id)

        # Run the model on the image to get detected results
        results = model.detect([image], verbose=0)
        r = results[0]

        # Only work with highest score prediction


        # Compare expected vs. ground truth
        AP, precision, recall, overlap = compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                    r["rois"], r["class_ids"], r["scores"], r['masks'])
        # print(AP, precision, recall, overlap)
        # if len(overlap) > 1:
        #     visualize.display_differences(image, gt_bbox, gt_class_id, gt_mask,
        #                                     r['rois'], r['class_ids'], r['scores'], r['masks'],
        #                                     dataset.class_names)
        #     print()

        # Plot various metrics that show model's accuracy
        # visualize.plot_precision_recall(AP, precision, recall)
        # plt.show()
        # visualize.display_differences(image, gt_bbox, gt_class_id, gt_mask,
        #                               r['rois'], r['class_ids'], r['scores'], r['masks'],
        #                               dataset.class_names)

        APs.append(AP)
        precisions.append(mean(precision))
        recalls.append(mean(recall))
        overlaps.append(mean(overlap))
        # print(image_id, AP, precision, recall, overlap)

    # Get end time
    stop = datetime.now()

    # Subtract times to get time elapsed
    time_difference = stop - start
    seconds = time_difference.total_seconds()

    # Compute time per image
    time_per_image = round(seconds / len(dataset.image_ids), 5)

    # Compute means of metrics
    mAP, mPre, mRe, mOv = mean(APs), mean(precisions), mean(recalls), mean(overlaps)

    # Add values to config.DATAFRAME
    cfg.create_dataframe()
    df = cfg.CONFIG_DATAFRAME
    df.at[0, 'subset'] = subset
    df.at[0, 'detecting_time_per_image'] = time_per_image
    df.at[0, 'mean_average_precision'] = mAP
    df.at[0, 'mean_precisions'] = mPre
    df.at[0, 'mean_recalls'] = mRe
    df.at[0, 'mean_overlaps'] = mOv

    # Open compareConfig.csv
    compare_df_path = os.path.join(CONFIG_DIR, 'compareConfigurations.csv')
    if os.path.exists(compare_df_path):
        compare_df = pd.read_csv(compare_df_path)
    else:
        compare_df = pd.DataFrame(columns=df.columns)

    # Merge dataframes
    full_df = pd.concat([compare_df, df], ignore_index=True)
    full_df.to_csv(compare_df_path, index=False)

    return mAP, mPre, mRe, mOv


def detect(customConfig, weights='last', evaluate=False):
    # Edit parameters for inference
    class InferenceConfig(CustomConfig):
        # Set batch size to 1 since we'll be running inference on one image at a time.
        # Batch size = GPU_COUNT * IMAGES_PER_GPU
        def load_inference(self):
            self.GPU_COUNT = 1
            self.IMAGES_PER_GPU = 1
            self.BATCH_SIZE = 1
            self.USE_MINI_MASK = False
            self.DETECTION_MIN_CONFIDENCE = 0.98
            self.MAX_GT_INSTANCES = 1

            try:
                self.TRAIN_TIME = customConfig.TRAIN_TIME
            except AttributeError:
                self.TRAIN_TIME = None

    inference_config = InferenceConfig()
    inference_config.load_from_yaml(customConfig.CONFIG_FILE)
    inference_config.load_inference()
    inference_config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=LOGS_DIR, config=inference_config)

    # Load weights from last model
    if weights == 'last':
        weights = model.find_last()
    elif weights == 'config':
        weights = customConfig.WEIGHTS
    model.keras_model.load_weights(weights, by_name=True)

    # Get classes from config file
    class_names = ['BG']
    class_names.extend(customConfig.CLASSES.values())

    if customConfig.EVALUATE:
        # Prepare datasets
        try:
            dataset_train = customConfig.TRAIN_DATASET
        except AttributeError:
            # Create new instance of training dataset.
            dataset_train = CustomDataset()
            dataset_train.load_from_config(inference_config)
            dataset_train.load_custom(DATASET_DIR, 'train')
        dataset_train.prepare()

        # Evaluate model on train dataset
        train_mAP = evaluate_model(dataset_train, model, inference_config, 'train')
        print(f'Train mAP: {train_mAP[0]}')
        print(f'Train mOv: {train_mAP[-1]}')

        # Prepare validation dataset
        try:
            dataset_val = customConfig.VAL_DATASET
        except AttributeError:
            # Validation dataset
            dataset_val = CustomDataset()
            dataset_val.load_from_config(inference_config)
            dataset_val.load_custom(DATASET_DIR, 'val')
        dataset_val.prepare()

        # Evaluate model on val dataset
        val_mAP = evaluate_model(dataset_val, model, inference_config, 'val')
        print(f'Val mAP: {val_mAP[0]}')
        print(f'Val mOv: {val_mAP[-1]}')

    else:
        # Get names of all files in images directory
        file_names = os.listdir(IMAGE_DIR)

        # Check to see if runs directory exists
        RUNS_DIR = os.path.join(LOGS_DIR, f"{customConfig.NAME}\\runs")
        if not os.path.exists(RUNS_DIR):
            os.makedirs(RUNS_DIR, exist_ok=True)

            for file in file_names:
                print(file)
                image = skimage.io.imread(os.path.join(IMAGE_DIR, file))

                # Run detection
                results = model.detect([image], verbose=1)

                # Visualize results
                r = results[0]
                visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                            class_names, r['scores'],
                                            display=False, savePath=os.path.join(RUNS_DIR, file))


