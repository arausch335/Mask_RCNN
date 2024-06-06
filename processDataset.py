import os.path
from trainCustomModel.train import CustomConfig
from pathVariables import *
import json
import shutil
from datetime import datetime
import cv2
import random


def getFrames(videoDirPath, dstDirPath):
    for video in list(filter(lambda x: ('.mp4' in x), os.listdir(videoDirPath))):
        vidcap = cv2.VideoCapture(os.path.join(videoDirPath, video))
        success, image = vidcap.read()

        count = 0
        while success:
            cv2.imwrite(os.path.join(dstDirPath, f'frame_{str(count).zfill(6)}.PNG'), image)
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1
        print(f'Video {video} successfully converted to images')


def COCO_to_VIA(ANNOTATION_PATH):
    # ANNOTATION_PATH = os.path.join(MASK_RCNN_DIR, 'instances_default.json')
    annotation_file = open(ANNOTATION_PATH)
    annotations = json.load(annotation_file)

    data = annotations.copy()
    del data['licenses'], data['info']

    categories = data['categories']
    categoryDict = {}
    for category in categories:
        categoryDict[category['id']] = category['name']

    annotations = data['annotations']
    annotation_ids = [int(x['image_id']) for x in annotations]
    annotated_images = []

    images = data['images']
    imagesByID = {}
    for i in range(0, len(images)):
        del data['images'][i]['license']
        del data['images'][i]['flickr_url']
        del data['images'][i]['coco_url']
        del data['images'][i]['date_captured']

        image = images[i]
        id = image['id']
        imagesByID[id] = image

        if id in annotation_ids:
            annotated_images.append(data['images'][i])

    data['images'] = annotated_images

    new_annotations = {}
    for i in range(len(annotations)):
        annotation = annotations[i]
        image_id = annotation['image_id']
        label = categoryDict[annotation['category_id']]

        image = imagesByID[image_id]
        fileName = image['file_name']
        height = image['height']
        width = image['width']

        segmentation = annotation['segmentation']  # x, y, x, y
        allPointsX = segmentation[0][::2]
        allPointsX.append(allPointsX[0])
        allPointsX = [int(x) if x % int(x) == 0 else x for x in allPointsX]
        allPointsY = segmentation[0][1::2]
        allPointsY.append(allPointsY[0])
        allPointsY = [int(x) if x % int(x) == 0 else x for x in allPointsY]

        if fileName in new_annotations.keys():
            new_annotations[fileName]['regions'].append({
                'shape_attributes': {
                    'name': 'polygon',
                    'all_points_x': allPointsX,
                    'all_points_y': allPointsY},
                'region_attributes': {
                    'label': label}}
            )
        else:
            new_annotations[fileName] = {
                'filename': fileName,
                'regions': [{
                    'shape_attributes': {
                        'name': 'polygon',
                        'all_points_x': allPointsX,
                        'all_points_y': allPointsY},
                    'region_attributes': {
                        'label': label}
                    }],
                'image_attributes': {
                    'height': height,
                    'width': width
                }
            }

    data['annotations'] = new_annotations

    # Serialize data and write back to file
    with open(os.path.join(os.path.dirname(ANNOTATION_PATH), 'annotations.json'), 'w') as f:
        json.dump(new_annotations, f)

    print(f'COCO JSON file {os.path.basename(ANNOTATION_PATH)} converted to VIA JSON format')

    return os.path.join(os.path.dirname(ANNOTATION_PATH), 'annotations.json')


def processDataset(custom, split=(85, 15), get_frames=False, replace=False):
    # Check to see if dataset exists, and replace is False
    if os.path.exists(DATASET_DIR) and replace is False:
        print('Processed dataset already exists\n\n')
        return

    print('Processing Data\n')

    # Read preprocessed data and output data paths from config
    dataDir = os.path.join(MASK_RCNN_DIR, 'trainingData') if custom.PREPROCESSED_DATA_DIR == 'trainingData' \
        else custom.PREPROCESSED_DATA_DIR
    outputDir = CUSTOM_DIR if custom.DATASET_DIR == 'dataset' else custom.DATASET_DIR

    # Convert frames of videos (.mp4) to images (.PNG) if necessary
    videosDir = os.path.join(dataDir, 'videos')
    imagesDir = os.path.join(dataDir, 'images')
    os.makedirs(imagesDir, exist_ok=True)
    images = os.listdir(imagesDir)
    if len(images) == 0 or get_frames is True:
        getFrames(videosDir, imagesDir)
        images = os.listdir(imagesDir)

    # Convert CVAT-exported COCO format to VIA format if necessary
    annotationPathCOCO = os.path.join(dataDir, r'annotations\instances_default.json')
    annotationPathVIA = os.path.join(dataDir, r'annotations\annotations.json')
    if os.path.exists(annotationPathVIA):
        annotations = json.load(open(annotationPathVIA))
    else:
        annotationPath = COCO_to_VIA(annotationPathCOCO)
        annotations = json.load(open(annotationPath))

    # Split annotated images into train and validation
    annotatedImages = [os.path.basename(x) for x in annotations.keys()]
    random.shuffle(annotatedImages)
    splitIndex = int(round((split[0]/100) * len(annotatedImages), 0))
    trainImages, valImages = annotatedImages[:splitIndex], annotatedImages[splitIndex:]

    # Create dataset directory tree
    tempDatasetDir = os.path.join(dataDir, r'dataset')
    trainDir = os.path.join(tempDatasetDir, r'train')
    valDir = os.path.join(tempDatasetDir, r'val')
    testDir = os.path.join(tempDatasetDir, r'test')
    os.makedirs(trainDir, exist_ok=True)
    os.makedirs(valDir, exist_ok=True)
    os.makedirs(testDir, exist_ok=True)

    # Add images to corresponding directories
    for image in trainImages:
        shutil.copy(os.path.join(imagesDir, image), os.path.join(trainDir, image))
    for image in valImages:
        shutil.copy(os.path.join(imagesDir, image), os.path.join(valDir, image))
    for image in images:
        if image not in annotatedImages:
            shutil.copy(os.path.join(imagesDir, image), os.path.join(testDir, image))

    # Split up annotation file into train and validate
    trainAnnotations = {}
    valAnnotations = {}
    for fileName, annotation in annotations.items():
        if fileName in trainImages:
            trainAnnotations[fileName] = annotation
        elif fileName in valImages:
            valAnnotations[fileName] = annotation
        else:
            print('Something is wrong.')

    # Add annotations to corresponding directories
    with open(os.path.join(trainDir, 'annotations.json'), 'w') as f:
        json.dump(trainAnnotations, f)
    with open(os.path.join(valDir, 'annotations.json'), 'w') as f:
        json.dump(valAnnotations, f)

    # Move dataset to proper location
    if os.path.exists(os.path.join(outputDir, 'dataset')):
        os.rename(os.path.join(outputDir, 'dataset'),
                  os.path.join(outputDir, f'dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}'))
    shutil.move(tempDatasetDir, outputDir)

    print('Processed dataset created\n\n')
