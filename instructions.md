Here are the instructions to use this repository:
- Set up the environment using conda and the provided yaml file
  - conda env update -n (environment_name) --file Mask_RCNN\environment.yml
- Download ms coco weights from https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
- Annotate data (I used CVAT) and export either MS-COCO or VIA JSON file
- Upload media and annotations to trainingData directory
- Create new YAML configuration file in trainCustomModel/configurations with training parameters
- Add the new config file to handler.py
- Run handler.py