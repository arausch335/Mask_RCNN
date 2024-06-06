from Mask_RCNN.trainCustomModel.train import train, CustomConfig
from Mask_RCNN.trainCustomModel.detect import detect
# from Mask_RCNN.trainCustomModel.compareConfigurations import evaluate
from processDataset import *
from pathVariables import *
from time import sleep

if __name__ == '__main__':
    configurationFiles = [
        # os.path.join(CONFIG_DIR, 'heart_300s30e500v_aug.yaml'),
        # os.path.join(CONFIG_DIR, 'heart_300s30e500v.yaml'),
        os.path.join(CONFIG_DIR, 'test2.yaml'),
        # os.path.join(CONFIG_DIR, 'heart_800s80e100v_1.5mml.yaml'),
        # os.path.join(CONFIG_DIR, 'heart_800s80e100v_5mml.yaml')
    ]

    for configuration in configurationFiles:
        # Create instance of configuration class
        config = CustomConfig()
        config.load_from_yaml(configuration)

        # Process data into a format able to be used by the program
        # processDataset(config)
        # sleep(10)
        #
        # # Train the model
        # train(config)
        # sleep(30)

        # Perform detections on dataset
        detect(config)



