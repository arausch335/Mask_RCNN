from Mask_RCNN.trainCustomModel.train import train
from Mask_RCNN.trainCustomModel.detect import detect
from processDataset import *
from pathVariables import *
import os
from time import sleep

if __name__ == '__main__':
    configurationFiles = [
        os.path.join(CONFIG_DIR, 'test2.yaml'),
        # os.path.join(CONFIG_DIR, 'heart_800s80e100v_1.5mml.yaml'),
        # os.path.join(CONFIG_DIR, 'heart_800s80e100v_5mml.yaml')
    ]

    for configuration in configurationFiles:
        processDataset(configuration)
        sleep(10)
        train(configuration)
        sleep(30)
        detect(configuration)



