import json
import sys
import os
import time
import numpy as np
import cv2

def BGR2YUV420(image_path):
    IMG_SIZE_in = (1440,960) #RGB format

    img_bgr = cv2.imread(image_path)
    img_bgr = cv2.resize(img_bgr,IMG_SIZE_in)

    # convert bgr picture to YUV420 
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)

    y_channel = img_yuv[:, :, 0]
    output = y_channel.reshape((1,1382400))

    return output

def combine_inputs(IMG_PATH = "dataset/ecam.jpeg"):
    img = BGR2YUV420(IMG_PATH)
    # img = np.zeros(MODEL_HEIGHT * MODEL_WIDTH, dtype=np.uint8)
    calib_input = np.zeros((1,3),dtype=np.float32)
    return [img,calib_input]

 
# test script to test preprocess step
if __name__ == "__main__":
    value = combine_inputs()
    print(value[0].shape)
    print(value[1].shape)


