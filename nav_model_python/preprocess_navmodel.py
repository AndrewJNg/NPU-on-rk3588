import json
import sys
import os
import time
import numpy as np
import cv2

def BGR2GRAY(image_path):
    IMG_SIZE_in = (256,256) #RGB format

    img_bgr = cv2.imread(image_path)
    img_bgr = cv2.resize(img_bgr,IMG_SIZE_in)

    # convert bgr picture to greyscale
    img_grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)/255

    output = img_grey
    return output

def combine_inputs(IMG_PATH = "dataset/map.png"):
    img = BGR2GRAY(IMG_PATH)

    output = img.reshape((1, 256, 256, 1)).astype(np.float32)
    # print(output)
    print(output.max())
    # cv2.imshow('image', output [0, :, :, 0])
    # cv2.waitKey(0)
    
    
    return [output]

 
# test script to test preprocess step
if __name__ == "__main__":
    value = combine_inputs()
    print(value.shape)


