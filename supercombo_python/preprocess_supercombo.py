import json
import sys
import os
import time
import numpy as np
import cv2

def BGR2YUV420(image_path):
    IMG_SIZE_in = (512,256) #RGB format

    img_bgr = cv2.imread(image_path)
    img_bgr = cv2.resize(img_bgr,IMG_SIZE_in)

    # convert bgr picture to YUV420 
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    y_channel = img_yuv[:, :, 0]
    u_channel = img_yuv[:, :, 1]
    v_channel = img_yuv[:, :, 2]


    # preallocate output format
    output = np.zeros((128, 256, 6), dtype=np.uint8)

    # Extract Y, U, and V channels (format is 6 * 128 * 256)
    '''
    Channels 0,1,2,3 represent the full-res Y channel and are represented in numpy as Y[::2, ::2], Y[::2, 1::2], Y[1::2, ::2], and Y[1::2, 1::2]
    Channel 4 represents the half-res U channel
    Channel 5 represents the half-res V channel
    '''
    output[:, :, 0] = y_channel[::2, ::2]
    output[:, :, 1] = y_channel[::2, 1::2]
    output[:, :, 2] = y_channel[1::2, ::2]
    output[:, :, 3] = y_channel[1::2, 1::2]

    output[:, :, 4] = u_channel[::2, ::2]  # Use half of U channel directly
    output[:, :, 5] = v_channel[::2, ::2]  # Use half of V channel directly

    # print(output.shape)
    return output

def combine_inputs(IMG_PATH_narrow1 = "narrow1.png" , 
                   IMG_PATH_narrow2 = "narrow2.png", 
                   IMG_PATH_wide1 = "wide1.png",
                   IMG_PATH_wide2 = "wide2.png"):
    
    # print(BGR2YUV420(IMG_PATH_narrow1).shape)
    # print(BGR2YUV420(IMG_PATH_narrow2).shape)

    # print(BGR2YUV420(IMG_PATH_wide1).shape)
    # print(BGR2YUV420(IMG_PATH_wide2).shape)

    '''
    TODO 
    unsure how we want to contatenate the input images for supercombo model
    '''

    input_imgs = np.concatenate((BGR2YUV420(IMG_PATH_narrow1), BGR2YUV420(IMG_PATH_narrow2)) ,axis=2)
    # input_imgs = input_imgs.reshape(1, 128, 256,12)
    input_imgs = input_imgs.reshape(1, 12, 128, 256)
    input_imgs = np.transpose(input_imgs, (0, 2, 3, 1))

    big_input_imgs = np.concatenate((BGR2YUV420(IMG_PATH_wide1), BGR2YUV420(IMG_PATH_wide2)) ,axis=2)
    # big_input_imgs = big_input_imgs.reshape(1, 128, 256,12)
    big_input_imgs = big_input_imgs.reshape(1, 12, 128, 256)
    big_input_imgs = np.transpose(big_input_imgs, (0, 2, 3, 1))


    desire = np.zeros((1,100, 8), dtype=np.float16)
    traffic_convention = np.array([[1,0]], dtype=np.float16) # [1,2] signify right hand drive and left hand drive respectively
    lat_planner_state = np.zeros((1,4), dtype=np.float16)

    nav_features = np.zeros((1,256), dtype=np.float16)
    nav_instructions = np.zeros((1,150), dtype=np.float16)
    features_buffer = np.zeros((1,99,512), dtype=np.float16)
    

    # print(desire.shape)
    # print(traffic_convention.shape)
    # print(lat_planner_state.shape)

    # print(nav_features.shape)
    # print(nav_instructions.shape)
    # print(features_buffer.shape)

    return [input_imgs, big_input_imgs, desire, traffic_convention, lat_planner_state, nav_features ,nav_instructions, features_buffer]

 
"""
**Neutron results**
Model inputs
1) 2 images of narrow angle camera, 20hz apart (1,12,128,256)
2) 2 images of wide angle camera, 20hz apart (1,12,128,256)

3) desire (1,100,8)
4) traffic_convention (1,2)
5) lat_planner_state (1,4)
6) nav_features (1,256)
7) nav_instructions (1,150)
8) features_buffer (1,99,512)



Model output
1x6768


"""


# test script to test preprocess step
if __name__ == "__main__":
    value = np.zeros(8) 
    value = combine_inputs()

    print(value[0].shape)
    print(value[1].shape)
    print(value[2].shape)
    print(value[3].shape)
    print(value[4].shape)
    print(value[5].shape)
    print(value[6].shape)
    print(value[7].shape)




