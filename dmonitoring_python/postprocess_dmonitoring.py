import numpy as np

def sigmoid (input):
  return 1 / (1 + np.exp(-input))


def get_result(outputs):
  t1 = 0 #time millis_since_boot();
  t2 = 0 #time millis_since_boot();


  result ={
    "face_orientation": outputs[0][:3],
    "face_orientation_meta": outputs[0][3:6],
    "face_position": outputs[0][6:8],
    "face_position_meta": outputs[0][8:10],

    "face_prob": sigmoid(outputs[0][12]),
    "left_eye_prob": sigmoid(outputs[0][21]),
    "right_eye_prob": sigmoid(outputs[0][12]),
    "left_blink_prob": sigmoid(outputs[0][13]),
    "right_blink_prob": sigmoid(outputs[0][14]),
    "sg_prob": sigmoid(outputs[0][15]),
    "poor_vision": sigmoid(outputs[0][16]),
    "partial_face": sigmoid(outputs[0][17]),
    "distracted_pose": sigmoid(outputs[0][18]),
    "distracted_eyes": sigmoid(outputs[0][19]),
    "occluded_prob": sigmoid(outputs[0][20]),
    "ready_prob": sigmoid(outputs[0][21:25]),
    "not_ready_prob": sigmoid(outputs[0][25:27]),
    "dsp_execution_time": (t2 - t1) / 1000.
}

  return result


# test script to test preprocess step
if __name__ == "__main__":
    sample_output=  np.array([[ 5.890625, 2.3046875, 0.328125, 0.03125, -1.859375, 1.59375, -7.0, -7.0, -7.0, -7.0,
                         -7.0, -7.0, 6.0, -7.0, 6.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0, 6.0, 6.0, -7.0, 6.0,
                         6.0, 6.0, -7.0, -7.0, 5.2421875, 6.0, -7.0, -7.0, -7.0, -0.25390625, 0.14453125, -0.109375,
                         0.3828125, -0.05859375, -1.1835938, 0.0703125, -2.8710938, -2.8515625, -1.4648438,
                         0.23828125, -0.25, 0.296875, -0.72265625, -2.5664062, 0.30078125, -7.0, -7.0, -7.0,
                         2.7578125, 6.0, -7.0, 6.0, -7.0, -7.0, -3.015625, -7.0, 6.0, 6.0, -7.0, -7.0, 6.0, -7.0,
                         -6.09375, -7.0, 6.0, 2.421875, 6.0, -7.0, -7.0, -7.0, -0.07421875, -0.01171875, 0.06640625,
                         0.26171875, -0.12890625, -4.2421875, 0.078125, -7.0, -7.0]], dtype=np.float32)
    
    for key, value in get_result(sample_output).items():
        print(key, ":", value)


