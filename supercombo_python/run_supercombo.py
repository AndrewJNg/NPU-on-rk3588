# rknn library for orange pi chip
from rknnlite.api import RKNNLite 

# import preprocess file
import preprocess_supercombo as pre

if __name__ == '__main__':
    # Initialise NPU with model
    rknn = RKNNLite(verbose=False)
    rknn.load_rknn("model/supercombo.rknn")
    ret = rknn.init_runtime()

    # feed inputs to the model, run and print out it's raw outputs
    input_data = pre.combine_inputs(IMG_PATH_narrow1 = "dataset/narrow1.png" , 
                                    IMG_PATH_narrow2 = "dataset/narrow2.png", 
                                    IMG_PATH_wide1 = "dataset/wide1.png",
                                    IMG_PATH_wide2 = "dataset/wide2.png")
    outputs = rknn.inference(inputs=input_data, data_format=None)
    print(outputs)

    rknn.release()


