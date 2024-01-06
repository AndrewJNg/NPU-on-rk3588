# rknn library for orange pi chip
from rknnlite.api import RKNNLite

# import preprocess file
import preprocess_dmonitoring as pre

if __name__ == '__main__':
    # Initialise NPU with model
    rknn = RKNNLite(verbose=False)
    rknn.load_rknn("model/dmonitoring_model.rknn")
    ret = rknn.init_runtime()
    
    # feed inputs to the model, run and print out it's raw outputs
    input_data = pre.combine_inputs(IMG_PATH = "dataset/ecam.jpeg")
    outputs = rknn.inference(inputs=input_data, data_format=None)
    print(outputs)

    rknn.release()


