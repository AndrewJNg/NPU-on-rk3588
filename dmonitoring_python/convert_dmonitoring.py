import numpy as np
from rknn.api import RKNN

# import preprocess file
import preprocess_dmonitoring as pre

ONNX_MODEL = 'model/dmonitoring.onnx' # input model
RKNN_MODEL = 'model/dmonitoring.rknn' # output model

# load onnx model and build rknn model from it
if __name__ == '__main__':
    rknn = RKNN(verbose=False)

    # pre-process config
    print('--> Config model')
    rknn.config(target_platform='rk3588') 
    print('Config model done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx( model=ONNX_MODEL )   #, inputs=['inputs', 'buffer'], input_size_list=[[514],[27724] ], outputs=['out', 'out_buffer'])  
    
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('Loading model done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('Building model done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('Export rknn model done')

    ####################################################################################
    # initialise test environement and give test data to mimic application board output
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    
    # feed inputs to the model, run and print out it's raw outputs
    input_data = pre.combine_inputs(IMG_PATH = "dataset/ecam.jpeg")
    outputs = rknn.inference(inputs=input_data, data_format=None)
    print(outputs)

    # save results
    np.save('result.npy', outputs)



    rknn.release()
