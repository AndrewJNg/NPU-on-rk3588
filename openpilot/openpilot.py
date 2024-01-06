import itertools
import os
import sys
import numpy as np
from typing import Tuple, Dict, Union, Any

# from openpilot.selfdrive.modeld.runners.runmodel_pyx import RunModel  #(TODO add 1)
from rknnlite.api import RKNNLite

ORT_TYPES_TO_NP_TYPES = {'tensor(float16)': np.float16, 'tensor(float)': np.float32, 'tensor(uint8)': np.uint8}


# class RKNNModel(RunModel): #(TODO add 2)
class RKNNModel():
  def __init__(self, path, output, runtime, use_tf8, cl_context): # runtime and cl_context are not used in this case
    self.inputs = {}
    self.output = output
    self.use_tf8 = use_tf8

    ################################################################
    #(TODO add 3)
    self.input_names = ['input_img','calib']

    self.input_shapes = {
    'input_img': [1, 1382400],
    'calib': [1, 3]}

    self.input_dtypes = {
    'input_img': np.float32,
    'calib': np.float32}
    
    ### TODO current problem with getting the input names, shape and dtypes needed by the rknn model (through python)

    # self.session = create_rknn_session(path, fp16_to_fp32=True)
    # self.input_names = [x.name for x in self.session.get_inputs()]
    # self.input_shapes = {x.name: [1, *x.shape[1:]] for x in self.session.get_inputs()}
    # self.input_dtypes = {x.name: ORT_TYPES_TO_NP_TYPES[x.type] for x in self.session.get_inputs()}

    ################################################################
    # initialise NPU on Rockchip
    self.rknn = RKNNLite(verbose=False)
    self.rknn.load_rknn(path)
    self.rknn.init_runtime()
    ################################################################
    

  def addInput(self, name, buffer):
    assert name in self.input_names
    self.inputs[name] = buffer

  def setInputBuffer(self, name, buffer):
    assert name in self.inputs
    self.inputs[name] = buffer

  def getCLBuffer(self, name):
    return None

  def execute(self):
    # input shaping and formatting
    inputs = {k: (v.view(np.uint8) / 255. if self.use_tf8 and k == 'input_img' else v) for k,v in self.inputs.items()}
    inputs = {k: v.reshape(self.input_shapes[k]).astype(self.input_dtypes[k]) for k,v in inputs.items()}

    # running inputs through model
    outputs = self.rknn.inference(inputs=[inputs[input_name] for input_name in self.input_names], data_format=None)

    # check that the output is valid
    assert len(outputs) == 1, "Only single model outputs are supported"
    self.output[:] = outputs[0]
    return self.output

################################################################################################################################
## dmonitoring test script (not used in normal operation)

if __name__ == '__main__':
  CALIB_LEN = 3
  REG_SCALE = 0.25
  MODEL_WIDTH = 1440
  MODEL_HEIGHT = 960
  OUTPUT_SIZE = 84

  # initialise input and output size, and start NPU chip
  output = np.zeros(OUTPUT_SIZE, dtype=np.float32)
  inputs = {
    'input_img': np.zeros(MODEL_HEIGHT * MODEL_WIDTH, dtype=np.uint8),
    'calib': np.zeros(CALIB_LEN, dtype=np.float32)}
    
  model = RKNNModel(path = 'model/dmonitoring_model.rknn', output = output, runtime = 0, use_tf8 = True, cl_context= None)
  
  # prefill all values with zeros 
  model.addInput("input_img", None)
  model.addInput("calib", inputs['calib'])

  ################################################################
  ## def run (execution stage)
  # run model with all zeros inputs
  inputs['calib'][:] = np.zeros(3,dtype=np.float32)
  model.setInputBuffer("input_img", inputs['input_img'].view(np.float32))
  print(model.execute())

  

