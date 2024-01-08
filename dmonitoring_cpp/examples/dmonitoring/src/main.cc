

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "rknn_api.h"
#include "preprocess_dmonitoring.h"

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
  std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
  for (int i = 1; i < attr->n_dims; ++i)
  {
    shape_str += ", " + std::to_string(attr->dims[i]);
  }

  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
         "type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
         attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
  unsigned char *data;
  int ret;

  data = NULL;

  if (NULL == fp)return NULL;

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0)
  {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char *)malloc(sz);
  if (data == NULL)
  {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
  FILE *fp;
  unsigned char *data;

  fp = fopen(filename, "rb");
  if (NULL == fp)
  {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);
  data = load_data(fp, 0, size);
  fclose(fp);

  *model_size = size;
  return data;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char **argv)
{
  if (argc < 3) // show how to use this command
  {
    printf("Usage: %s <rknn model> <input_image_path> <resize/letterbox> <output_image_path>\n", argv[0]);
    return -1;
  }

  int ret;
  char *model_name = (char *)argv[1];
  char *input_path = argv[2];

  // Initialising the model
  rknn_context ctx = 0;
  int model_len = 0;
  int model_data_size = 0;

  unsigned char* model = load_model(model_name,  &model_data_size);
  ret = rknn_init(&ctx, model, model_data_size, 0, NULL);
  
  // get input and output shape
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  /////////////////////////////////////////////////////////////////////////////
  // Model attributes

  // get model input attributes
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));

  for (int i = 0; i < io_num.n_input; i++)
  {
    input_attrs[i].index = i;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    dump_tensor_attr(&(input_attrs[i]));
  }

  //get model output attributes
  // rknn_tensor_attr output_attrs[io_num.n_output];
  // memset(output_attrs, 0, sizeof(output_attrs));
  // for (int i = 0; i < io_num.n_output; i++)
  // {
  //   output_attrs[i].index = i;
  //   rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
  //   dump_tensor_attr(&(output_attrs[i]));
  // }
  
  /////////////////////////////////////////////////////////////////////////////
  // Fill inputs
  rknn_input inputs[io_num.n_input];
  memset(inputs, 0, sizeof(inputs));  //preallocate 

  // Sample input, and print their shapes
  std::vector<cv::Mat> value = combine_inputs("dataset/ecam.jpeg");
   printf("---------\n" );
   printf("Input shape\n" );
  std::cout << value[0].size() << std::endl;
  std::cout << value[1].size() << std::endl;
   printf("---------\n" );
  printf("%s\n\n",get_type_string(input_attrs[0].type));

  inputs[0].index = 0; // img
  inputs[0].type  = RKNN_TENSOR_UINT8;  // TODO - figure out FP16 transform to RKNN_TENSOR format (dmonitoring model is tensor float32)
  inputs[0].size  = input_attrs[0].n_elems; 
  inputs[0].fmt   = RKNN_TENSOR_NHWC;
  inputs[0].buf   = value[0].data;
  
  inputs[1].index = 1; // calib
  inputs[1].type  = RKNN_TENSOR_UINT8;
  inputs[1].size  = input_attrs[1].n_elems; 
  inputs[1].fmt   = RKNN_TENSOR_NHWC;
  inputs[1].buf   = value[1].data;
  
  rknn_inputs_set(ctx, io_num.n_input, inputs);

  /////////////////////////////////////////////////////////////////////////////
  // Running the model
  rknn_run(ctx, nullptr);
  
  /////////////////////////////////////////////////////////////////////////////
  // Getting output
  rknn_output outputs[1];
  memset(outputs, 0, sizeof(outputs));
  outputs[0].want_float = 1;
  ret = rknn_outputs_get(ctx, 1, outputs, NULL);

  /////////////////////////////////////////////////////////////////////////////
  // Post Process
  for (int i = 0; i < io_num.n_output; i++) {
    float*   buffer = (float*)outputs[i].buf;
    uint32_t sz     = outputs[i].size / 4;

    printf(" --- output ---\n ");
    for (int i = 0; i < sz ; i++) {
      printf("%3f ", buffer[i]);
    }
  }
  // Release rknn_outputs
  rknn_outputs_release(ctx, 1, outputs);

  // Release
  if (ctx > 0) rknn_destroy(ctx);
  if (model) free(model);



  return 0;
}