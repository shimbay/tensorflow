#include <algorithm>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <string>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main() {
  std::string model_path =
      "/home/sunyunbo/data/model/model/mobilenet_v1_1.0_224.tflite";
  std::string image_path = "";

  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_path.data());
  TFLITE_MINIMAL_CHECK(model != nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("== Pre-invoke Interpreter State ==\n");
  tflite::PrintInterpreterState(interpreter.get());

  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("== Post-invoke interpreter State\n");
  tflite::PrintInterpreterState(interpreter.get());

  return 0;
}
