// Copyright (c) 2024 Philipp Rouast

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>

#define DIV_ROUND_CLOSEST(n, d) ((((n) < 0) == ((d) < 0)) ? (((n) + (d)/2)/(d)) : (((n) - (d)/2)/(d)))

static PyObject*
resample_bilinear_op(PyObject* self, PyObject* args) {
  PyArrayObject* input_array;
  PyObject* size_tuple;
  int new_height, new_width;

  // Parse the arguments
  if (!PyArg_ParseTuple(args, "O!O", &PyArray_Type, &input_array, &size_tuple)) {
    return NULL;
  }

  // Convert the size tuple to new_height and new_width
  if (!PyArg_ParseTuple(size_tuple, "ii", &new_height, &new_width)) {
    return NULL;
  }

  // Check input array type and dimensions
  if (PyArray_NDIM(input_array) != 4 || PyArray_TYPE(input_array) != NPY_UINT8 || PyArray_DIM(input_array, 3) != 3) {
      PyErr_SetString(PyExc_ValueError, "Input array must be 4-dimensional with shape (n, h, w, 3) and of type uint8.");
    return NULL;
  }

  // Get pointers to the data
  unsigned char* input_data = (unsigned char*)PyArray_DATA(input_array);

  // Get the dimensions of the input array
  npy_intp* dims = PyArray_DIMS(input_array);
  int n_frames = (int)dims[0];
  int height = (int)dims[1];
  int width = (int)dims[2];

  // Get the strides
  npy_intp* input_strides = PyArray_STRIDES(input_array);

  // Allocate memory for the positions and weights
  int* pos_flat = malloc(new_width * new_height * 2 * sizeof(int));
  float* weights_flat = malloc(new_width * new_height * 2 * 2 * sizeof(float));
  if (!pos_flat || !weights_flat) {
    PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for position and weights arrays");
    free(pos_flat);
    free(weights_flat);
    return NULL;
  }

  // Precompute the positions and weights
  for (int y = 0; y < new_height; ++y) {
    for (int x = 0; x < new_width; ++x) {
      // Determine the position of the current pixel in the input image
      float src_x = (x * (width - 1)) / (float)(new_width - 1);
      float src_y = (y * (height - 1)) / (float)(new_height - 1);
      // The four neighboring pixels
      int x0 = (int)src_x;
      int y0 = (int)src_y;
      // Calculate the weights for each pixel
      float dx = src_x - x0;
      float dy = src_y - y0;
      float dx1 = 1.0f - dx;
      float dy1 = 1.0f - dy;
      // Store the positions
      int flat_idx = (x * new_height + y) * 2; 
      pos_flat[flat_idx + 0] = x0;
      pos_flat[flat_idx + 1] = y0;
      // Store the weights
      weights_flat[(flat_idx + 0) * 2 + 0] = dx1 * dy1 * 256.0f;
      weights_flat[(flat_idx + 0) * 2 + 1] = dx  * dy1 * 256.0f;
      weights_flat[(flat_idx + 1) * 2 + 0] = dx1 * dy  * 256.0f;
      weights_flat[(flat_idx + 1) * 2 + 1] = dx  * dy  * 256.0f;
    }
  }

  // Create an output array
  npy_intp output_dims[4] = {n_frames, new_height, new_width, 3};
  PyArrayObject* output_array_np = (PyArrayObject*)PyArray_EMPTY(4, output_dims, NPY_UINT8, 0);
  if (output_array_np == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Unable to allocate output array");
    Py_DECREF(output_array_np);
    return NULL;
  }

  // Get a pointer to the output data
  unsigned char* output_data = (unsigned char*)PyArray_DATA(output_array_np);

  // Resample each frame
  for (int i = 0; i < n_frames; ++i) {
    // Extract the current frame
    unsigned char* current_frame = input_data + i * input_strides[0];
    // Resize the frame using bilinear resampling
    for (int y = 0; y < new_height; ++y) {
      for (int x = 0; x < new_width; ++x) {
        // Get position coordinates
        int pos_idx = (x * new_height + y) * 2;
        int x0 = pos_flat[pos_idx + 0];
        int y0 = pos_flat[pos_idx + 1];
        
        // Get weights
        int weights_idx = ((x * new_height + y) * 2) * 2;
        float w00 = weights_flat[weights_idx + 0];
        float w01 = weights_flat[weights_idx + 1];
        float w10 = weights_flat[weights_idx + 2];
        float w11 = weights_flat[weights_idx + 3];

        // Access pixels
        unsigned char* p1 = current_frame + y0 * input_strides[1] + x0 * input_strides[2];
        unsigned char* p2 = p1 + input_strides[2];
        unsigned char* p3 = p1 + input_strides[1];
        unsigned char* p4 = p3 + input_strides[2];

        // Make sure that p2 and p4 are within bounds
        if (x0 + 1 >= width) {
          p2 = p1;
          p4 = p3;
        }
        // Make sure that p3 and p4 are within bounds
        if (y0 + 1 >= height) {
          p3 = p1;
          p4 = p2;
        }
        
        // Calculate the weighted sum of pixels (for each color channel)
        int outr = p1[0] * w00 + p2[0] * w01 + p3[0] * w10 + p4[0] * w11;
        int outg = p1[1] * w00 + p2[1] * w01 + p3[1] * w10 + p4[1] * w11;
        int outb = p1[2] * w00 + p2[2] * w01 + p3[2] * w10 + p4[2] * w11;

        // Save
        int idx = i * new_height * new_width * 3 + y * new_width * 3 + x * 3;
        output_data[idx + 0] = (unsigned char)(outr >> 8);
        output_data[idx + 1] = (unsigned char)(outg >> 8);
        output_data[idx + 2] = (unsigned char)(outb >> 8);
      }
    }
  }

  // Free allocated memory
  free(pos_flat);
  free(weights_flat);

  return PyArray_Return(output_array_np);
}

static PyObject*
resample_box_op(PyObject* self, PyObject* args) {
  PyArrayObject* input_array;
  PyObject* size_tuple;
  int new_height, new_width;

  // Parse the arguments
  if (!PyArg_ParseTuple(args, "O!O", &PyArray_Type, &input_array, &size_tuple)) {
    return NULL;
  }

  // Convert the size tuple to new_height and new_width
  if (!PyArg_ParseTuple(size_tuple, "ii", &new_height, &new_width)) {
    return NULL;
  }

  // Check input array type and dimensions
  if (PyArray_NDIM(input_array) != 4 || PyArray_TYPE(input_array) != NPY_UINT8 || PyArray_DIM(input_array, 3) != 3) {
    PyErr_SetString(PyExc_ValueError, "Input array must be 4-dimensional with shape (n, h, w, 3) and of type uint8.");
    return NULL;
  }

  // Get pointers to the data
  unsigned char* input_data = (unsigned char*)PyArray_DATA(input_array);

  // Get the dimensions of the input array
  npy_intp* dims = PyArray_DIMS(input_array);
  int n_frames = (int)dims[0];
  int height = (int)dims[1];
  int width = (int)dims[2];

  // Make sure that we are downsampling
  if (height / new_height < 2 || width / new_width < 2) {
    PyErr_SetString(PyExc_ValueError, "This implementation of box resampling can only be used for downsampling.");
    return NULL;
  }

  // Get the strides
  npy_intp* input_strides = PyArray_STRIDES(input_array);

  // Create an output array
  npy_intp output_dims[4] = {n_frames, new_height, new_width, 3};
  PyArrayObject* output_array_np = (PyArrayObject*)PyArray_EMPTY(4, output_dims, NPY_UINT8, 0);
  if (output_array_np == NULL) {
    Py_DECREF(output_array_np);
    PyErr_SetString(PyExc_MemoryError, "Unable to allocate output array");
    return NULL;
  }

  // Get a pointer to the output data
  unsigned char* output_data = (unsigned char*)PyArray_DATA(output_array_np);

  // Allocate memory for start and end indices for rows and columns and num_pixels
  int* start_x = (int*)malloc(new_width * sizeof(int));
  int* end_x = (int*)malloc(new_width * sizeof(int));
  int* start_y = (int*)malloc(new_height * sizeof(int));
  int* end_y = (int*)malloc(new_height * sizeof(int));
  int* num_pixels_flat = malloc(new_width * new_height * sizeof(int));
  if (!start_x || !end_x || !start_y || !end_y || !num_pixels_flat) {
    // Clean up allocated memory before returning error
    free(start_x);
    free(end_x);
    free(start_y);
    free(end_y);
    free(num_pixels_flat);
    PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for indices");
    Py_DECREF(output_array_np);
    return NULL;
  }

  // Precompute start and end indices for rows and columns and num_pixels
  for (int x = 0; x < new_width; ++x) {
    start_x[x] = DIV_ROUND_CLOSEST(x * width, new_width);
    end_x[x] = DIV_ROUND_CLOSEST((x + 1) * width, new_width);
  }
  for (int y = 0; y < new_height; ++y) {
    start_y[y] = DIV_ROUND_CLOSEST(y * height, new_height);
    end_y[y] = DIV_ROUND_CLOSEST((y + 1) * height, new_height);
    for (int x = 0; x < new_width; ++x) {
      num_pixels_flat[y * new_width + x] = (end_y[y] - start_y[y]) * (end_x[x] - start_x[x]);
    }
  }

  // Resample each frame
  for (int i = 0; i < n_frames; ++i) {
    // Extract the current frame
    unsigned char* current_frame = input_data + i * input_strides[0];
    // Resize the frame using box resampling (average pooling)
    for (int y = 0; y < new_height; ++y) {
      for (int x = 0; x < new_width; ++x) {
        // Compute the mean of pixels in the input image block
        int sum_r = 0, sum_g = 0, sum_b = 0;
        int num_pixels = num_pixels_flat[y * new_width + x];
        for (int src_y = start_y[y]; src_y < end_y[y]; ++src_y) {
          for (int src_x = start_x[x]; src_x < end_x[x]; ++src_x) {
            unsigned char* pixel = current_frame + src_y * input_strides[1] + src_x * input_strides[2];
            sum_r += pixel[0];
            sum_g += pixel[1];
            sum_b += pixel[2];
          }
        }
        // Calculate the average value of the pixels
        unsigned char avg_r = (unsigned char)(sum_r / num_pixels);
        unsigned char avg_g = (unsigned char)(sum_g / num_pixels);
        unsigned char avg_b = (unsigned char)(sum_b / num_pixels);
        // Save the average value to the output image
        int idx = i * new_height * new_width * 3 + y * new_width * 3 + x * 3;
        output_data[idx + 0] = avg_r;
        output_data[idx + 1] = avg_g;
        output_data[idx + 2] = avg_b;
      }
    }
  }

  // Free allocated memory
  free(start_x);
  free(end_x);
  free(start_y);
  free(end_y);
  free(num_pixels_flat);

  return PyArray_Return(output_array_np);
}

static PyObject*
reduce_roi_op(PyObject* self, PyObject* args) {
  PyArrayObject* input_array_video;
  PyArrayObject* input_array_roi;

  // Parse the arguments
  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &input_array_video, &PyArray_Type, &input_array_roi)) {
    return NULL;
  }

  // Check input video array type and dimensions
  if (PyArray_NDIM(input_array_video) != 4 || PyArray_TYPE(input_array_video) != NPY_UINT8 || PyArray_DIM(input_array_video, 3) != 3) {
    PyErr_SetString(PyExc_ValueError, "Input video array must be 4-dimensional with shape (n, h, w, 3) and of type uint8.");
    return NULL;
  }

  // Check input roi array type and dimensions
  if (PyArray_NDIM(input_array_roi) != 2 || PyArray_TYPE(input_array_roi) != NPY_INT64 || PyArray_DIM(input_array_roi, 1) != 4 || PyArray_DIM(input_array_roi, 0) != PyArray_DIM(input_array_video, 0)) {
    PyErr_SetString(PyExc_ValueError, "Input roi array must be 2-dimensional with shape (n, 4) and of type int64. First dim must match first dim of video array.");
    return NULL;
  }

  // Get pointers to the data
  unsigned char* video_data = (unsigned char*)PyArray_DATA(input_array_video);
  npy_int64* roi_data = (npy_int64*)PyArray_DATA(input_array_roi);

  // Get the strides
  npy_intp* video_strides = PyArray_STRIDES(input_array_video);
  // npy_intp* roi_strides = PyArray_STRIDES(input_array_roi);

  // Get the dimensions of the input video array
  npy_intp* video_dims = PyArray_DIMS(input_array_video);
  int n_frames = (int)video_dims[0];

  // Create an output array
  npy_intp output_dims[2] = {n_frames, 3};
  PyArrayObject* output_array_np = (PyArrayObject*)PyArray_EMPTY(2, output_dims, NPY_FLOAT32, 0);
  if (output_array_np == NULL) {
    Py_DECREF(output_array_np);
    return NULL;
  }
  // Get a pointer to the output data
  float* output_data = (float*)PyArray_DATA(output_array_np);
  
  // Reduce each frame
  for (int i = 0; i < n_frames; ++i) {
    // Extract the current frame
    unsigned char* current_frame_video = video_data + i * video_strides[0];
    npy_int64* current_frame_roi = roi_data + i * 4; // TODO: Why doesn't roi_strides[0] work?
    // Compute the mean of pixels within the ROI
    int sum_r = 0, sum_g = 0, sum_b = 0;
    int x0 = current_frame_roi[0];
    int y0 = current_frame_roi[1];
    int x1 = current_frame_roi[2];
    int y1 = current_frame_roi[3];
    for (int src_y = y0; src_y < y1; ++src_y) {
      for (int src_x = x0; src_x < x1; ++src_x) {
        unsigned char* pixel = current_frame_video + src_y * video_strides[1] + src_x * video_strides[2];
        sum_r += pixel[0];
        sum_g += pixel[1];
        sum_b += pixel[2];
      }
    }
    // Calculate the average value of the pixels
    int num_pixels = (y1 - y0) * (x1 - x0);
    float avg_r = sum_r / (float)num_pixels;
    float avg_g = sum_g / (float)num_pixels;
    float avg_b = sum_b / (float)num_pixels;
    // Save the average value to the output image
    int idx = i * 3;
    output_data[idx + 0] = avg_r;
    output_data[idx + 1] = avg_g;
    output_data[idx + 2] = avg_b;
  }

  return PyArray_Return(output_array_np);
}

static PyMethodDef image_ops_methods[] = {
    {"resample_bilinear_op", resample_bilinear_op, METH_VARARGS, PyDoc_STR("Resize video frames using bilinear resampling.")},
    {"resample_box_op", resample_box_op, METH_VARARGS, PyDoc_STR("Resize video frames using box resampling.")},
    {"reduce_roi_op", reduce_roi_op, METH_VARARGS, PyDoc_STR("Reduce spatial dimensions of video by mean using ROI.")},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyDoc_STRVAR(image_ops_doc,
"Image ops implemented in C.");

static struct PyModuleDef image_ops = {
  PyModuleDef_HEAD_INIT,
  "image_ops",
  image_ops_doc,
  -1,
  image_ops_methods
};

PyMODINIT_FUNC
PyInit_image_ops(void) {
  import_array();
  return PyModule_Create(&image_ops);
}
