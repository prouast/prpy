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

#include <Python.h>
#include <numpy/arrayobject.h>

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
  int n_frames = dims[0];
  int height = dims[1];
  int width = dims[2];

  // Get the strides
  npy_intp* input_strides = PyArray_STRIDES(input_array);

  // Precompute the positions and weights
  int pos[new_width][new_height][2];
  int weights[new_width][new_height][2][2];
  for (int y = 0; y < new_height; ++y) {
    for (int x = 0; x < new_width; ++x) {
      // Determine the position of the current pixel in the input image
      float src_x = (x * (width - 1)) / (float)(new_width - 1);
      float src_y = (y * (height - 1)) / (float)(new_height - 1);
      // The four neighboring pixels
      int x0 = (int)src_x;
      int y0 = (int)src_y;
      int x1 = x0 + 1;
      int y1 = y0 + 1;
      // Calculate the weights for each pixel
      float dx = src_x - x0;
      float dy = src_y - y0;
      float dx1 = 1.0f - dx;
      float dy1 = 1.0f - dy;
      // Store the positions
      pos[x][y][0] = x0;
      pos[x][y][1] = y0;
      // Store the weights
      weights[x][y][0][0] = dx1 * dy1 * 256.0f;
      weights[x][y][0][1] = dx  * dy1 * 256.0f;
      weights[x][y][1][0] = dx1 * dy  * 256.0f;
      weights[x][y][1][1] = dx  * dy  * 256.0f;
    }
  }

  // Create an output array
  npy_intp output_dims[4] = {n_frames, new_height, new_width, 3};
  PyArrayObject* output_array_np = (PyArrayObject*)PyArray_EMPTY(4, output_dims, NPY_UINT8, 0);
  if (output_array_np == NULL) {
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
        unsigned char* p1 = current_frame + pos[x][y][1] * input_strides[1] + pos[x][y][0] * input_strides[2];
        unsigned char* p2 = p1 + input_strides[2];
        unsigned char* p3 = p1 + input_strides[1];
        unsigned char* p4 = p3 + input_strides[2];
        
        // Calculate the weighted sum of pixels (for each color channel)
        int outr = p1[0] * weights[x][y][0][0] + p2[0] * weights[x][y][0][1] + 
                   p3[0] * weights[x][y][1][0] + p4[0] * weights[x][y][1][1];
        int outg = p1[1] * weights[x][y][0][0] + p2[1] * weights[x][y][0][1] + 
                   p3[1] * weights[x][y][1][0] + p4[1] * weights[x][y][1][1];
        int outb = p1[2] * weights[x][y][0][0] + p2[2] * weights[x][y][0][1] + 
                   p3[2] * weights[x][y][1][0] + p4[2] * weights[x][y][1][1];
        // Save
        int idx = i * new_height * new_width * 3 + y * new_width * 3 + x * 3;
        output_data[idx + 0] = (unsigned char)(outr >> 8);
        output_data[idx + 1] = (unsigned char)(outg >> 8);
        output_data[idx + 2] = (unsigned char)(outb >> 8);
      }
    }
  }

  Py_INCREF(output_array_np);
  return (PyObject*)output_array_np;
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
  int n_frames = dims[0];
  int height = dims[1];
  int width = dims[2];

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
    return NULL;
  }

  // Get a pointer to the output data
  unsigned char* output_data = (unsigned char*)PyArray_DATA(output_array_np);

  // Precompute start and end indices for rows and columns and num_pixels
  int start_x[new_width];
  int end_x[new_width];
  int start_y[new_height];
  int end_y[new_height];
  int num_pixels_[new_width][new_height];
  for (int x = 0; x < new_width; ++x) {
    start_x[x] = DIV_ROUND_CLOSEST(x * width, new_width);
    end_x[x] = DIV_ROUND_CLOSEST((x + 1) * width, new_width);
  }
  for (int y = 0; y < new_height; ++y) {
    start_y[y] = DIV_ROUND_CLOSEST(y * height, new_height);
    end_y[y] = DIV_ROUND_CLOSEST((y + 1) * height, new_height);
    for (int x = 0; x < new_width; ++x) {
      num_pixels_[x][y] = (end_y[y] - start_y[y]) * (end_x[x] - start_x[x]);
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
        for (int src_y = start_y[y]; src_y < end_y[y]; ++src_y) {
          for (int src_x = start_x[x]; src_x < end_x[x]; ++src_x) {
            unsigned char* pixel = current_frame + src_y * input_strides[1] + src_x * input_strides[2];
            sum_r += pixel[0];
            sum_g += pixel[1];
            sum_b += pixel[2];
          }
        }
        // Calculate the average value of the pixels
        int num_pixels = num_pixels_[x][y];
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

  Py_INCREF(output_array_np);
  return (PyObject*)output_array_np;
}

static PyMethodDef image_ops_methods[] = {
    {"resample_bilinear_op", resample_bilinear_op, METH_VARARGS, PyDoc_STR("Resize video frames using bilinear resampling.")},
    {"resample_box_op", resample_box_op, METH_VARARGS, PyDoc_STR("Resize video frames using box resampling.")},
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
