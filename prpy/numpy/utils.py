# Copyright (c) 2024 Rouast Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import reduce
import numpy as np
import operator
import psutil

def enough_memory_for_ndarray(
    shape: tuple,
    dtype: np.dtype,
    max_fraction_of_available_memory_to_use: float = 0.7
  ) -> bool:
  """Check if there is enough virtual memory available to store an ndarray.
  
  Args:
    shape: The shape of the ndarray.
    dtype: The data type of the ndarray.
    max_fraction_of_available_memory_to_use: The maximum fraction of available memory to use.
  Returns:
    bool: True if there is enough memory available, False otherwise.
  """
  total_elements = reduce(operator.mul, shape)
  element_size = np.dtype(dtype).itemsize
  required_bytes = total_elements * element_size
  available_memory_bytes = psutil.virtual_memory().available
  return max_fraction_of_available_memory_to_use * available_memory_bytes > required_bytes
