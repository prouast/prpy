# Copyright (c) 2024 Philipp Rouast
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

"""Save the best models"""

import glob
import logging
import numpy as np
import os
import shutil
import tensorflow as tf
from typing import Union, Callable

class Candidate(object):
  """A candidate model with a score"""
  def __init__(
      self,
      score: Union[float, int, np.float32, np.float64, np.int32, np.int64],
      dir: str,
      filename: str
    ):
    """Initialise a new candidate.

    Args:
      score: The score achieved by the candidate
      dir: The directory where the candidate model can be saved
      filename: The filename under which the candidate model can be saved
    """
    assert isinstance(score, (float, int, np.float32, np.float64, np.int32, np.int64))
    assert isinstance(dir, str)
    assert isinstance(filename, str)
    self.score = score
    self.filepath = os.path.join(dir, filename)

class ModelSaver(object):
  """Save the best models to disk"""
  def __init__(
      self,
      dir: str = "checkpoints",
      keep_best: int = 5,
      keep_latest: int = 1,
      save_format: str = "tf",
      save_optimizer: bool = False,
      compare_fn: Callable[[float, float], bool] = lambda x,y: x.score < y.score,
      sort_reverse: bool = False,
      log_fn: Callable[[str], None] = logging.info
    ):
    """Init the ModelSaver
    
    Args:
      dir: The directory where models should be saved
      keep_best: The number of best scoring models to keep
      keep_latest: The number of latest models to keep
      save_format: Model format for saving ['tf' or 'h5']
      save_optimizer: Also save optimizer state?
      compare_fn: Function that compares two scores
      sort_reverse: Reverse sort order?
      log_fn: Function to write logs
    """
    assert isinstance(dir, str)
    assert isinstance(keep_best, int)
    assert isinstance(keep_latest, int)
    assert isinstance(save_format, str)
    assert isinstance(save_optimizer, bool)
    assert callable(compare_fn)
    assert isinstance(sort_reverse, bool)
    assert callable(log_fn)
    self.best_candidates = []
    self.latest_candidates = []
    # The destination directory (make if necessary)
    self.dir = dir
    if not os.path.exists(self.dir):
      os.makedirs(self.dir)
    self.keep_best = keep_best
    self.keep_latest = keep_latest
    self.save_format = save_format
    self.save_optimizer = save_optimizer
    self.compare_fn_best = compare_fn
    self.compare_fn_latest = lambda x,y: x.score > y.score
    self.sort_reverse = sort_reverse
    self.log_fn = log_fn

  def __save(
      self,
      model: tf.keras.Model,
      filepath: str
    ):
    """Save a model to disk.
    
    Args:
      model: The keras model to be saved
      filepath: The filepath to save to
    """
    assert isinstance(model, tf.keras.Model)
    assert isinstance(filepath, str)
    if self.save_format == 'h5': filepath += '.h5'
    # Save model
    if self.save_optimizer:
      model.save(
        filepath=filepath, overwrite=True, include_optimizer=True,
        save_format=self.save_format)
    else:
      model.save_weights(
        filepath=filepath, overwrite=True, save_format=self.save_format)

  def save_keep(
      self,
      model: tf.keras.Model,
      step: Union[int, np.int32, np.int64],
      name: str
    ):
    """Save and keep the given model.

    Args:
      model: The model to be saved and kept
      step: The current training step
      name: The model name
    """
    assert isinstance(model, tf.keras.Model)
    assert isinstance(step, (int, np.int32, np.int64))
    assert isinstance(name, str)
    self.log_fn(f"Saving and keeping model for step {step}")
    filepath = os.path.join(self.dir, name + "_keep_" + str(step))
    self.__save(model=model, filepath=filepath)

  def save_latest(
      self,
      model: tf.keras.Model, 
      step: Union[int, np.int32, np.int64],
      name: str
    ):
    """Save the given model as currently latest.
    
    Args:
      model: The model to be saved as latest
      step: The current training step
      name: The model name
    """
    assert isinstance(model, tf.keras.Model)
    assert isinstance(step, (int, np.int32, np.int64))
    assert isinstance(name, str)
    name = name + "_latest_" + str(step)
    # Use step as score
    candidate = Candidate(score=step, dir=self.dir, filename=name)
    if len(self.latest_candidates) < self.keep_latest \
      or self.compare_fn_latest(candidate, self.latest_candidates[-1]):
      self.log_fn(f"Saving latest model for step {step}")
      # Keep candidate
      self.latest_candidates.append(candidate)
      self.latest_candidates = sorted(
        self.latest_candidates, key=lambda x: x.score, reverse=True)
      # Save candidate
      self.__save(model, filepath=candidate.filepath)
      # Prune candidate
      for candidate in self.latest_candidates[self.keep_latest:]:
        for file in glob.glob(fr"{candidate.filepath}*"):
          if self.save_format == 'tf' and self.save_optimizer:
            shutil.rmtree(file)
          else:
            os.remove(file)
      self.latest_candidates = self.latest_candidates[0:self.keep_latest]

  def save_best(
      self,
      model: tf.keras.Model,
      score: Union[float, np.float32, np.float64],
      step: Union[int, np.int32, np.int64],
      name: str
    ):
    """Save the given model as a candidate for best model.

    Args:
      model: The model to be saved as candidate for best model
      score: The score achieved by the model
      step: The current training step
      name: The name of the model
    """
    assert isinstance(model, tf.keras.Model)
    assert isinstance(score, (float, np.float32, np.float64))
    assert isinstance(step, (int, np.int32, np.int64))
    assert isinstance(name, str)
    self.log_fn(f"Saving model for step {step}")
    name = name + "_best_" + str(step)
    candidate = Candidate(score, dir=self.dir, filename=name)
    if len(self.best_candidates) < self.keep_best \
      or self.compare_fn_best(candidate, self.best_candidates[-1]):
      # Keep candidate
      self.log_fn(f"Keeping model {candidate.filepath} with score {candidate.score:.4f}")
      self.best_candidates.append(candidate)
      self.best_candidates = sorted(
        self.best_candidates, key=lambda x: x.score, reverse=self.sort_reverse)
      # Save candidate
      self.__save(model, filepath=candidate.filepath)
      # Prune candidates
      for candidate in self.best_candidates[self.keep_best:]:
        self.log_fn(f"Removing old model {candidate.filepath} with score {candidate.score:.4f}")
        for file in glob.glob(fr"{candidate.filepath}*"):
          if self.save_format == 'tf' and self.save_optimizer:
            shutil.rmtree(file)
          else:
            os.remove(file)
      self.best_candidates = self.best_candidates[0:self.keep_best]
    else:
      # Skip the candidate
      self.log_fn(f"Skipping candidate {candidate.filepath}")
